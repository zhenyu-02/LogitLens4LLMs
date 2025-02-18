import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
from activation_analyzer import ActivationAnalyzer, PredictionStep

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,) + output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, lm_head, norm):
        super().__init__()
        self.block = block
        self.lm_head = lm_head
        self.norm = norm
        
        # Wrap attention module
        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        # Store intermediate activations
        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None

    def forward(self, x, past_key_value=None, attention_mask=None, position_ids=None, **kwargs):
        output = self.block(x, past_key_value=past_key_value, attention_mask=attention_mask, 
                          position_ids=position_ids, **kwargs)
        
        # Get hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Get attention output and compute intermediate activations
        attn_output = self.block.self_attn.activations
        self.attn_mech_output_unembedded = self.lm_head(self.norm(attn_output))
        
        # Add residual connection
        attn_output += x
        self.intermediate_res_unembedded = self.lm_head(self.norm(attn_output))
        
        # MLP output
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_output_unembedded = self.lm_head(self.norm(mlp_output))
        
        # Block output
        self.block_output_unembedded = self.lm_head(self.norm(hidden_states))

        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None

    def get_attn_activations(self):
        return self.block.self_attn.activations

class QwenHelper:
    def __init__(self, use_local: bool = True, local_path: str = "./explanation/models_hf", token: str = None):
        print("Initializing Qwen Helper...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        
        print("Loading tokenizer...")
        try:
            if use_local:
                self.tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token, use_fast=True)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
        print("Tokenizer loaded successfully")
        
        print("Loading model...")
        try:
            if use_local:
                self.model = AutoModelForCausalLM.from_pretrained(local_path).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=token).to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
        print("Model loaded successfully")
        
        # Wrap all layers
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, 
                self.model.lm_head, 
                self.model.model.norm
            )
        print("All layers wrapped successfully")

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(
            inputs.input_ids.to(self.device), 
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_ids.to(self.device)).logits
            return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(tokens, probs_percent)))

    def collect_decoded_activations(self, decoded_activations: torch.Tensor, topk: int = 10) -> List[Tuple[str, int]]:
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent))

    def decode_all_layers_to_dict(
        self, 
        text: str, 
        topk: int = 10, 
        collect_attn_mech: bool = True,
        collect_intermediate_res: bool = True, 
        collect_mlp: bool = True,
        collect_block: bool = True
    ) -> Dict[int, Dict[str, List[Tuple[str, int]]]]:
        self.get_logits(text)
        all_layers_data = {}
        
        for i, layer in enumerate(self.model.model.layers):
            layer_data = {}
            
            if collect_attn_mech:
                layer_data['attention_mechanism'] = self.collect_decoded_activations(
                    layer.attn_mech_output_unembedded, topk=topk)
            
            if collect_intermediate_res:
                layer_data['intermediate_residual'] = self.collect_decoded_activations(
                    layer.intermediate_res_unembedded, topk=topk)
            
            if collect_mlp:
                layer_data['mlp_output'] = self.collect_decoded_activations(
                    layer.mlp_output_unembedded, topk=topk)
            
            if collect_block:
                layer_data['block_output'] = self.collect_decoded_activations(
                    layer.block_output_unembedded, topk=topk)
            
            all_layers_data[i] = layer_data
            
        return all_layers_data

    def decode_all_layers(self, text, topk=10, print_attn_mech=True, 
                         print_intermediate_res=True, print_mlp=True, print_block=True):
        self.get_logits(text)
        for i, layer in enumerate(self.model.model.layers):
            print(f'Layer {i}: Decoded intermediate outputs')
            if print_attn_mech:
                self.print_decoded_activations(layer.attn_mech_output_unembedded, 
                                            'Attention mechanism', topk=topk)
            if print_intermediate_res:
                self.print_decoded_activations(layer.intermediate_res_unembedded, 
                                            'Intermediate residual stream', topk=topk)
            if print_mlp:
                self.print_decoded_activations(layer.mlp_output_unembedded, 
                                            'MLP output', topk=topk)
            if print_block:
                self.print_decoded_activations(layer.block_output_unembedded, 
                                            'Block output', topk=topk)

    def generate_with_probing(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
        topk: int = 10,
        threshold: int = 3,
        print_details: bool = True
    ) -> List[PredictionStep]:
        """
        Generate text and record the prediction process for each token
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            topk: Number of top predictions to save for each layer
            threshold: Threshold for important layers
            print_details: Whether to print detailed prediction information
            
        Returns:
            List[PredictionStep]: List of prediction steps
        """
        prediction_steps = []
        current_text = prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        for step_idx in range(max_new_tokens):
            # Get layer activation data for current step
            all_layers_data = self.decode_all_layers_to_dict(current_text, topk=topk)
            
            # Generate next token
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top_p sampling
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                next_token_id = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
            
            # Decode generated token
            predicted_token = self.tokenizer.decode(next_token_id[0])
            
            # Get important layers data
            important_layers = ActivationAnalyzer.filter_important_layers(all_layers_data, threshold=threshold)
            
            # Record step data
            step_data = {
                "step_idx": step_idx,
                "input_text": current_text,
                "predicted_token": predicted_token,
                "all_layers_data": all_layers_data,
                "important_layers": important_layers
            }
            prediction_steps.append(step_data)
            
            # Update input
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            current_text += predicted_token
            
            # Print current step information
            if print_details:
                print(f"\nStep {step_idx + 1}: Predicted token: {predicted_token}")
                print(f"Current text: {current_text}")
                print("\nImportant layers for this prediction:")
                for layer_idx, components in important_layers.items():
                    print(f"\nLayer {layer_idx}:")
                    for component_name, tokens_probs in components.items():
                        top_preds = ActivationAnalyzer.get_top_predictions(components, top_k=5)
                        print(f"  {component_name}: {top_preds[component_name]}")
            else:
                print(f"Step {step_idx + 1}: Generated '{predicted_token}'")
        
        return prediction_steps 