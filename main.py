import sys
import os
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings  # Import warnings module
from enum import Enum
from explanation.logit_lens.model_factory import ModelFactory, ModelType
from activation_analyzer import ActivationAnalyzer
from prompt_templates import TaskConfig, Concept, CrossDomainAnalogyConfig, PromptTemplate

class ModelPath(Enum):
    """Enumeration of local model paths"""
    DEEPSEEK_LLAMA_8B = "/root/autodl-fs/models_hf/DeepSeek-R1-Distill-Llama-8B"
    DEEPSEEK_QWEN_7B = "/root/autodl-fs/models_hf/DeepSeek-R1-Distill-Qwen-7B"
    LLAMA_2_7B = "/root/autodl-fs/models_hf/Llama-2-7b"
    LLAMA_3_1_8B = "/root/autodl-fs/models_hf/Llama-3.1-8b"
    QWEN_7B = "/root/autodl-fs/models_hf/Qwen2.5-7B-Instruct"
    
    @classmethod
    def get_path(cls, model_type: ModelType) -> str:
        """Get local path corresponding to ModelType"""
        path_mapping = {
            ModelType.LLAMA_7B: cls.LLAMA_2_7B.value,
            ModelType.LLAMA_3_1_8B: cls.LLAMA_3_1_8B.value,
            ModelType.QWEN_7B: cls.QWEN_7B.value
        }
        if model_type not in path_mapping:
            raise ValueError(f"No local path found for model type {model_type}")
        return path_mapping[model_type]

# HF_ENDPOINT=https://hf-mirror.com huggingface-cli download meta-llama/Llama-3.1-8b --local-dir /root/LACM/explanation/models_hf

# warnings.filterwarnings("ignore")

def run_analysis(
    model_type: ModelType,
    use_local: bool = False,
    token: str = None,
    prompt: str = "",
    num_trials: int = 5,
    extract_middle_token_num: int = 15,
    print_details: bool = False,
    max_output_new_tokens: int = 10,
    save_output: bool = True,
    output_base_path: str = "./explanation/logit_lens"
):
    """
    Run logit lens analysis on large language models, generate prediction steps and save visualization results.
    
    Args:
        model_type: Type of model to analyze
        use_local: Whether to use locally saved model
        token: HuggingFace token for model access
        prompt: Input text prompt
        num_trials: Number of generation trials to run
        extract_middle_token_num: Number of intermediate tokens to extract
        print_details: Whether to print detailed prediction info
        max_output_new_tokens: Maximum number of new tokens to generate
        save_output: Whether to save visualization results
        output_base_path: Base directory for saving outputs
    """
    if use_local:
        local_path = ModelPath.get_path(model_type)
        
    model = ModelFactory.create_model(
        model_type=model_type,
        use_local=use_local,
        local_path=local_path,
        token=token
    )
    analyzer = ActivationAnalyzer()
    
    print(f"\nRunning logit lens analysis:")
    print(f"Model type: {model_type.value}")
    print(f"Prompt: {prompt}\n")

    for i in range(num_trials):
        print(f"Running trial {i+1}...")
        prediction_steps = model.generate_with_probing(
            prompt=prompt,
            max_new_tokens=max_output_new_tokens,
            temperature=0.7,
            top_p=0.95,
            topk=extract_middle_token_num,
            threshold=3,
            print_details=print_details
        )
        
        print(f"Output: {[prediction_steps[-1]['input_text']]+[prediction_steps[-1]['predicted_token']]}\n")

        if save_output:
            trial_path = f"{output_base_path}/{model_type.value}/trial_{i}"
            
            # Save visualizations
            for step in prediction_steps:
                analyzer.visualize_layer_predictions(
                    max_tokens=extract_middle_token_num,
                    prediction_step=step,
                    output_dir=f"{trial_path}/visualizations",
                    step_idx=step['step_idx'],
                    log_scale=False
                )

            # Save prediction step data
            analyzer.save_prediction_steps(
                prediction_steps=prediction_steps,
                output_dir=trial_path,
                save_all_data=True
            )

def main():
    # Simple test example
    token = input("Please enter your HuggingFace token: ")
    # Test 1: Basic logit lens functionality
    test_prompt = "Complete this sentence: The cat sat on the"
    print("\nRunning basic logit lens test...")
    run_analysis(
        model_type=ModelType.LLAMA_3_1_8B,
        token=token,
        prompt=test_prompt,
        num_trials=1,
        print_details=True,
        save_output=True
    )
 

if __name__ == "__main__":
    main()

# Generating trail 1...
# Step 1: Generated '
# '
# Step 2: Generated '##'
# Step 3: Generated 'Solution'
# Step 4: Generated '
# '
# Step 5: Generated '
# '
# Step 6: Generated 'Answer'
# Step 7: Generated ':'
# Step 8: Generated '['
# Step 9: Generated 'In'
# Step 10: Generated 'fer'
# Trail 1 completed.

# Generating trail 2...
# Step 1: Generated '
# '
# Step 2: Generated '1'
# Step 3: Generated '.'
# Step 4: Generated 'Dim'
# Step 5: Generated 'in'
# Step 6: Generated 'ishing'
# Step 7: Generated 'marg'
# Step 8: Generated 'inal'
# Step 9: Generated 'utility'
# Step 10: Generated '
# '
# Trail 2 completed.

# Generating trail 3...
# Step 1: Generated '
# '
# Step 2: Generated '>'
# Step 3: Generated 'A'
# Step 4: Generated '.'
# Step 5: Generated 'Dim'
# Step 6: Generated 'in'
# Step 7: Generated 'ishing'
# Step 8: Generated 'Marg'
# Step 9: Generated 'inal'
# Step 10: Generated 'Util'
# Trail 3 completed.

# Generating trail 4...
# Step 1: Generated '*'
# Step 2: Generated '['
# Step 3: Generated 'b'
# Step 4: Generated ']'
# Step 5: Generated 'Ind'
# Step 6: Generated 'ividual'
# Step 7: Generated 'demand'
# Step 8: Generated 'curve'
# Step 9: Generated '['
# Step 10: Generated '/'
# Trail 4 completed.

# Generating trail 5...
# Step 1: Generated '
# '
# Step 2: Generated '##'
# Step 3: Generated 'How'
# Step 4: Generated 'to'
# Step 5: Generated 'solve'
# Step 6: Generated '
# '
# Step 7: Generated '
# '
# Step 8: Generated 'The'
# Step 9: Generated 'answers'
# Step 10: Generated 'are'
# Trail 5 completed.