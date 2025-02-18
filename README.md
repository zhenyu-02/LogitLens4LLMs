# LogitLens4LLMs

[中文文档](README_zh.md)

LogitLens4LLMs is a toolkit for applying Logit Lens to modern large language models (LLMs). It supports layer-wise analysis of hidden states and predictions, currently compatible with Llama-3.1-8B and Qwen-2.5-7B models.

## Features

- **Layer-wise Analysis**: Analyze model's hidden states and predictions layer by layer using Logit Lens technique.
- **Multiple Model Support**: Currently supports Llama-3.1-8B and Qwen-2.5-7B models, with more to come.
- **Visualization Output**: Generates heatmaps for intuitive visualization of predictions at each layer.
- **Local Model Support**: Load models locally to reduce network dependency.

## Project Structure

```
LogitLens4LLMs/
├── README.md
├── README_zh.md
├── requirements.txt
├── main.py                    # Main program entry
├── model_factory.py           # Model factory class
├── activation_analyzer.py     # Activation analyzer
├── model_helper/             # Model helper classes
│   ├── llama_2_helper.py     # Llama-2 model helper
│   ├── llama_3_1_helper.py   # Llama-3.1 model helper
│   └── qwen_helper.py        # Qwen model helper
├── colab_notebook/          # Colab example notebooks
│   ├── logit_lens_Llama_3_1_8b.ipynb  # Llama-3.1-8B example
│   └── logit_lens_Qwen_7b.ipynb       # Qwen-7B example
└── output/                   # Output directory
    └── visualizations/       # Visualization results
```

## Quick Start

### Local Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/zhenyu-02/LogitLens4LLMs
   cd LogitLens4LLMs
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Google Colab

We provide Google Colab notebooks for you to run examples directly in your browser:

- [Llama-3.1-8B Logit Lens Analysis](https://colab.research.google.com/drive/1OcYRKAsVI-me1zmtnhwoQfdyg6y_SVd3?usp=sharing)
- [Qwen-7B Logit Lens Analysis](https://colab.research.google.com/drive/1xpBQSnukiNPka2oQDtLd7sKdQ0gt74PU?usp=sharing)

## Usage Example

Here's a simple example showing how to perform Logit Lens analysis on the Llama-3.1-8B model:

```python
from logit_lens.model_factory import ModelFactory, ModelType

# Initialize model
model_type = ModelType.LLAMA_3_1_8B
model = ModelFactory.create_model(model_type, use_local=False)

# Run Logit Lens analysis
prompt = "Complete this sentence: The cat sat on the"
prediction_steps = model.generate_with_probing(prompt, max_new_tokens=10, print_details=True)

# Output results
for step in prediction_steps:
    print(f"Step {step['step_idx']}: Predicted token: {step['predicted_token']}")
```

### Sample Output
Command line output, automatically saved to local JSON file
```
Running logit lens analysis:
Model type: llama_3_1_8b
Prompt: Complete this sentence: The cat sat on the

Step 1: Predicted token: " mat"
Current text: Complete this sentence: The cat sat on the mat

Important layers for this prediction:

Layer 15:
  attention_mechanism: [('mat', 85), ('floor', 12), ('carpet', 3)]
  mlp_output: [('mat', 82), ('floor', 15), ('rug', 3)]
  block_output: [('mat', 88), ('floor', 10), ('carpet', 2)]

Layer 20:
  attention_mechanism: [('mat', 90), ('floor', 8), ('rug', 2)]
  block_output: [('mat', 92), ('floor', 7), ('carpet', 1)]
```

## Visualization Output

The tool generates two types of heatmaps for each generation step:

1. Important Layers Heatmap - Shows predictions from layers above threshold
2. All Layers Heatmap - Shows predictions from all layers

![ALL Layers Visualization](./output/visualizations/all_layers_step_0.png)

## Supported Models

Currently supported models include:

- **Llama-3.1-8B**
- **Qwen-2.5-7B**
- **Llama-2-7B**

More models will be supported in the future. Contributions are welcome.

## Contributing

Contributions are welcome! Please fork the repository and submit a Pull Request. We'll review and merge your code after review.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the following projects for inspiration and support:

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- [Logit Lens for Llama-2](https://www.lesswrong.com/posts/fJE6tscjGRPnK8C2C/decoding-intermediate-activations-in-llama-2-7b)

## Contact

For any questions or suggestions, please contact me at [zhenyu_wang02@outlook.com](mailto:zhenyu_wang02@outlook.com).

## Citation

If you use this tool in your research, please cite it as:

```bibtex
@software{wang2024logitlens,
  title = {LogitLens4LLMs: A Logit Lens Toolkit for Modern Large Language Models},
  author = {Wang, Zhenyu},
  year = {2025},
  url = {https://github.com/zhenyu-02/LogitLens4LLMs},
  version = {1.0.0},
  date = {2025-02-17}
}
```

Or in text:

```
Wang, Z. (2025). LogitLens4LLMs: A Logit Lens Toolkit for Modern Large Language Models (Version 1.0.0) [Computer software]. https://github.com/zhenyu-02/LogitLens4LLMs
```

---

Happy Coding! 🚀
