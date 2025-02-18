# LogitLens4LLMs

LogitLens4LLMs 是一个用于现代大语言模型（LLMs）的 Logit Lens 工具包。它支持对模型的隐藏状态和预测进行逐层分析，目前支持 Llama-3.1-8B 和 Qwen-2.5-7B 模型。

## 功能特性

- **逐层分析**：通过 Logit Lens 技术，逐层分析模型的隐藏状态和预测结果。
- **多模型支持**：目前支持 Llama-3.1-8B 和 Qwen-2.5-7B 模型，未来将支持更多模型。
- **可视化输出**：生成热力图，直观展示每一层的预测结果。
- **本地模型支持**：支持从本地加载模型，减少对网络的依赖。

## 文件结构

```
LogitLens4LLMs/
├── README.md
├── README_zh.md
├── requirements.txt
├── main.py                    # 主程序入口
├── model_factory.py           # 模型工厂类
├── activation_analyzer.py     # 激活值分析器
├── model_helper/             # 模型辅助类
│   ├── llama_2_helper.py     # Llama-2 模型辅助类
│   ├── llama_3_1_helper.py   # Llama-3.1 模型辅助类
│   └── qwen_helper.py        # Qwen 模型辅助类
├── colab_notebook/          # Colab 示例笔记本
│   ├── logit_lens_Llama_3_1_8b.ipynb  # Llama-3.1-8B 示例
│   └── logit_lens_Qwen_7b.ipynb       # Qwen-7B 示例
└── output/                   # 输出目录
    └── visualizations/       # 可视化结果
```

## 快速开始

### 本地运行

1. 克隆本仓库：
   ```bash
   git clone https://github.com/zhenyu-02/LogitLens4LLMs
   cd LogitLens4LLMs
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### Google Colab

我们提供了 Google Colab 笔记本，您可以直接在浏览器中运行示例：

- [Llama-3.1-8B Logit Lens 分析](https://colab.research.google.com/drive/1OcYRKAsVI-me1zmtnhwoQfdyg6y_SVd3?usp=sharing)
- [Qwen-7B Logit Lens 分析](https://colab.research.google.com/drive/1xpBQSnukiNPka2oQDtLd7sKdQ0gt74PU?usp=sharing)


## 使用示例

以下是一个简单的使用示例，展示如何对 Llama-3.1-8B 模型进行 Logit Lens 分析：

```python
from logit_lens.model_factory import ModelFactory, ModelType

# 初始化模型
model_type = ModelType.LLAMA_3_1_8B
model = ModelFactory.create_model(model_type, use_local=False)

# 运行 Logit Lens 分析
prompt = "Complete this sentence: The cat sat on the"
prediction_steps = model.generate_with_probing(prompt, max_new_tokens=10, print_details=True)

# 输出结果
for step in prediction_steps:
    print(f"Step {step['step_idx']}: Predicted token: {step['predicted_token']}")
```

### 输出示例
命令行输出，可自动保存到本地json文件
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

## 可视化输出

该工具会为每个生成步骤生成两种热力图：

1. 重要层热力图 - 展示概率高于阈值的层的预测
2. 全层热力图 - 展示所有层的预测情况

![ALL Layers Visualization](./output/visualizations/all_layers_step_0.png)

## 模型支持

目前支持的模型包括：

- **Llama-3.1-8B**
- **Qwen-2.5-7B**
- **Llama-2-7B**

未来将支持更多模型，欢迎贡献代码。

## 贡献

欢迎贡献代码！请先 fork 本仓库，然后提交 Pull Request。我们会在审核后合并您的代码。

## 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 致谢

感谢以下项目的启发和支持：

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- [Logit Lens for Llama-2](https://www.lesswrong.com/posts/fJE6tscjGRPnK8C2C/decoding-intermediate-activations-in-llama-2-7b)

## 联系方式

如有任何问题或建议，请通过 [zhenyu_wang02@outlook.com](mailto:zhenyu_wang02@outlook.com) 联系我。

## 引用

如果您在研究中使用了本工具，请按以下格式引用：

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

或在文章中引用：

```
Wang, Z. (2025). LogitLens4LLMs: A Logit Lens Toolkit for Modern Large Language Models (Version 1.0.0) [Computer software]. https://github.com/zhenyu-02/LogitLens4LLMs
```

---

Happy Coding! 🚀