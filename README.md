# SciVQR: A Multidisciplinary Multimodal Benchmark for Advanced Scientific Reasoning Evaluation

Longteng Guo*, Xuanxu Lin*, Dongze Hao*, Tongtian Yue, Pengkang Huo, Jiatong Ma, Yuchen Liu, Jing Liu (*Equal Contribution)

<p align="center">
    <a href="https://huggingface.co/datasets/l205/SciVQR"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffd21e.svg" alt="Dataset"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

## 🌕 Abstract
Scientific reasoning is a key aspect of human intelligence, requiring the integration of multimodal inputs, domain expertise, and multi-step inference across various subjects. Existing benchmarks for multimodal large language models (MLLMs) often fail to capture the complexity and traceability of reasoning processes necessary for rigorous evaluation. To fill this gap, we introduce SciVQR, a multimodal benchmark covering 54 subfields in mathematics, physics, chemistry, geography, astronomy, and biology. SciVQR includes domain-specific visuals, such as equations, charts, and diagrams, and challenges models to combine visual comprehension with reasoning. The tasks range from basic factual recall to complex, multi-step inferences, with 46\% including expert-authored solutions. SciVQR not only evaluates final answers but also examines the reasoning process, providing insights into how models reach their conclusions. Our evaluation of leading MLLMs, including both proprietary and open-source models, reveals significant limitations in handling complex multimodal reasoning tasks, underscoring the need for improved multi-step reasoning and better integration of interdisciplinary knowledge in advancing MLLMs toward true scientific intelligence. 

## 🌖 SciVQR Benchmark
SciVQR covers 6 core scientific domains: **Mathematics, Physics, Chemistry, Geography, Astronomy, and Biology**. It challenges models to integrate fine-grained visual understanding, deep subject knowledge, and sophisticated reasoning.

## 🌗 Dataset (Hosted on Hugging Face)
**Note: This repository contains only the evaluation code. The full dataset is hosted on Hugging Face.**

The dataset contains 3,254 multimodal questions, with 46% accompanied by detailed, expert-authored solution traces. You can easily load the dataset using the `datasets` library:

```python
from datasets import load_dataset
dataset = load_dataset("l205/SciVQR", split="train")  
```

## 🌘 Evaluation Pipeline
We provide three core scripts to comprehensively evaluate MLLMs on the SciVQR benchmark:

* `evaluate_multichoice.py`: For rule-based and symbolic equivalence evaluation of multiple-choice questions.
* `evaluate_open.py`: For evaluating open-ended free-form questions using an LLM-as-a-judge approach.
* `evaluate_reasoning.py`: For fine-grained evaluation of CoT reasoning quality across 5 dimensions (Faithfulness, Informativeness, Redundancy, Hallucination, Missing Steps).
