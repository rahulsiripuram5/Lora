# LoRA vs. Full Fine-Tuning for Llama 3 8B 🚀

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers%20%7C%20PEFT-yellow)

This project provides a rigorous, hands-on comparison of fine-tuning strategies for large language models (LLMs). It specifically evaluates the trade-offs between performance and computational efficiency by comparing a zero-shot baseline, Parameter-Efficient Fine-Tuning (PEFT) with **LoRA (Low-Rank Adaptation)**, and traditional full fine-tuning on the state-of-the-art **Meta Llama 3 8B Instruct** model.

The primary goal is to demonstrate how LoRA can achieve performance nearly identical to a full fine-tune while training only a tiny fraction of the model's parameters, making it a highly effective and resource-friendly approach for adapting LLMs to specific tasks.

## 📊 Results Summary

The experiment was conducted on the `databricks/databricks-dolly-15k` dataset. The key metric, **perplexity** (a measure of how well a model predicts a sequence, where lower is better), was used to evaluate performance.

| Method | Final Perplexity | Trainable Parameters |
| :--- | :---: | :---: |
| **1. Base Model (Zero-Shot)** | ~878,532 | 0 |
| **2. LoRA Fine-Tuning** | **1.420** | ~13.6 Million (0.17%) |
| **3. Full Fine-Tuning** | **1.418** | ~8.04 Billion (100%) |



## 💡 Analysis & Conclusion

The results clearly demonstrate the power and efficiency of LoRA. Both fine-tuning methods led to a massive improvement over the baseline, reducing perplexity from over 800,000 to ~1.4.

Most importantly, the **LoRA model achieved virtually the same performance as the fully fine-tuned model**, with a statistically insignificant difference in perplexity. This was accomplished by training only **0.17%** of the total parameters, which drastically reduces VRAM requirements, training time, and the storage footprint of the final model.

This project validates that PEFT techniques like LoRA are a critical and highly effective strategy for democratizing the customization of large-scale AI models.

## 🛠️ Setup and Usage

To replicate this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create --name lora_env python=3.11 -y
    conda activate lora_env
    ```

3.  **Install dependencies:**
    ```bash
    # Install PyTorch with CUDA support
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    # Install the rest of the packages
    pip install transformers datasets accelerate peft bitsandbytes scikit-learn
    ```

4.  **Log in to Hugging Face:**
    ```bash
    huggingface-cli login
    ```

5.  **Run the scripts:**
    * To get the baseline performance:
        ```bash
        python evaluate_base_model.py
        ```
    * To run LoRA fine-tuning:
        ```bash
        python finetune.py
        ```
    * To run full fine-tuning:
        ```bash
        python full_finetune.py
        ```

## 💻 Technology Stack

* **Model:** Meta Llama 3 8B Instruct
* **Dataset:** Databricks Dolly 15k
* **Core Libraries:** PyTorch, Hugging Face (`Transformers`, `PEFT`, `Datasets`, `Accelerate`), `bitsandbytes`
