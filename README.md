# Building-NanoGPT
Following GPT-2, GPT-3 and Attention is All You Need papers aswell as a tutorial from Andrej Karpathy to create a 124M parameter version of GPT-2

# Enhanced Large Language Model (LLM) Optimization

This project focuses on optimizing a large language model using various techniques to improve memory efficiency, training stability, and performance. Key components include using the fineWeb dataset for training and HellaSwag for validation.

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Features](#features)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project implements several techniques to optimize a large language model (LLM). Key aspects include efficient memory usage, stable parameter initialization, weight sharing, and utilizing multi-GPU setups for faster training.

## Datasets

- **Training Dataset:** [fineWeb](https://example.com/fineWeb-dataset)
- **Validation Dataset:** [HellaSwag](https://example.com/hellaswag-dataset)

## Features

- **Efficient Attention Mechanism:** Implemented using `F.scaled_dot_product_attention` to reduce memory usage.
- **Parameter Scaling:** Used `NANOGPT_SCALE_INIT` for stable initialization of model parameters.
- **Weight Sharing:** Applied weight sharing between embedding and output layers to reduce model size.
- **Gradient Accumulation:** Utilized to handle large batch sizes on limited hardware.
- **Fused AdamW Optimizer:** Configured for faster convergence on CUDA devices.
- **Distributed Data Parallel (DDP):** Implemented for multi-GPU training, ensuring scalability and faster training times.
  
## Results
The training process and model evaluation results indicate the effectiveness of the applied optimizations:

- **Loss Metrics**:
  - The training and validation loss steadily decrease, with the validation loss stabilizing and surpassing the value of the OpenAI GPT-2 (124M) checkpoint.
  - This trend demonstrates that the `nanogpt (124M)` model learns effectively and generalizes well without significant overfitting.

- **Evaluation Accuracy**:
  - The HellaSwag evaluation accuracy shows a consistent improvement, starting from approximately 0.24 and reaching around 0.30.
  - The final accuracy surpasses the performance of the OpenAI GPT-2 (124M) but falls behind the GPT-3 (124M) checkpoints, underscoring the model's competitive performance.

These results highlight the success of the optimizations implemented in the `nanogpt (124M)` model, showcasing its efficient learning and strong performance in comparison to established benchmarks.

![image](https://github.com/user-attachments/assets/15d16b34-28cd-4c66-bdba-0f0a30c6018a)


## Acknowledgments
- Special thanks to the contributors of the fineWeb and HellaSwag datasets.
- Inspiration and guidance from the AI research community.
