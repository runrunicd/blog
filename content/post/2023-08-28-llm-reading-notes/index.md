---
title: "Learning Notes on Large Language Model - Continued..."
date: 2023-08-01T13:33:00-07:00
draft: true
tags: ["large-language-model"]
categories: ["AI/ML"]
---

## Introduction to Large Language Model

-  Auto-regressive transformers are pretrained on an extensive corpus of self-supervised data,
followed by alignment with human preferences via techniques such as Reinforcement Learning with Human
Feedback (RLHF).

- Training methodology is simple but is limited to a few players with high computational requirements

- Public pretrained LLM
    - BLOOM (Scao et al., 2022)
    - LLaMa 1 (Touvron et al., 2023)
    - LLaMa 2, LLaMa 2-Chat which is optimized for dialogue use cases
    - Falcon (Penedo et al., 2023)
- Closed pretrained LLM
    - GPT-3 (Brown et al., 2020)
    - Chinchilla (Hoffmann et al., 2022)
- Closed "product" LLM that are heavily fine-tuned to align with human preferences to enhance their usability and safety
    - ChatGPT (OpenAI)
    - BARD (Google)
    - Claude (Antropic)



## Pretraining
- Use standard transformer architecture
    - From LLaMa 1 to LLaMa 2, increased context length and grouped-query attention (GQA)
- Apply pre-normalization using RMSNorm
- Use SwiGLU activation function & rotary positional embeddings
- Tokenizr - the same as LLaMa 1

##### Variables
- Params
- Context length
- Tokens
- Learning rate (LR)
- GQA

##### Hyperparameters
- AdamW optimizer
- Learning rate schedule (warmup steps)
- Weight decay
- Gradient clipping

#### Pretrained Model Evaluation
- Summarize the overall performance across a suite of popular benchmarks.
![model eval](images/model_eval.png#center)
{{< embedded_citation >}}(Image Source: Touvron et al. 2023){{< /embedded_citation >}}

## Fine-Tuning
Llama 2-Chat is the result of several months of research and iterative applications of alignment techniques, including both instruction tuning and RLHF, requiring significant computational and annotation resources.

- Supervised fine-tuning (SFT) with iterative reward modeling and RLHF

- Annotators have written both the prompt and its answer
    - helpfulness: how well Llama 2-Chat responses fulfill users’ requests and provide requested information
    - safety: whether Llama 2-Chat’s responses are unsafe

- Find-tuning started with the SFT stage with publicly available instruction tuning data (Chung et al., 2022)

- To improve alignment, high-quality vendor-based SFT data in the order of tens of thousands is shown to improve the results (Touvron et al. 2023)

- Data checks are important as different annotation platforms and vendors result in different model performance (Touvron et al. 2023)

- RLHF is a model training procedure that is applied to a fine-tuned language model to further align model behavior with human preferences and instruction following. 

- Human Preference Data Collection: We ask annotators to first write a prompt, then choose
between two sampled model responses, based on provided criteria. In order to maximize the diversity, the
two responses to a given prompt are sampled from two different model variants, and varying the temperature
hyper-parameter. In addition to giving participants a forced choice, we also ask annotators to label the degree
to which they prefer their chosen response over the alternative: either their choice is significantly better, better,
slightly better, or negligibly better/ unsure. (Touvron et al. 2023)

- Reward Modeling: The reward model takes a model response and its corresponding prompt (including contexts from previous
turns) as inputs and outputs a scalar score to indicate the quality (e.g., helpfulness and safety) of the model
generation. Leveraging such response scores as rewards, we can optimize Llama 2-Chat during RLHF for
better human preference alignment and improved helpfulness and safety. We train two separate reward
models, one optimized for helpfulness (referred to as Helpfulness RM) and another for safety (Safety RM) to address helpfulness vs. safety trade-off.


## Approach to Model Safety
- Use safety-specific data annotation and tuning, conduct red-teaming, and employ iterative evaluations.
![safety evaluation](images/safety_evaluation.png#center)
{{< embedded_citation >}}(Image Source: Touvron et al. 2023){{< /embedded_citation >}}


- Testing conducted to date has been in English and has not — and could not — cover all scenarios.


![training of llama-2-chat](images/training_of_llama2_chat.png#center)
{{< embedded_citation >}}(Image Source: Touvron et al. 2023){{< /embedded_citation >}}

## Reference
[1] [Llama 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al. 2023)
](https://arxiv.org/pdf/2307.09288.pdf)
