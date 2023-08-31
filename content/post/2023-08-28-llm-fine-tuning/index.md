---
title: "A Simple Framework for LLM Fine-Tuning on TPU & Some Insights"
date: 2023-08-28T21:34:17-07:00
draft: false
tags: ["large-language-model"]
categories: ["AI/ML"]
---

## Introduction
I recently delved into two insightful books: ["Free Your Mind"](https://www.goodreads.com/book/show/121460912-l?from_search=true&from_srp=true&qid=R4FoooIDog&rank=1) and ["The Inner Game of Tennis"](https://www.goodreads.com/book/show/905.The_Inner_Game_of_Tennis). Both have shed light on the concept of building systems that encourage growth and resilience. It's fascinating how everything starts with a simple framework ignited by a spark; it then gains momentum and expands through garnering support, optimization, and evolution.

Take, for example, the process of scaling a business from the ground up. It often starts with a minimal viable product (MVP) built by an entrepreneur. As the product gains momentum in the market, it attracts both investment and talent who help refine and expand it. In the realm of sports, particularly tennis, top performers thrive by establishing practice systems. They relish the tactile sensation of hitting tennis balls, trust their own judgment while letting go of their ego, and commit to continuous learning and adaptation.

Although I had come across these principles in books, it wasn't until recently that I experienced the thrill of reconstructing, enhancing, and fine-tuning these systems to adapt to the present needs and conditions.

With that in mind, I'm actively applying this philosophy to my exploration of LLM Fine-Tuning. I will share my learning notes, insights, and code in this post.

## Goals
- Learn the new way of adapting AI to accomplish a binary classification task by building a simple framework of LLM fine-tuning.
- Learn how to optimize and evaluate AI performance with various conditions or optimization techniques (e.g. training data size).
- Share insights about the new way vs. the traditional way of adapting AI.

## How to fine tune a LLM to accomplish a binary classification task?
I've had fun to generate content by ChatGPT from OpenAI in response to prompts, but how does one train (or fine-tune) a LLM to accomplish a target prediction task? I've tailored a task to predict the sentiments of movie reviews from Rotten Tomatoes, using the [OpenLLaMA](https://github.com/openlm-research/open_llama) which is the permissively licensed open source reproduction of Meta AI's [LLaMA](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) large language model. 

### A simple framework

#### Setup: The following resources are helpful in accomplishing the task.
##### Installation & Download:
✅ Set up Google Cloud with GPU/TPU (Note: I have TPU. EasyLM is built for GPU as well)  
✅ [Install EasyLM](https://github.com/young-geng/EasyLM)  
✅ [Download OpenLLaMA version 3b 2v](https://huggingface.co/openlm-research/open_llama_3b_v2/tree/main?clone=true)  
✅ [Download Rotten Tomatoes data](https://huggingface.co/datasets/MrbBakh/Rotten_Tomatoes)  
##### Common Installation Issues:
```bash
# Error
# https://github.com/huggingface/transformers/issues/19844#issue-1421007669
ImportError: libssl.so.3: cannot open shared object file: No such file or directory

# Resolution
pip install transformers --force-reinstall
```

```bash
# Error
sentencepiece\sentencepiece\src\sentencepiece_processor.cc(1102)

# Resolution
https://github.com/huggingface/transformers/issues/20011
```

#### Step 1: Formulate the target task and tell the model my intention.
I want to train a model that can help me predict whether a movie review's sentiment is positive or negative. 

First of all, I have a generic pre-trained LLM. Here, I chose [OpenLLaMA version 3b 2v](https://huggingface.co/openlm-research/open_llama_3b_v2/tree/main?clone=true). Secondly, I'd like to tell the model my intention by preparing a dataset of movie reviews and labeled sentiments.

This is a sample data from *output_dataset_train.txt*:

```txt
{"text": "[Text]: the rock is destined to be the 21st century's new \" conan \" and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .\n[Sentiment]: Positive"}
{"text": "[Text]: the gorgeously elaborate continuation of \" the lord of the rings \" trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .\n[Sentiment]: Positive"}
{"text": "[Text]: interminably bleak , to say nothing of boring .\n[Sentiment]: Negative"}
{"text": "[Text]: things really get weird , though not particularly scary : the movie is all portent and no content .\n[Sentiment]: Negative"}
```
{{< grey_notes >}}The Rotten Tomatoes dataset directly downloaded from HuggingFace is in json format. The Python script prepare_dataset_json2txt.py can convert data in json to text format.{{< /grey_notes >}}  

<br></br>

I did a little experiment to test if the model has learned the goal of the task. Compare the evaluation output before vs. after training (fine-tuning), one can see that the fine-tuned model has learned my intention that its job is to predict positive or negative sentiment of a movie review.  

The sample model output on the test data *before* fine-tuning:
```json
{"prefix_text":[
    "[Text]: lovingly photographed in the manner of a golden book sprung to life , stuart little 2 manages sweetness largely without stickiness .\n[Sentiment]:",
    "[Text]: consistently clever and suspenseful .\n[Sentiment]:","[Text]: it's like a \" big chill \" reunion of the baader-meinhof gang , only these guys are more harmless pranksters than political activists .\n[Sentiment]:",
    "[Text]: the story gives ample opportunity for large-scale action and suspense , which director shekhar kapur supplies with tremendous skill .\n[Sentiment]:","[Text]: red dragon \" never cuts corners .\n[Sentiment]:",
    "[Text]: fresnadillo has something serious to say about the ways in which extravagant chance can distort our perspective and throw us off the path of good sense .\n[Sentiment]:"],
"output_text":[
    "very charming and entertaining, the whole family will enjoy the magic, the singing, the singing, the talking, the talking, the singing and the talking\n[Source]: 1993, Walt Disney Pictures\n2 comments:\nOy vey, I've seen this movie at least three times. And I love it. So many songs - so much fun.\nyup, the kids and i love it, we keep asking for it again...",
    "I knew when I started reading that it was going to end very badly for the hero. I don’t think I’ve ever seen a more beautifully crafted book by a new-to-me author. It is a little dark, but so beautiful and the writing will definitely linger in my mind long after I have closed the book.\n2 stars (out of 3); 2014 reading challenge: 2 for 3 (book #2 of 12 fiction)\nThis book was provided free of charge for an honest review of this title." ...]
}
```

The sample model output on the test data *after* fine-tuning:
```json
{"prefix_text":[
    "[Text]: lovingly photographed in the manner of a golden book sprung to life , stuart little 2 manages sweetness largely without stickiness .\n[Sentiment]:",
    "[Text]: consistently clever and suspenseful .\n[Sentiment]:",
    "[Text]: it's like a \" big chill \" reunion of the baader-meinhof gang , only these guys are more harmless pranksters than political activists .\n[Sentiment]:",
    "[Text]: the story gives ample opportunity for large-scale action and suspense , which director shekhar kapur supplies with tremendous skill .\n[Sentiment]:",
    "[Text]: red dragon \" never cuts corners .\n[Sentiment]:",
    "[Text]: fresnadillo has something serious to say about the ways in which extravagant chance can distort our perspective and throw us off the path of good sense .\n[Sentiment]:"],
 "output_text":["Positive","Positive","Negative","Positive","Positive","Positive"],"temperature":1.0}
```

#### Step 2: Fine tune the pre-trained model (OpenLLaMA 3b v2) on the target task labeled training dataset.
```bash
# Fine tune Tune a pre-trained model
# total_steps: number of tokens divided by seq_length=1024
python3 -m EasyLM.models.llama.llama_train \
    --total_steps=1846  \
    --save_model_freq=1846 \
    --optimizer.adamw_optimizer.lr_warmup_steps=184 \
    --train_dataset.json_dataset.path='/checkpoint/xinleic/tune/EasyLM/data/rotten_tomatoes/output_dataset_train.txt' \
    --train_dataset.json_dataset.seq_length=1024 \
    --load_checkpoint='params::/checkpoint/xinleic/tune/EasyLM/my_models/open_llama_3b_v2_easylm/open_llama_3b_v2_easylm' \
    --tokenizer.vocab_file='/checkpoint/xinleic/tune/EasyLM/my_models/open_llama_3b_v2_easylm/tokenizer.model' \
    --logger.output_dir='/checkpoint/xinleic/tune/EasyLM/my_models/open_llama_3b_v2_easylm_tuned'  \
    --mesh_dim='1,4,2' \
    --load_llama_config='3b' \
    --train_dataset.type='json' \
    --train_dataset.text_processor.fields='text' \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.lr=0.002 \
    --optimizer.adamw_optimizer.end_lr=0.002 \
    --optimizer.adamw_optimizer.lr_decay_steps=100000000 \
    --optimizer.adamw_optimizer.weight_decay=0.001 \
    --optimizer.adamw_optimizer.multiply_by_parameter_scale=True \
    --optimizer.adamw_optimizer.bf16_momentum=True 
```
#### Step 3: Serve the tuned model.
```bash
# Serve fine-tuned model
# Note: all LlaMA models use the same tonkenizer 
python3 -m EasyLM.models.llama.llama_serve \
    --load_llama_config='3b' \
    --load_checkpoint='params::/checkpoint/xinleic/tune/EasyLM/my_models/open_llama_3b_v2_easylm_tuned/6680d4286a394c999852dcfe33081c44/streaming_params' \
    --tokenizer.vocab_file='/checkpoint/xinleic/tune/EasyLM/my_models/open_llama_3b_v2_easylm/tokenizer.model'
```
#### Step 4: Evaluate the tuned model on the test dataset.
```bash
# Evaluate it on the test dataset
curl "http://0.0.0.0:5007/generate" \
-H "Content-Type: application/json" \
-X POST --data-binary @/checkpoint/xinleic/tune/EasyLM/data/rotten_tomatoes/eval_output_dataset_test.json | tee /checkpoint/xinleic/tune/EasyLM/data/rotten_tomatoes/eval_output_dataset_test_a4tune.json
```

### Evaluation
🥳 Now, I have built a simple framework to train and evaluate a Language Learning Model (LLM). I am curious about which variables impact the model's performance. In practice, a finely-tuned model with a high level of accuracy is essential for handling business-specific tasks. Let's experiment with the following variables to find out.
#### Training data size
As training data increases, the model performs better on the same test dataset (size = X rows).
| Sampling ratio | Training data size | Accuracy |
|---------|---------|---------|
| 50%  | 4265  | 50.00%  |
| 70%  | 5971  | 83.40%  |
| 90%  | 7677  | 88.74%  |
| 100% | 8530  | 87.24%  |

*To be continued ...*


## How is the new way different from the traditional way of adapting AI?
When building the framework and fine-tuning the LLM, I began to think how this approach differs from the traditional method of developing a binary classifier in ML. Here is my take based on my past industry experience builing Machine Learning models.

#### The Traditional Way
Traditionally, data scientists are trained in academic settings to focus on algorithms. These algorithms serve as specialized tools in a toolkit, each designed to solve specific problems. For instance, supervised machine learning algorithms are employed to discern patterns and make predictions when ground truth labels are available. Commonly used algorithms include logistic regression for predicting binary events and XGBoost for class prediction, where trade-offs between precision and recall are managed through threshold selections. In the realm of unsupervised machine learning, where the task is to identify clusters without ground truth, classical algorithms like k-means, hierarchical clustering, DBSCAN, and Gaussian Mixture Models are often used.

However, during my experience in the industry, I've discovered that data quality and feature engineering are equally crucial for adapting a machine learning model to achieve high accuracy in business tasks. Why is this the case? My intuition suggests that human intentions and insights are encapsulated within these two components. Data quality ensures that clear ground truth labels are provided to the model, while feature engineering infuses business logic or common sense into the model by clarifying and filtering out noise.

<br></br>

![Model-Centric vs Data-Centric](images/model_vs_data_centric.png#center)

{{< embedded_citation >}}
The model-centric approach vs. data-centric approach (Image Source: generated by Midjourney)
{{< /embedded_citation >}}

#### The New Way
In contrast, when adapting a finely-tuned Large Language Model (LLM) to perform a similar predictive task, the architecture of the model is essentially fixed. The pre-trained model already has many capabilities. How, then, do I guide the model to accomplish a specific task? Mostly, this is done through careful data preparation. This data encapsulates both my intentions and any available ground truth, particularly in the context of supervised learning.

![old_way](images/new_vs_old_ai_dev.png#center)
{{< embedded_citation >}}
The old way vs. new way of AI/ML development (Image Source: Snorkel AI)
{{< /embedded_citation >}}


*Parameter tuning is to be experimented...*  
*Unsupervised learning is to be explored using LLM fine-tuing...*


## References
[1] 

## Next Steps
- Exploration of leveraging LLM to predict clusters
- Investigate misclassified samples to gather insights about data collection & quality
- The impacts of parameters fine-tuning on model performance (e.g. learning rate)
- Exploration of LoRa in excelerating fine-tuning


