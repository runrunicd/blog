---
title: "Build a Simple Framework for LLM Fine-Tuning on TPU & Applications"
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

## How to fine tune an AI Model (LLM) to accomplish a binary classification task?
I've had fun to generate content by ChatGPT from OpenAI in response to prompts, but how does one train (or fine-tune) a LLM to accomplish a target prediction task? I've tailored a task to predict the sentiments of movie reviews from Rotten Tomatoes, using the [OpenLLaMA](https://github.com/openlm-research/open_llama) which is the permissively licensed open source reproduction of Meta AI's [LLaMA](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) large language model. 

##### AI models
- GPT-4 (OpenAI): [white papaer](https://arxiv.org/abs/2303.08774)
- LLaMA 2 (Meta AI): [white paper](https://arxiv.org/abs/2307.09288)
- Claude (Anthropic)
- LaMDA (Google)
- PaLM (Google)
- Gopher (DeepMind)

### A simple framework

#### Setup: The following resources are helpful in accomplishing the task.
##### Installation & download
✅ Set up Google Cloud with GPU/TPU (Note: I have TPU. EasyLM is built for GPU as well)  
✅ [Install EasyLM](https://github.com/young-geng/EasyLM)  
✅ [Download OpenLLaMA version 3b v2](https://huggingface.co/openlm-research/open_llama_3b_v2/tree/main?clone=true)  
✅ [Download Rotten Tomatoes data](https://huggingface.co/datasets/MrbBakh/Rotten_Tomatoes)  
##### Common installation issues
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

First of all, I have a generic pre-trained LLM. Here, I chose [OpenLLaMA version 3b v2](https://huggingface.co/openlm-research/open_llama_3b_v2/tree/main?clone=true). Secondly, I'd like to tell the model my intention by preparing a dataset of movie reviews and labeled sentiments.

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

#### Step 2: Fine tune the pre-trained model on the target task labeled training dataset.

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
    --logger.output_dir='/checkpoint/xinleic/tune/EasyLM/my_models/open_llama_3b_v2_easylm_tuned_002'  \
    --mesh_dim='1,4,2' \
    --load_llama_config='3b' \
    --train_dataset.type='json' \
    --train_dataset.text_processor.fields='text' \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.lr=0.0002 \ #the initial learning rate
    --optimizer.adamw_optimizer.end_lr=0.0002 \ #the final learning rate after decay
    --optimizer.adamw_optimizer.lr_decay_steps=100000000 \ #the number of steps for cosine learning rate decay
    --optimizer.adamw_optimizer.weight_decay=0.001 \
    --optimizer.adamw_optimizer.multiply_by_parameter_scale=True \
    --optimizer.adamw_optimizer.bf16_momentum=True \
    --optimizer.adamw_optimizer.b1=0.9 \
    --optimizer.adamw_optimizer.b2=0.9
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
<br></br>

### How to improve accuracy
🥳 Now, I have built a simple framework to train and evaluate a Language Learning Model (LLM). I am curious about which variables impact the model's performance. In practice, a finely-tuned model with a high level of accuracy is essential for handling business-specific tasks. Let's experiment with the following variables to find out (model ID 001).

#### Training data size
<span style="background-color: #FFDAB9"> The larger the training data set, the better the model performs. </span>
| Sampling ratio | Training data size | Accuracy |
|---------|---------|---------|
| 50%  | 4265  | 50.00%  |
| 70%  | 5971  | 83.40%  |
| 90%  | 7677  | 88.74%  |
| 100% | 8530  | 87.24%  |  


#### Training data label quality
<span style="background-color: #FFDAB9">Human subjective judgments are injected into the training dataset, thereby affecting the performance of the model.</span>  

Is the label from the original dataset considered the ground truth? If so, it's worth noting that there may be bias involved, as the sentiment of a movie review being categorized as 'positive' or 'negative' can be subjective. 

To understand how the original labels differ from my judgment, I selected 30 movie reviews. I then recalculated the accuracy, using my judgment as the new ground truth, to evaluate the impact of human curation of data labels on model performance. The comparison results are as follows. 

*Selected 30 data samples and they reviewed by me:*

<div style="width: 100%; max-height: 400px; overflow: auto; border: 1px solid #ccc; font-size: 12px;;">

Text|Sentiment (Original)|Sentiment (LLM)|Sentiment (Mine)
|---------|---------|---------|---------|
lovingly photographed in the manner of a golden book sprung to life , stuart little 2 manages sweetness largely without stickiness .|Positive|Positive|Positive
consistently clever and suspenseful .|Positive|Positive|Positive
it's like a " big chill " reunion of the baader-meinhof gang , only these guys are more harmless pranksters than political activists .|Positive|Negative|Positive
the story gives ample opportunity for large-scale action and suspense , which director shekhar kapur supplies with tremendous skill .|Positive|Positive|Positive
red dragon " never cuts corners .|Positive|Positive|Positive
fresnadillo has something serious to say about the ways in which extravagant chance can distort our perspective and throw us off the path of good sense .|Positive|Positive|Positive
throws in enough clever and unexpected twists to make the formula feel fresh .|Positive|Positive|Positive
weighty and ponderous but every bit as filling as the treat of the title .|Positive|Positive|Positive
a real audience-pleaser that will strike a chord with anyone who's ever waited in a doctor's office , emergency room , hospital bed or insurance company office .|Positive|Positive|Positive
generates an enormous feeling of empathy for its characters .|Positive|Positive|Positive
exposing the ways we fool ourselves is one hour photo's real strength .|Positive|Positive|Positive
it's up to you to decide whether to admire these people's dedication to their cause or be repelled by their dogmatism , manipulativeness and narrow , fearful view of american life .|Positive|Positive|Negative
mostly , [goldbacher] just lets her complicated characters be unruly , confusing and , through it all , human .|Positive|Positive|Positive
. . . quite good at providing some good old fashioned spooks .|Positive|Positive|Positive
at its worst , the movie is pretty diverting ; the pity is that it rarely achieves its best .|Positive|Negative|Negative
scherfig's light-hearted profile of emotional desperation is achingly honest and delightfully cheeky .|Positive|Positive|Positive
a journey spanning nearly three decades of bittersweet camaraderie and history , in which we feel that we truly know what makes holly and marina tick , and our hearts go out to them as both continue to negotiate their imperfect , love-hate relationship
the wonderfully lush morvern callar is pure punk existentialism , and ms . ramsay and her co-writer , liana dognini , have dramatized the alan warner novel , which itself felt like an answer to irvine welsh's book trainspotting .|Positive|Negative|Negative
as it turns out , you can go home again .|Positive|Negative|Negative
you've already seen city by the sea under a variety of titles , but it's worth yet another visit .|Positive|Positive|Positive
this kind of hands-on storytelling is ultimately what makes shanghai ghetto move beyond a good , dry , reliable textbook and what allows it to rank with its worthy predecessors .|Positive|Positive|Positive
making such a tragedy the backdrop to a love story risks trivializing it , though chouraqui no doubt intended the film to affirm love's power to help people endure almost unimaginable horror .|Positive|Negative|Positive
grown-up quibbles are beside the point here . the little girls understand , and mccracken knows that's all that matters .|Positive|Positive|Positive
a powerful , chilling , and affecting study of one man's dying fall .|Positive|Positive|Positive
this is a fascinating film because there is no clear-cut hero and no all-out villain .|Positive|Positive|Positive
a dreadful day in irish history is given passionate , if somewhat flawed , treatment .|Positive|Positive|Positive
. . . a good film that must have baffled the folks in the marketing department .|Positive|Negative|Positive
. . . is funny in the way that makes you ache with sadness ( the way chekhov is funny ) , profound without ever being self-important , warm without ever succumbing to sentimentality .|Positive|Positive|Positive
devotees of star trek ii : the wrath of khan will feel a nagging sense of deja vu , and the grandeur of the best next generation episodes is lacking .|Positive|Negative|Negative
a soul-stirring documentary about the israeli/palestinian conflict as revealed through the eyes of some children who remain curious about each other against all odds .|Positive|Positive|Positive
</div>  

<br></br>

*The accuracy based on the original label vs. my label (sample size = 30):*
| # Misclassified Samples | Accuracy | 
|---------|---------|
| 8  | 76.7%  | 
| 5  | 83.3%  |  


#### Hyperparameter tuning
I trained using the AdamW optimizer (Loshchilov and Hutter, 2017). Accuracy was evaluated based on the same test dataset. 
##### [*AdamW Optimizer*](https://github.com/young-geng/EasyLM/blob/main/docs/optimizer.md)


<div style="width: 100%; max-height: 400px; overflow: auto; border: 1px solid #ccc; font-size: 12px;;">

 ID|LR|params|b1|b2|warmup steps|weight decay|tokens|accuracy
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
 001|0.002|3B|0.9|0.9|184|0.001|378K|87.24%
 002|0.0002|3B|0.9|0.9|184|0.001|378K|
 003|0.00002|3B|0.9|0.9|184|0.001|378K|
 004|0.000002|3B|0.9|0.9|184|0.001|378K|


</div>  

*To be continued...*


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

 
* LLM fine-tuing application is to explore for unsupervised learning...*


## References
...

## Further Exploration
- Exploration of leveraging LLM to predict clusters - unsupervised leanring
- The impacts of parameters fine-tuning on model performance (e.g. learning rate)
- Exploration of LoRa in excelerating fine-tuning

## Thoughts
#### Data collection & curation
I'm reading white paper and human evaluation is now golden standard to assess model performance. I'm thinking for personalized GPT, what would be the best interface to collect personal perference?
#### Application
First of all, there's text lol... Can we mine insights and provide data analysis? Can we predict sentiments (BloombergGPT does it) to signal trends? 
#### Fine-tuning
Temperature: a hyperparameter that controls the randomness of the model's output. When the temperature is close to zero (e.g., 0.1), the model becomes deterministic; when the temperature is high (e.g., 2 or 3), the model's output becomes more random and creative. Okie, according to this information, maybe higher temperature is correlated with higher halluciation? It's worth to try it out.
  



