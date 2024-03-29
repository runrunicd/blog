---
title: "Embeddings Use Cases"
date: 2023-11-14
draft: false
tags: ["large-language-model", "embeddings"]
categories: ["AI/ML"]
---

## Introduction
When I first heard about text embeddings, I was perplexed. My tech lead handed me an 800-page Natural Language Processing textbook, but I did not finish reading it. Now, with the capabilities of OpenAI models, it's time to learn and broadly leverage embeddings to accomplish many tasks that were hard be achieved by human labor or simple data analysis, thanks to easy access to these embeddings. 

Let's start exploring:
- Clustering
- Search
- Recommendations
- Classification
- Anomaly detection

## Goals
The ultimate goal is to become familiar with the framework of leveraging embeddings to accomplish applicable tasks. The best way to learn and improvise is first by getting your hands dirty with the data and tools. Optimization and deep-diving happen later naturally.
- Summarize key terminology, concept, and usage of text embeddings
- Build the basic framework and standardized iPython notebooks for each use case that could benefit from text embeddings
- Brainstorm business use cases and ideas for improvements

## What do you need to know about embeddings?
- An embedding is a vector (list) of floating point numbers, representing a text string.
- The distance between two embeddings (vectors) measures their relatedness. The smaller the distance, the higher they are related, vice versa.
- Dimensionality reduction methods: 
    - t-SNE, a non-linear dimension reduction method, which stands for t-distributed Stochastic Neighbor Embedding. The ML algorithm calculates the similarity in both high dimensional sapce and low dimensional space, then the similarity difference in both spaces is minimized using an optimization method, for instance, gradient descend. It was developed by Laurens van der Maaten and Geoffrey Hinton in 2008. Here's a brief overview of what t-SNE does and how it work.
    - PCA, a linear dimension reduction method, where the data in high dimensional space is mapped linearly into low dimensional space while maximizing the variance of the data.

## Use Cases
### Clustering
Can we identify clusters among movie reviews and their themes? It's going to be difficult to review through all reviews and identify clusters of movie reviews. With AI, we can achieve that and I'll demo it here. Let's use [Rotten Tomatoes dataset](https://huggingface.co/datasets/rotten_tomatoes). To obtain text embeddings, let's use OpenAI's embeddings API and the model text-embedding-ada-002 is recommended. Note: all the code and data are open to public.

```bash
# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage
import openai
openai.api_key = '[openai_key]'

import pandas as pd
import numpy as np
import tiktoken
import sys
from typing import List, Optional
from sklearn.manifold import TSNE
from ast import literal_eval

# Now try to import your module again
def get_embedding(text: str, engine="text-similarity-davinci-001", **kwargs) -> List[float]:

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], engine=engine, **kwargs)["data"][0]["embedding"]
```


```bash
# Embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
```

```bash
from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes")
df = dataset['train'].to_pandas()
print("{x} rows in df.".format(x=len(df)))
```
<br></br>

This is the sample data. The label is either 1 for positive sentiment or 0 for negative sentiment for the movie reviews.

| text | label |
|------|------|
| the rock is destined to be the 21st century's ...	| 1
| the gorgeously elaborate continuation of " the...	| 1


Here, we get embeddings for the top 5000 move reviews.

```bash
top_n = 5000

encoding = tiktoken.get_encoding(embedding_encoding)

# omit reviews that are too long to embed
df["n_tokens"] = df['text'].apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens].tail(top_n)

# get embeddings and save them
df["embedding"] = df['text'].apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("./rotten_tomatoes_with_embeddings_{x}.csv".format(x=top_n))
```


Apply Kmeans algorithm to the embeddings to identify clusters.

```bash
import numpy as np
from sklearn.cluster import KMeans

# Convert to a list of lists of floats
matrix = np.vstack(df.embedding.apply(literal_eval).to_list())

n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans.fit(matrix)
df['cluster'] = kmeans.labels_

df.groupby("cluster").label.mean().sort_values()
```

We can see that the mean of label values varies by cluster, suggesting that the sentiments are separated by the clustering, espectially cluster (2), cluster (1, 0, 4), and cluster (3).

| Cluster | Mean | 
|---------|------|
| 2  | 0.010241 |
| 1  | 0.091311 |
| 0  | 0.099548 |
| 4  | 0.113420 | 
| 3  | 0.480874 |

To visualize the clusters, transform the embeddings of high-dimension 1536 to 2D by t-SNE.

```bash
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px

tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
vis_dims2 = tsne.fit_transform(matrix)

# Define the color map for the clusters
color_map = {
    0: 'firebrick',         # Cluster 0
    1: 'orangered',  # Cluster 1
    2: 'gold',      # Cluster 2
    3: 'limegreen',       # Cluster 3
    4: 'maroon'       # Cluster 4
}

# Map the cluster labels to colors
df['color'] = df['cluster'].map(color_map)

# Convert cluster to a categorical type with the specified order
df['cluster'] = pd.Categorical(df['cluster'], categories=[0, 1, 2, 3, 4], ordered=True)

# Create a DataFrame from the t-SNE results
df_tsne = pd.DataFrame({'Dimension 1': vis_dims2[:, 0], 
                        'Dimension 2': vis_dims2[:, 1], 
                        'Cluster': df['cluster'],
                        'Color': df['color']})

# Create the 2D scatter plot using Plotly Express
fig = px.scatter(
    df_tsne, x='Dimension 1', y='Dimension 2',
    color='Cluster', labels={'Cluster': 'Cluster'},
    title='Clusters identified visualized in 2D using t-SNE',
    color_discrete_map=color_map # Apply the color map
)

# Define the category orders for the legend to make it discrete
fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))  # Adjust marker size
fig.update_layout(
    legend=dict(
        traceorder='normal',
        title_text='Cluster',
        title_font=dict(size=14)
    )
)


# Show the plot
fig.show()
```
<br></br>
#### 2D visualization from t-SNE
We can identify the clusters by the differently colored dense cores.  
![2D visualization of clusters](images/2d_tsne.png#center)
<br></br>

#### 2D visualization from PCA
It's exploratory, so it's worthwhile to see PCA 2D results too. We label positive reviews green and negative red.
![2D visualization of clusters](images/2d_pca.png#center)
<br></br>

#### 3D visualization from t-SNE
Sometimes, it's helpful to identify clusters in 3D visualization as there might be more than two major forces critical in classifying the move reviews. You may wonder why are the movie reviews in each cluster grouped together? When we go on to the next section, we may see that the green cluster is for positive reviews and the other four are quite negative.
<br></br>

{{< plotly >}}
<iframe width="100%" height="550" name="iframe", src="iframes/3d_tsne.html"></iframe>
{{< /plotly >}}

<br></br>



#### Cluster theme
Furthermore, we can leverage the model text-davinci-003 to summarize the theme for each cluster of movie reviews on Rotten Tomatoes.

Let’s summarize the clusters, the mean sentiment, and the theme. Cluster 2 is very negative, expressing disappointment with the movies. In contrast, cluster 3 is very positive, with praise. Finally, clusters 1, 0, and 4 are closer to negative reviews but are not as disappointed as cluster 2.

| Cluster | Mean Sentiment | Theme |
|---------|------|------|
| 2  | 0.010241 | Disappointed with the quality of the movie. |
| 1  | 0.091311 | The reviews are all negative and critical of the movie.
| 0  | 0.099548 | All of the reviews are negative and express dissatisfaction with the product or experience. |
| 4  | 0.113420 | Disappointment with the quality of the product or experience.
| 3  | 0.480874 | All of the reviews are positive and praise the movie for its unique qualities, such as its surreal sense of humor, technological finish, insightful writing, delicate performances, and character-driven storytelling.


```txt
Cluster 0 Theme:  All of the reviews are negative and express dissatisfaction with the product or experience.
1, now as a former gong show addict , i'll admit it , my only complaint i
0, there's just no currency in deriding james bond for being a clichéd , 
0, ecks this one off your must-see list .
0, skip this turd and pick your nose instead because you're sure to get m
0, this movie is about the worst thing chan has done in the united states
----------------------------------------------------------------------------------------------------
Cluster 1 Theme:  The reviews are all negative and critical of the movie.
0, the type of dumbed-down exercise in stereotypes that gives the [teen c
0, just not campy enough
0, not so much farcical as sour .
0, a one-trick pony whose few t&a bits still can't save itself from being
0, cuba gooding jr . valiantly mugs his way through snow dogs , but even 
----------------------------------------------------------------------------------------------------
Cluster 2 Theme:  Disappointed with the quality of the movie.
0, its generic villains lack any intrigue ( other than their funny accent
0, one of the most highly-praised disappointments i've had the misfortune
0, . . . with the candy-like taste of it fading faster than 25-cent bubbl
0, for all its impressive craftsmanship , and despite an overbearing seri
0, the script by vincent r . nebrida . . . tries to cram too many ingredi
----------------------------------------------------------------------------------------------------
Cluster 3 Theme:  All of the reviews are positive and praise the movie for its unique qualities, such as its surreal sense of humor, technological finish, insightful writing, delicate performances, and character-driven storytelling.
1, what elevates the movie above the run-of-the-mill singles blender is i
0, at least it's a fairly impressive debut from the director , charles st
1, insightfully written , delicately performed
1, one of those exceedingly rare films in which the talk alone is enough 
1, a stylish but steady , and ultimately very satisfying , piece of chara
----------------------------------------------------------------------------------------------------
Cluster 4 Theme:  Disappointment with the quality of the product or experience.
0, qualities that were once amusing are becoming irritating .
0, everything's serious , poetic , earnest and -- sadly -- dull .
1, what's infuriating about full frontal is that it's too close to real l
0, befuddled in its characterizations as it begins to seem as long as the
0, human nature talks the talk , but it fails to walk the silly walk that
----------------------------------------------------------------------------------------------------
```

```bash
# Reading a review which belong to each group.
rev_per_cluster = n_clusters

for i in range(n_clusters):
    print(f"Cluster {i} Theme:", end=" ")

    reviews = "\n".join(
        df[df['cluster'] == i]
        .text.str.replace("Title: ", "")
        .str.replace("\n\nContent: ", ":  ")
        .sample(rev_per_cluster, random_state=42)
        .values
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f'What do the following customer reviews have in common?\n\nCustomer reviews:\n"""\n{reviews}\n"""\n\nTheme:',
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print(response["choices"][0]["text"].replace("\n", ""))

    sample_cluster_rows = df[df.cluster == i].sample(rev_per_cluster, random_state=42)
    for j in range(rev_per_cluster):
        print(sample_cluster_rows.label.values[j], end=", ")
        print(sample_cluster_rows.text.str[:70].values[j])

    print("-" * 100)
```

#### Insight
The Rotten Tomatoes movie reviews can be clearly classified into 2 categories: positive reviews (cluster 3) and negative reviews (cluster 2, 1, 0, & 4). With the 2d/3d visuals, we can zoom in and see that the orange postive reviews can be distinguished from the other three negative review clusters. 
## Further Exploration
- Other use cases and their application

## Learning & Thought
### Clustering Business Use Cases
1. [Assistant] Categorization tasks 
    - Categorize documents (Doc2Vec)
    - Identify collaborators
    - Discover mood patterns from notes & diary
    - Organize bookmarks
2. [Personalization] 
    - Understand user behavior - recommendations, search, mood
3. [Operations] 
    - Identify fraudulent users
    - Triage and tag tickets or reports





