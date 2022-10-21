---
layout: post
title:  "Competition-Level Code Generation with AlphaCode"
date:   2022-08-17 18:17:31 +0900
category: [Paper Review]
tags: [NLP, Deep Learning, Code Generation]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}
<a href="https://arxiv.org/abs/2203.07814">Source Paper</a>

<img src="/assets/posts/paper_review/4.alphacode/title.png">

<h2>1. Brief Summary & Significance</h2>
This paper introduces a novel model called AlphaCode which is trained to solve competitive programming tasks. In contrast to prior code generating models such as Codex which generated code that could solve relatively simple problems these models still perform poorly when evaluated on more complex, unseen problems that require problem-solving skills beyond simply translating instructions into code. For example, competitive programming problems which require an understanding of algorithms and complex natural language remain extremely challenging. To address this gap, the researchers introduced a system for code generation that can create novel solutions to these problems that require deeper reasoning.

<h2>2. Competitive Programming Tasks and Evaluation</h2>
<h4>1) Example</h4>
<img src="/assets/posts/paper_review/4.alphacode/fig2.png">
<br>This is a sample competitive programming problem from CodeForces with a rating of 1500. 
<img src="/assets/posts/paper_review/4.alphacode/fig3.png">
<br>AlphaCode was able to generate a solution for this problem which requires deeper algorithmic reasoning.
<h4>2) Evaluation</h4>
DeepMind came up with a generic metric: n@k.
> The metric is defined as the “percentage of problems solved using n submissions from k samples per problem,” and it’s sufficiently similar to the way Codeforces evaluates submissions. AlphaCode is allowed to generate k samples (up to the range of 100K-1M) and from those, the system can submit n solutions (n≤k) to evaluation. AlphaCode is successful if at least one of the n submissions solves the problem.

<h2>3. AlphaCode</h2>
<h4>1) Overview</h4>
<img src="/assets/posts/paper_review/4.alphacode/fig4.png">
- Pre-training: AlphaCode is a transformer-based large language model initially trained on GitHub code repositories
- Fine-tuning:  DeepMind created a competitive programming dataset named CodeContests to fine-tune and specialize AlphaCode. It compensates for the low amount of public examples of competitive programming problems.
- Generation (k)
- Filtering and Clustering (n)
<h4>2) Model Architecture</h4>
The authors propose an encoder-decoder model architecture of various sizes (300M, 1.1B, 2.8B, 8.7B, 41.1B). Asymmetric architecture 1536 tokens in encoder, 768 tokens in decoder. SentencePiece tokenizer trained on GitHub + CodeContest dataset with 8,000 tokens. Same tokenizer is used for encoder and decoder, across both programming languages and natural language

<h4>3) Datasets</h4>
- Pre-training: GitHub Open source code
- Fine-tuning: CodeContests, custom data set. Fixed false positives (30~60%) by generating additional test cases by mutating existing ones: “randomly incrementing or decrementing integers, and swapping and changing characters in strings.” These tests are then verified with correct solutions. The result was impressive, reducing the false positive rate down to 4%.

<h4>4) Sampling, filtering, clustering</h4>
<h6>Sampling</h6>
- The model generates several million samples for each problem.
- By configuring the metadata (random tags, ratings etc and high temperature) it can generate a diverse array of possible candidate solutions.
<h6>Filtering</h6>
- By testing the samples with the test cases, 99% of the samples are filtered leaving with 100s to 1000s of samples left.
<h6>Clustering</h6>
- This process helps to group semantically similar programs into clusters to avoid selecting programs with similar behaviors/solutions. Once this is done, they sample a solution from each cluster ordered from largest to smallest in a round robin manner.
- To infer similarity the researchers pre-trained a test input generation model with the same architecture as the main models on the same GitHub dataset. The idea was to have the model generate new test inputs for the problem. AlphaCode’s samples could be evaluated on these new tests — not necessarily valid or correct, but useful regardless. Then, evaluated samples were clustered depending on the solutions they gave to the new tests — wrong answers gave even more information than correct ones.

<h2>4. Results</h2>
<img src="/assets/posts/paper_review/4.alphacode/fig1.png">
<br>AlphaCode was able to be ranked in the top 54.3% among participants in 10 simulated CodeForces competitions. 
<img src="/assets/posts/paper_review/4.alphacode/table10.png">
<br>Compared to other language models and code generation models, although Codex performs slightly better for Introductory problems, AlphaCode performs much better than previous models for both Interview and Competition-level programming problems.

<h2>5. References</h2>
<ul>
<li><a href="https://arxiv.org/abs/2203.07814">Competition-Level Code Generation with AlphaCode</a></li>
<li><a href="https://victordibia.com/blog/alpha-code/">AlphaCode: Competition-Level Code Generation with Transformer Based Architectures | Paper Review</a></li>
</ul>
