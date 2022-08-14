---
layout: post
title:  "Attention is all you need - Transformer"
date:   2022-08-12 12:02:31 +0900
category: Paper Review
tags: [NLP, Deep Learning]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}
<a href="https://arxiv.org/abs/1706.03762?context=cs">Source Paper</a>

<img src="/assets/posts/paper_review/1.transformer/title.png">

<h2>Brief Summary & Comments</h2>
This is a breakthrough paper which introduces a novel architecture called Transformer. This model dispenses recurrence and convolutions entirely, solely relying on an attention mechanism. This model achieved state-of-the-art results in Machine Translation, achieving 27.5 BLEU on English-to-German translation and 41.1 on English-to-French translation. 

<h2>Transformer Architecture & Self-Attention</h2>
<img src="/assets/posts/paper_review/1.transformer/transformer.png">
<br>
<h3>Encoder-Decoder</h3>
The Transformer has an encoder-decoder structure like other competitive neural sequence transduction models. 

They are both composed of N=6 layers (which was arbitrarily chosen by the authors, could be reconfigured). 
<h3>Self-Attention</h3>
<img src="/assets/posts/paper_review/1.transformer/attention.png">
<img src="/assets/posts/paper_review/1.transformer/attention_equation.png">
<br>
The input is consisted of Query, Key, Value vectors. After the dot product, the result is divided by the square root of the dimension dk for normalization and better convergence. 

In multi-head attention, the query, key, value vectors and linearly projected h times where h=8 (this value is also reconfigurable). 



<h2>Python Implementation</h2>

<h2>PyTorch Project</h2>

<h2>Key Takeaways</h2>

