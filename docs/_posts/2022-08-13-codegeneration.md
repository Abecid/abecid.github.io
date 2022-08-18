---
layout: post
title:  "Future of Code Generation"
date:   2022-08-13 15:02:31 +0900
category: Blog
tags: [ NLP, Deep Learning]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<h2>Introduction</h2>
Recently there have been several models that have shown immense prospects for the field of code generation with deep learning.

<img src="/assets/posts/blog/1.code_generation/alphacode_sample.png">

Starting with GPT-3 and <a href="https://openai.com/blog/openai-codex/">Codex</a>, large language models started to be utilized for code generation after fine-tuning a pre-trained model with specific programming tasks. <a href="https://www.deepmind.com/blog/competitive-programming-with-alphacode">AlphaCode</a> has also pushed the state-of-the-art for competitive programming problems. 

However more recently there have been more innovative work in this arena and I will cover 2 papers broadly and speculate on what's next for the field of code generation.

<h2>CodeT: Code Generation with Generated Tests</h2>
<a href="https://arxiv.org/abs/2207.10397v1">Source Paper</a><br>
<img src="/assets/posts/blog/1.code_generation/codet.png">
As mentioned, pre-trained code generating models such as Codex have demonstrated the ability to generate multiple different code samples to solve a given programming task. However selecting among those samples remains a difficult task, while testing the samples with test cases in the standard, it is quite expensive to devise these tests. **This paper uses pre-trained language models to automatically generate test cases**.

The method is called **CodeT: CODE generation with generated Tests**. CODET executes the code solutions using the generated test cases, and then chooses the best solution based on a dual execution agreement with both the generated test cases and other generated solutions.

CODET can achieve significant, consistent, and surprising improvements over previous methods. For example, CODET improves the pass@1 on HumanEval to 65.8%, an increase of absolute 18.8% on the code-davinci-002 model, and an absolute 20+% improvement over previous state-of-the-art results.

<h2>CoditT5: Pretraining for Source Code and Natural Language Editing</h2>
<a href="https://arxiv.org/abs/2208.05446">Source Paper</a><br>

<h2>Benchmarks and SOTA Performances</h2>

