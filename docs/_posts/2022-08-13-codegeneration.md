---
layout: post
title:  "Future of Code Generation"
date:   2022-08-13 15:02:31 +0900
category: Blog
tags: [Deep Learning, NLP]
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
<img src="/assets/posts/blog/1.code_generation/T5_fig3.png">
Large language models have shown effectiveness in generating software; **however, they are not well-suited for editing tasks as they are not designed to reason about edits**.

This paper proposes CoditT5, a large language model for software-related editing tasks that is pretrained on large amounts of source code and natural language comments. 

This model is fine-tuned on various downstream tasks including comment updating, bug fixing, and automated code review.

The training data consisted of 6 programming languages from the CodeSearchNet dataset and the model consists of 12 encoder and decoder
layers, 12 attention heads, and a hidden dimension size of 768.

<h2>Benchmarks and SOTA Performances</h2>
<h4>HumanEval</h4>
<img src="/assets/posts/blog/1.code_generation/humaneval.png">
The HumanEval benchmark was introduced by OpenAI in their paper for Codex. Models have been submitted in this benchmark starting this year with AlphaCode and then Code-T which was released by Microsoft in July. 
<h4>CoNaLa</h4>
<img src="/assets/posts/blog/1.code_generation/conala.png">
  The CMU CoNaLa, the Code/Natural Language Challenge dataset is a joint project from the Carnegie Mellon University NeuLab and Strudel labs. Its purpose is for testing the generation of code snippets from natural language. The data comes from StackOverflow questions. There are 2379 training and 500 test examples that were manually annotated. Every example has a natural language intent and its corresponding python snippet.
Starting with a seq2seq architecture of TranX, more techniques have been used in lexical preprocessing, input representations, and copy mechanisms to reach the current state-of-the-art model. Most models rely on a BERT encoder and decoder architecture.


<h2>Limitations and Challenges of Deep Learning based Code Generation</h2>
<h4>1. Complexity</h4>
Most models across different benchmarks have been tested primarily with snippets and simplistic features and algorithms instead. As mentioned in Codex, simply stating a logical sequence of events could be too much for the model comprehend. 

<h4>2. Cheating by using training data</h4>
There are major security issues concerning deep learning based code generation models since there have been reported cases where the model just suggested code word for word from its training data. Even though most models in the literature are trained on open source software, still several licenses require one for citing the source when used. There have been several works that have been proposed to prevent models from regurgitating training data and also citing properly in case code was used under a license. 

<h4>3. Not as good as the average software engineer</h4>
Programming and software engineering is different. Programming is solving a specific task by designing an algorithm that outputs the expected code based on the provided input. However software engineering concerns software architecture, different objects and classes and are used in different modules. Since current models still lack the capability to output algorithms based on explicit input, it will take several breakthroughs until models can generate convoluted software systems that is consisted of several classes and modules. But this will be the future.

<h4>4. Great Cheating for Class?</h4>
There has been a famous post on HackerNews which was allegedly written by a instructor for an intro to programming class. The author worried about code generation models, specifically Codex, would prevent their students from completing tasks in introductory algorithms since models are capable of automatically suggesting the correct implementation of the algorithm that was assigned. This doesn't concern research per se, but maybe we need to implement different methods to evaluate students by tweaking the instructions or focus on theory and train more on how to fix implementations rather than writing implementations from scratch? 

Since modern software development involves using different modules written by others, **maybe the future of programming is not only using other modules and rearranging, editing, and adding code to suit one's objectives but also using pre-implemented code as a given and fine-tune the code from there?**
