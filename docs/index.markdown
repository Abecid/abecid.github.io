---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: Home
permalink: /
---
<style>
.container {
  /* flex-direction:row;
  align-items: center;
  justify-content:space-between; */
  /* border: 3px solid blue; */
  display:flex;
  width:100%;
}
.description {
  height:100%;
  margin: auto;
  margin-right:5%;
  margin-left:5%;
  text-align: center;
  /* border: 3px solid green; */
  align-items: stretch;
}
.profile {
  width:70%;
  height:auto;
  text-align: center;
}
.main {
  margin-top:2rem;
}
.hobby_img {
  /* display: inline-block; */
}
</style>
<!-- <h1>Adam Lee</h1> -->
<div class="container">

  <!-- <img class="profile" src="/assets/img/profile.jpeg"> -->

  <div class="description">
  <h1>Adam Lee</h1>
  <div style="display:fex;justify-content:center;">
  <img class="profile" src="/assets/img/profile2.jpeg">
  </div>
  Hi I was a computer science major at GeorgiaTech but will be attending Columbia University starting in Fall 2023. My research interests are transformers, code generation, NERF, Multimodal models, and diffusion models. I also spend time building defi apps on Ethereum. 
  <!--
  I previously attended Seoul National University (SNU) in 2021 as an Electrical and Computer Engineering major. While at SNU, I co-founded XREAL, the first metaverse club based in Korea.
  -->
  <br>
  <br>
  (현재 병특을 구하는 중입니다!)
  <br>
  <a href="mailto:adamlee3211@gmail.com">Email</a>  /  <a href="https://twitter.com/abecidadam">Twitter</a>
  <!-- >
  /  <a href="https://www.linkedin.com/in/adam-lee-653aa018b">LinkedIn</a>  /  <a href="/assets/pdf/AdamLee_Resume_20221013.pdf" download>CV</a>
  <-->
  
  </div>

</div>

<!-- {% for tag in site.tags %}
  <a href="#{{ tag[0] | slugify }}" class="post-tag">{{ tag[0] }}</a>
{% endfor %} -->

<div class="main">
{% assign href = "/featured" %}

<a href="{{ href }}" class="post-cat"><h4>Featured</h4></a>
<ul>
  {% for post in site.categories.Featured limit:3 %}
    <li><a href="{{ post.url }}">{{ post.title }}</a>
    <!-- ({{ post.tags | join: ", " }}) -->
    <!-- - {{ post.date | date: "%-d %B %Y"}} -->
    </li>
  {% endfor %}
  {% if site.categories.Featured.size > 3 %}
    <a href="{{ href }}"> View More.. </a>
  {% endif %}
</ul>

{% for cat in site.categories %}
  {% case cat[0] %}
    {% when "Paper Review" %}
      {% assign href = "/paper-review" %}
    {% when "Blog" %}
      {% assign href = "/blog" %}
    {% when "Research" %}
      {% assign href = "/research" %}
    {% when "Projects" %}
      {% assign href = "/projects" %}
    {% when "Other Pursuits" %}
      {% assign href = "/pursuits" %}
    {% when "Featured" %}
      {% continue %}
    {% else %}
      {% assign href = "/" %}
  {% endcase %}

  <a href="{{ href }}" class="post-cat"><h4>{{ cat[0] }}</h4></a>
  <ul>
    <!-- 
    {% assign base_max_posts = 3 %}
    {% assign base_minus_one = base_max_posts | minus:1 %}
    {% assign max_posts = base_max_posts %}
    {% assign complete_post_counter = 0 %}
    {% for post in cat[1] limit:max_posts %}
      {% if post.tags contains "Hidden" %}
        {% assign max_posts = max_posts | plus:1 %}
      {% else %}
        {% if complete_post_counter == base_minus_one %}
          {% break %}
        {% endif %}
        {% assign complete_post_counter = complete_post_counter | plus:1 %}
      {% endif %}
    {% endfor %}
    -->

    {% assign base_max_posts = 3 %}
    {% for post in cat[1] limit:max_posts %}
      {% if post.tags contains "Hidden" %}
        {% assign max_posts = max_posts | plus:1 %}
        {% continue %}
      {% endif %}
      <li><a href="{{ post.url }}">{{ post.title }}</a>
      <!-- ({{ post.tags | join: ", " }}) -->
      <!-- - {{ post.date | date: "%-d %B %Y"}} -->
      </li>
    {% endfor %}
    {% if cat[1].size > 3 %}
      <a href="{{ href }}"> View More.. </a>
    {% endif %}
  </ul>
{% endfor %}
  <p>Hobbies</p>
  <div class="hobby_img">
  <img class="profile" src="/assets/img/saxophone.jpeg" style="width:40%;">
  <video width="400" height="240" controls>
  <source src="/assets/vid/side3.mp4" type="video/mp4">
  Your browser does not support the video tag.
  </video>
  
  <!--
  <video width="400" height="240" controls>
  <source src="/assets/vid/front3.mp4" type="video/mp4">
  Your browser does not support the video tag.
  </video>
  -->
  </div>
</div>

<!-- Paper Reviews
(Recently Read Interesting Papers)

Blog

Research

Projects

Hobby

Other Pursuits -->

<!-- <ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul> -->
