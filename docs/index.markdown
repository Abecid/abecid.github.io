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
  width:35%;
  float: left;
}
.main {
  margin-top:2rem;
}
</style>
<!-- <h1>Adam Lee</h1> -->
<div class="container">

  <img class="profile" src="/assets/img/profile.jpeg">

  <div class="description">
  <h1>Adam Lee</h1>
  Hi I'm currently a computer science major at GeorgiaTech and previously attended Seoul National University (SNU) as an Electrical Engineering major in 2021. While at SNU, I co-founded XREAL, the first metaverse club based in Korea. My primary interests are in transformers, NERF, and 3D Reconstruction. However I also spend time building defi apps on Ethereum. 
  <br>
  <br>
  (현재 병특을 구하는 중입니다!)
  <br>
  <a href="mailto:adamlee3211@gmail.com">Email</a> <a href="https://twitter.com/abecidadam">Twitter</a> <a href="https://www.linkedin.com/in/adam-lee-653aa018b/">LinkedIn</a> <a href="/assets/pdf/AdamLee_CV_2022_8_13.pdf" download>CV</a>
  
  </div>

</div>

<!-- {% for tag in site.tags %}
  <a href="#{{ tag[0] | slugify }}" class="post-tag">{{ tag[0] }}</a>
{% endfor %} -->

<div class="main">
{% assign href = "/" %}
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
    {% else %}
      {% assign href = "/" %}
  {% endcase %}

  <a href="{{ href }}" class="post-cat">{{ cat[0] }}</a>
  <ul>
    {% for post in cat[1] limit:5 %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
    {% if cat[1].size > 5 %}
      <a href="{{ href }}"> View More </a>
    {% endif %}
  </ul>
{% endfor %}
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
