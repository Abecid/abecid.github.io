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
  Hi I'm a 4th year computer science major at GeorgiaTech. I previously attended Seoul National University (SNU) in 2021 as an Electrical and Computer Engineering major. While at SNU, I co-founded XREAL, the first metaverse club based in Korea. My primary research interests are transformers, NERF, and 3D Reconstruction. I also spend time building defi apps on Ethereum. 
  <br>
  <br>
  (현재 병특을 구하는 중입니다!)
  <br>
  <a href="mailto:adamlee3211@gmail.com">Email</a>  /  <a href="https://twitter.com/abecidadam">Twitter</a>  /  <a href="https://www.linkedin.com/in/adam-lee-653aa018b">LinkedIn</a>  /  <a href="/assets/pdf/AdamLee_CV_2022_8_13.pdf" download>CV</a>
  
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
      <li><a href="{{ post.url }}">{{ post.title }}</a>
      <!-- ({{ post.tags | join: ", " }}) -->
      <!-- - {{ post.date | date: "%-d %B %Y"}} -->
      </li>
    {% endfor %}
    {% if cat[1].size > 5 %}
      <a href="{{ href }}"> View More </a>
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
  <video width="400" height="240" controls>
  <source src="/assets/vid/front3.mp4" type="video/mp4">
  Your browser does not support the video tag.
  </video>
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
