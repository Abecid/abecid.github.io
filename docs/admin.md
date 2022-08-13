---
layout: page
title: Admin
permalink: /admin/
---

Admin

Tags
{% for tag in site.tags %}
  <ul>
    <li>{{ tag[0] }}</li>
  </ul>
{% endfor %}

Categories
{% for cat in site.categories %}
  <ul>
    <li>{{ cat[0] }}</li>
  </ul>
{% endfor %}
