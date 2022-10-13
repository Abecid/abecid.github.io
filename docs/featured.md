---
layout: page
title: Featured
permalink: /featured/
---

Featured

 <ul>
    {% for cat in site.categories %}
        {% unless cat[0] == "Featured" %}
            {% continue %}
        {% endunless %}
        {% for post in cat[1] %}
            <li><a href="{{ post.url }}">{{ post.title }}</a></li>
        {% endfor %}
    {% endfor %}
</ul>

