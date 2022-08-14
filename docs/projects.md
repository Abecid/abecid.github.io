---
layout: page
title: Projects
permalink: /projects/
---

Projects

<ul>
    {% for cat in site.categories %}
        {% unless cat[0] == "Projects" %}
            {% continue %}
        {% endunless %}
        {% for post in cat[1] %}
            <li><a href="{{ post.url }}">{{ post.title }}</a></li>
        {% endfor %}
    {% endfor %}
</ul>
