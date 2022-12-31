---
layout: page
title: Research
permalink: /research/
---

Research

<ul>
    {% for cat in site.categories %}
        {% unless cat[0] == "Research" %}
            {% continue %}
        {% endunless %}
        {% for post in cat[1] %}
            {% if post.tags contains "Hidden" %}
                {% continue%}
            {% endif %}
            <li><a href="{{ post.url }}">{{ post.title }}</a></li>
        {% endfor %}
    {% endfor %}
</ul>
