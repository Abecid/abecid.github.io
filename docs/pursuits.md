---
layout: page
title: Other Pursuits
permalink: /pursuits/
---

Other Pursuits

<ul>
    {% for cat in site.categories %}
        {% unless cat[0] == "Other Pursuits" %}
            {% continue %}
        {% endunless %}
        {% for post in cat[1] %}
            <li><a href="{{ post.url }}">{{ post.title }}</a></li>
        {% endfor %}
    {% endfor %}
</ul>
