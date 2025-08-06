---
layout: page
permalink: /publications/
title: publications and talks
description: Below is the list of my publications and talks/presentations
nav: true
nav_order: 2
---

{% include bib_search.liquid %}

## Publications

<div class="publications">
    {% bibliography %}
</div>

## Posters

<div class="publications">
    {% bibliography --file posters.bib %} 
</div>

## Talks

<div class="publications">
    {% bibliography --file talks.bib %}
</div>
