---
title: DeepVibes
summary: Estimating and correcting micro-vibrations in satellite images acquired with pushbroom cameras, validated on Sentinel-2B data.
role: Research engineer, Agenium Space
years: 2022 – 2023
image: publication_preview/vibration.gif
show_image: false # the interactive demo below replaces the header image on this page
order: 3
links:
  - name: Paper
    url: https://doi.org/10.1109/TGRS.2024.3415372
  - name: Interactive demo
    url: /projects/deepvibes/#demo
---

Pushbroom cameras build an image line by line as the satellite moves along its orbit. Tiny oscillations of the
satellite's attitude during acquisition, the _micro-vibrations_, make each line look in a slightly different direction:
straight roads become wavy and the spectral bands, which sit at different places on the focal plane, no longer line up.

DeepVibes models the acquisition process with the satellite attitude as a parameter, then corrects the images in two
stages: an identification network estimates the attitude, and a deep unrolled network uses it to reconstruct the image.
We found that the geometry of the focal plane plays a key role in making the vibrations identifiable.

## Interactive demo

{% include compare.html id="demo" before="deepvibes/vibrated.webp" after="deepvibes/reconstruction.webp" before_label="Vibrated" after_label="DeepVibes" focus="0.47,0.47" caption="Sentinel-2B scene. Drag the divider to compare the image affected by micro-vibrations (left) with the DeepVibes reconstruction (right); zoom in to see the colour fringes and wobbly edges disappear." %}
