---
title: Blind deblurring in fluorescence microscopy
summary: Jointly estimating the sharp image and the unknown blur from a single fluorescence microscopy image, with deep learning and diffusion priors.
role: PhD project
years: Ongoing
image: publication_preview/deepblur_estimated_psf.png
order: 2
links:
  - name: Code
    url: https://github.com/mh-nguyen712/blind_deblur_diffusion
  - name: Paper
    url: https://arxiv.org/abs/2508.02923
  - name: Poster
    url: /assets/pdf/poster_iabm_2025.pdf
---

- Covers diffraction-limited blurs (Zernike polynomials), spatially varying blurs, and 2D/3D settings.
- Efficient PyTorch implementation, with ongoing validation on RIM, TIRF and confocal microscopes.
- Theory: a Morse-Bott analysis of MAP estimation with learned priors, explaining when gradient-based methods recover the right image and why the "blurry trap" persists.
- Joint work with Pierre Weiss, Florian Sarron, Paul Escande, Thomas Mangeat and Sylvain Cantaloube. Presented at [MIFOBIO 2025](https://imabio-cnrs.fr/mifobio-2025/).
