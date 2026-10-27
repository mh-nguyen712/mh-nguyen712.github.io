# mh-nguyen712.github.io

Personal website of Minh Hai Nguyen. A small custom Jekyll site, built and deployed to the `gh-pages` branch by
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) on every push to `master`.

## Adding content

Every kind of content lives in exactly one place. The pages (and the CV) pick it up automatically.

| What                | Where                                       |
| ------------------- | ------------------------------------------- |
| News item           | `_data/news.yml`                            |
| Publication         | `_bibliography/papers.bib`                  |
| Talk / poster       | `_data/talks.yml`                           |
| Project             | a new file in `_projects/`                  |
| Teaching            | `_data/teaching.yml`                        |
| CV-only sections    | `_data/cv.yml` (education, experience, ...) |
| Bio, photo, address | `_pages/about.md`                           |
| Name, email, links  | `author:` in `_config.yml`                  |
| Menu                | `nav:` in `_config.yml`                     |

### News

Add an entry at the **top** of `_data/news.yml` (newest first). The text is Markdown:

```yaml
- date: 2026-11-15
  text: >-
    Our paper "..." was accepted at [NeurIPS](https://neurips.cc/).
```

The home page shows the latest 5 (`home_news_limit` in `_config.yml`). `/news/` shows everything.

### Publications

Paste the BibTeX entry (from Google Scholar, arXiv, ...) into `_bibliography/papers.bib`. Optional extra fields:

```bibtex
  abbr        = {ICML},                % badge, and placeholder thumbnail if there is no preview
  selected    = {true},                % also show on the home page
  preview     = {my_figure.png},       % thumbnail, in assets/img/publication_preview/
  pdf         = {my_paper.pdf},        % file in assets/pdf/, or a full URL
  code        = {https://github.com/...},
  website     = {https://...},
  slides      = {my_slides.pdf},       % file in assets/pdf/, or a full URL
  poster      = {my_poster.pdf},
  html        = {demo.html},           % interactive demo in assets/html/, or a full URL
  bibtex_show = {true},                % adds a "BibTeX" button
```

`eprint` (arXiv id) and `doi` become arXiv/DOI links automatically. Your name is shown in bold.

### Talks and posters

Add an entry at the **top** of `_data/talks.yml`:

```yaml
- date: 2026-11 # YYYY-MM or YYYY-MM-DD
  type: Talk # Talk, Poster, Invited talk, ...
  title: My talk title
  event: Conference name
  event_url: https://...
  location: Paris, France
  slides: my_slides.pdf # optional, in assets/pdf/ or a full URL
```

### Projects

Create `_projects/my-project.md`:

```markdown
---
title: My project
summary: One or two sentences shown on the card.
role: PhD project
years: 2025 – present
image: publication_preview/my_figure.png # in assets/img/
image_fit: contain # optional: show the whole image (logos) instead of cropping
order: 4 # position on the Projects page
links:
  - name: Code
    url: https://github.com/...
---

Optional longer description in Markdown. If this is not empty, the card gets a "More →" link
to a page of its own at /projects/my-project/.
```

## Running locally

```bash
bundle install
bundle exec jekyll serve --livereload
```

Then open http://localhost:4000.

## Structure

```
_config.yml          site settings, author links, menu
_data/               news, talks, teaching, CV content
_bibliography/       papers.bib
_projects/           one Markdown file per project
_pages/              the pages (home, news, publications, projects, teaching, cv)
_layouts/            default, page, home, project, bib (one publication), redirect
_includes/           header, footer, news, talks, project card, ...
assets/css/main.scss the whole stylesheet (colours at the top; dark is the default)
assets/js/site.js    theme toggle + abstract/BibTeX buttons
```
