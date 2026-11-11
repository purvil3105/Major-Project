# Paper draft (IEEE conference format)

- `main.tex`: IEEEtran conference paper.
- `refs.bib`: references, checked against the PDFs in `../papers` (and Crossref for Wu et al.).
- `figs/`: PDF figures, copied from `../nbiot_ntn_qlearning/figs/` (v2 results).
- `table_main.tex`, `table_versions.tex`: auto-generated tables (the paper inlines its own copy of Table I).
- `check_tex.py`: static checks (braces, environments, citations, figures), for use when no LaTeX compiler is installed.

## Build
LaTeX is not installed on this machine. Upload this folder to Overleaf (choose pdfLaTeX), or run locally:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Before submission
1. **Author block:** add your guide/co-authors, department, institution and e-mail (marked `TODO(author)`).
2. **Page limit:** check it against the venue. The draft is about 3 100 words with 6 figures, roughly 6–7 pages. If you need to cut, drop the Fig. 5 panel or move the sweep list into a table.
3. **Template:** apply the venue's template and copyright notice if it differs from plain IEEEtran.
4. **Citations:** do a last Google Scholar check for 2026 work on RL-based NB-IoT NTN access control. The literature review is a snapshot, not an exhaustive search.
5. **Figures:** if you re-run the simulations, copy the regenerated figures from `../nbiot_ntn_qlearning/figs/`.
