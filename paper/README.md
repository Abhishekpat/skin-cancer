# Skin Cancer Detection Using Deep Learning: IEEE Paper

| File | What it is |
|---|---|
| **sc_two_column.pdf** | Main deliverable: IEEE two-column paper (14 pages) **with measured results** of the 3-model ensemble (training curves, comparison charts, confusion matrix, ROC, ablation, external test) |
| **Training_and_Evaluation_Explained.pdf** | For the teacher (8 pages): which dataset each model used, how we trained, how accuracy / precision / recall / F1 / confusion matrix are calculated with worked examples from our numbers, real example images, and where each item is in the paper |
| **Training_Results_Report.pdf** | For the teacher (6 pages): all measured accuracy values, graphs, ablation (USP proof), external test, skin-type breakdown, conclusion |
| **Reference_Links.pdf** | All 71 references of the paper, numbered as in the paper ([1]–[71]), each with a clickable link (all checked 31 Oct 2026) |
| **sc_one_column.pdf** | Same content in one-column journal format |
| **Datasets_Report.pdf** | SIMPLE team guide (3 pages): what each dataset is, how we use it, and a link for every dataset |
| Datasets_Report_Detailed.pdf | Full detailed dataset survey (8 pages, charts, licences, references) |
| **Algorithms_Report.pdf** | SIMPLE team guide (5 pages): each algorithm in plain words, which ones we use, with paper and code links |
| Algorithms_Report_Detailed.pdf | Full detailed algorithm comparison (10 pages, 4 comparison panels, references) |
| **Paper_Explained_Simple.pdf** | For the team (7 pages): the IEEE paper explained section by section in plain words: every figure, table and equation, glossary, and 10 likely viva questions with answers |
| **Project_Flow.pdf** | For the guide (4 pages): what the project is, system flow, 7 development phases with timeline, paper-section-to-project mapping, tools, team roles, deliverables |
| **PAPER_GUIDE.md** | What each section means and why it is there; teacher's checklist mapping; reviewer assessment; datasets/algorithms summary; plagiarism advice |
| `sc_body.tex` | All paper content (title, abstract, sections, tables, figures) |
| `sc_preamble.tex` | Packages, colours, chart sizes |
| `sc_two_column.tex` / `sc_one_column.tex` | Layout wrappers |
| `references.bib` | Bibliography (every DOI/link checked) |
| `Datasets_Report*.tex`, `Algorithms_Report*.tex`, `Project_Flow.tex`, `report_preamble.tex` | Sources of the two reports (build with `tectonic <file>.tex`) |
| `build.ps1` | `./build.ps1` builds both PDFs (Tectonic / latexmk / pdflatex) |

**Overleaf:** upload all `.tex` files and `references.bib`, then set
`sc_two_column.tex` as the main document.

**Results:** Section VI of the paper contains our own measured results
(trained 16-17 Oct 2026 on an RTX 3060 laptop GPU). The figures are in
`figures/` and come from `../training/results/paper_assets/`.
Headline (internal ISIC 2019 test, 3-model ensemble): 77.95% accuracy,
62.99% balanced accuracy, macro-AUC 0.950, 95.4% malignant sensitivity.
