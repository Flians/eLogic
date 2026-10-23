# ePMapping paper and academic talk

- `DATE27-ePMapping-sub/DATE27-ePMapping.tex`: current 6+1-page paper.
- `DATE27-ePMapping-sub/DATE27-ePMapping.pdf`: compiled paper snapshot.
- `talks/epmapping_30min/epmapping_30min.tex`: English 30-minute Beamer lecture with Chinese ready-to-deliver speaker notes (26 main slides and 8 backup slides).
- `talks/epmapping_30min/epmapping_30min.pdf`: compiled lecture.
- `talks/epmapping_30min/speaker_notes.md`: per-slide speaker notes and pacing.

The paper and talk use the verified `results_v28_three_parts_query_budget` campaign. Both retain the fixed initial delay constraint, disable whole-window synthesis, and use a 1000-second post-mapping budget.

## Compile

Keep the paper as the main document for the Overleaf paper project. Compile it with pdfLaTeX and BibTeX from `DATE27-ePMapping-sub`; its frozen tables and figures are checked in. The paper Makefile's data-refresh targets require the full local experiment repository.

Compile the talk separately with XeLaTeX: `cd talks/epmapping_30min && make`. It uses Noto Sans CJK SC, Noto Sans Mono CJK SC, and TeX Gyre Heros. It includes frozen data and does not need experiment files to compile. See the talk README for notes-only output and the 30-minute plan. Selecting the talk as the Overleaf main document also requires selecting XeLaTeX and ensuring these fonts are available.
