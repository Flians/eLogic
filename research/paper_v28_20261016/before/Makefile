ENGINE := pdflatex
BIB := bibtex

PROJECT := DATE27-ePMapping
RESULTS ?= results_v27_three_parts_build_reuse
.DEFAULT_GOAL := all

.PHONY: all fast clean verify-data verify-mfs

verify-data:
	cd ../.. && python3 scripts/experiments/report_three_parts_profiles.py --root $(RESULTS) --out epmapping_paper/DATE27-ePMapping-sub/doc/generated_results.tex
	cd ../.. && python3 scripts/experiments/plot_paper.py --root $(RESULTS)/cold --out epmapping_paper/DATE27-ePMapping-sub/figs/results
	cd ../.. && python3 scripts/experiments/plot_cofactor_ablation.py --root $(RESULTS) --out epmapping_paper/DATE27-ePMapping-sub/figs/results
	cd ../.. && python3 scripts/experiments/plot_profile_cache.py --root $(RESULTS) --out epmapping_paper/DATE27-ePMapping-sub/figs/results
	cd ../.. && python3 scripts/experiments/plot_build_ablation.py --root $(RESULTS) --out epmapping_paper/DATE27-ePMapping-sub/figs/results

verify-mfs: verify-data

all: verify-data
	$(ENGINE) $(PROJECT).tex
	$(BIB)    $(PROJECT)
	$(BIB)    $(PROJECT)
	$(ENGINE) $(PROJECT).tex
	$(ENGINE) $(PROJECT).tex

fast: verify-data
	$(ENGINE) $(PROJECT).tex

clean:
	rm -rf *.out *.bbl *.aux *.log *.blg \
		*.log *.gz *.thm *.ps *.dvi *.dat \
		doc/*-diff.tex *.fls *.fdb_latexmk \
		*.xdv
