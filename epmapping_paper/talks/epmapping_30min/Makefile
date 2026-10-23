LATEX ?= xelatex
.PHONY: all refresh-data notes clean
all:
	$(LATEX) -interaction=nonstopmode -halt-on-error epmapping_30min.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error epmapping_30min.tex
refresh-data:
	python3 build_data.py
notes:
	$(LATEX) -interaction=nonstopmode -halt-on-error -jobname=epmapping_notes '\def\printnotes{1}\input{epmapping_30min.tex}'
clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb
