.PHONY: all clean figures

all:
	latexmk -pdf -interaction=nonstopmode main.tex

figures:
	cd .. && python scripts/make_figures.py --tag $(TAG)

clean:
	latexmk -C
	rm -f *.bbl *.blg
