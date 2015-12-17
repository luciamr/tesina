#
#   Makefile
#

MAINTEX = propuesta
CP = cp
RM = rm -f
LATEX = latex
PDFLATEX = pdflatex
LHS = lhs2TeX
BIB = biber
LHSPAR = --poly
LHSFILES = $(MAINTEX)
EXT = *.nav *.snm *.ptb *.blg *.log *.aux *.lof *.lot *.bit *.idx *.glo *.bbl *.ilg *.toc *.out *.ind *~ *.ml* *.mt* *.th* *.bmt *.xyc *.bcf *.run.xml
PAG =
NUM =

default: pdflatex

all: dvi ps pdf

dvi: 	$(LATEX) $(MAINTEX).tex
	#bibtex $(MAINTEX)

	@latex_count=5 ; \
	while egrep -s '(Rerun (LaTeX|to get cross-references right)|rerun LaTeX)' $(MAINTEX).log && [ $$latex_count -gt 0 ] ;\
	    do \
	      echo "Rerunning latex...." ;\
	      $(LATEX) $(MAINTEX).tex ;\
	      latex_count=`expr $$latex_count - 1` ;\
	    done

ps: dvi
	dvips -f $(MAINTEX).dvi > $(MAINTEX).ps

pdflatex: 
	$(PDFLATEX) $(MAINTEX).tex
	$(BIB) $(MAINTEX)
	@latex_count=5 ; \
	while egrep -s 'Rerun (LaTeX|to get cross-references right)' $(MAINTEX).log && [ $$latex_count -gt 0 ] ;\
	    do \
	      echo "Rerunning latex...." ;\
	      $(PDFLATEX) $(MAINTEX).tex ;\
	      latex_count=`expr $$latex_count - 1` ;\
	    done

clean:
	$(RM) $(EXT)


clean-all: clean
	$(RM) *.dvi
	$(RM) PS/*.ps
	$(RM) PDF/*.pdf

tar: clean
	alias NOMBRE="basename `pwd`";\
	tar -cvjf `NOMBRE`.tar.bz2\
	        --exclude "*.bz2"\
	        --exclude "*.dvi"\
		--exclude "*.tar.bz2"\
	        ../`NOMBRE`/ ;\
	unalias NOMBRE

help:
	@echo "    make dvi"
	@echo "    make all           -- tex dvi ps pdf"
	@echo "    make ps"
	@echo "    make pdflatex      -- default"
	@echo "    make clean"
	@echo "    make clean-all"
	@echo "    make tar"

