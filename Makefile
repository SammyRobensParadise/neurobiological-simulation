# makefile


.PHONY: help


help:
	@echo "--- Options ---"
	@echo "markdown ...... generates markdown files from  *.ipynb files"

MYDIR = .

markdown:
	jupyter nbconvert ./assignment-1/syde556_assignment_01_20709541.ipynb --to markdown --output assignment-1.md
	jupyter nbconvert ./assignment-2/syde556_assignment_02_20709541.ipynb --to markdown --output assignment-2.md
	jupyter nbconvert ./assignment-3/syde556_assignment_03_20709541.ipynb --to markdown --output assignment-3.md
	jupyter nbconvert ./assignment-3/syde556_assignment_04_20709541.ipynb --to markdown --output assignment-4.md
