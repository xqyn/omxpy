# omxpy/Makefile

PYTHON = python
MAKE_INITS = src/dev/make_inits.py

.PHONY: make_inits

make_inits:
	$(PYTHON) $(MAKE_INITS)