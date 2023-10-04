PYTHON=python3.10


testsuite:
	$(PYTHON) -m pytest exercices.py

tiny-testsuite:
	$(PYTHON) tiny-test.py