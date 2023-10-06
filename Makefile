PYTHON=python3.10


testsuite: reformat
	$(PYTHON) -m pytest -q exercices.py


tiny-testsuite: reformat
	$(PYTHON) tiny-test.py

reformat: 
	$(PYTHON) -m black .
	$(PYTHON) -m isort .
	