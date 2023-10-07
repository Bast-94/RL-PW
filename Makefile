PYTHON=python3.10


testsuite: reformat
	$(PYTHON) -m pytest  exercices.py


tiny-testsuite: reformat
	$(PYTHON) tiny-test.py

reformat: 
	$(PYTHON) -m black .
	$(PYTHON) -m isort .

fast_commit: reformat
	sh pusher.sh

fast_push: fast_commit
	git push
	