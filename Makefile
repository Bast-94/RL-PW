testsuite:
	poetry run pytest
	date > testsuite.txt

tiny-testsuite:
	poetry run python trials.py