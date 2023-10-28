reformat:
	poetry run black .
	poetry run isort .

testsuite:
	poetry run pytest
	date > testsuite.txt

tiny-testsuite:
	poetry run python trials.py
	
taxi:
	poetry run python taxi.py

commit: reformat
	sh committer.sh

video:
	# check artifacts directory exists
	mkdir -p artifacts
	poetry run python video_maker.py