reformat:
	poetry run black .
	poetry run isort .

testsuite:
	poetry run pytest

tiny-testsuite:
	poetry run python trials.py
	
taxi:
	poetry run python taxi.py

produced_files = $(wildcard artifacts/*.gif) $(wildcard artifacts/*.png) $(wildcard artifacts/*.jpg)

save_produced: reformat taxi	
	mkdir -p img
	mv $(produced_files) -t img

commit: reformat
	sh committer.sh

push: commit
	git push

quick_commit:
	sh committer.sh quick

quick_push: quick_commit
	git push
