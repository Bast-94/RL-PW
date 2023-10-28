all: reformat testsuite save_produced

reformat:
	@poetry run black .
	@poetry run isort .

testsuite:
	@poetry run pytest

tiny-testsuite:
	@poetry run python trials.py
	
taxi:
	@poetry run python taxi.py

produced_files = $(wildcard artifacts/*.gif) $(wildcard artifacts/*.png) $(wildcard artifacts/*.jpg)


save_produced: reformat 	
	@mkdir -p img
	@make taxi
	

commit: reformat
	@sh committer.sh

push: commit
	@git push

quick_commit:
	@sh committer.sh quick

quick_push: quick_commit
	@git push

full_test: reformat testsuite save_produced
	# echo in green
	@echo "\033[0;32mAll tests passed\033[0m"

clean:
	@rm -rf img
	
