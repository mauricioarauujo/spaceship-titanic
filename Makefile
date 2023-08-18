install:
	pip install -r src/requirements.txt

install-dev:
	make install
	git init
	pre-commit install

lint:
	pre-commit

lint-all:
	pre-commit run --all-files

test:
	pytest