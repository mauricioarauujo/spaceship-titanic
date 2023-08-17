install:
	pip install -r src/requirements.txt

install-dev:
	make install
	pip install -r src/requirements-dev.txt
	pip install -r src/requirements-test.txt
	git init
	pre-commit install

lint:
	pre-commit

lint-all:
	pre-commit run --all-files

test:
	pytest