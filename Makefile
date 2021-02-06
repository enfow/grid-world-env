# Makefile
format:
	black .
	isort .

lint:
	env PYTHONPATH=. pytest --flake8 --pylint --mypy

utest:
	env PYTHONPATH=. pytest test/ -s --verbose

setup:
	pip install -r requirements.txt
	pre-commit install
