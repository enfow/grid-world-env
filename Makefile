# Makefile
format:
	black .
	isort .

lint:
	env PYTHONPATH=./src/ pytest src --flake8 --pylint --mypy

utest:
	env PYTHONPATH=./src/ pytest test/ -s --verbose

setup:
	pip install -r requirements.txt
	pre-commit install
