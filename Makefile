
.PHONY: help install test

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install    to install the package"
	@echo "  test       to run the tests"

install:
	./scripts/install.sh

test:
	python -m pytest -s -v -n auto --dist=loadfile --junitxml=tests.xml --no-cov --benchmark-disable

benchmark:
	python -m pytest -s -v -n 0 --no-cov benchmarks

coverage:
	python -m pytest --cov=sources --cov-report=html --cov-report=xml --benchmark-disable
