
.PHONY: help install test video

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install    to install the package"
	@echo "  test       to run the tests"

clean: 
	rm -rf build dist *.egg-info .pytest_cache .coverage .benchmarks .mypy_cache .tox .hypothesis

# video: 
# 	ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4

install:
	./scripts/install.sh

test:
	python -m pytest -s -v -n auto --dist=loadfile --junitxml=tests.xml --no-cov --benchmark-disable

benchmark:
	python -m pytest -s -v -n 0 --no-cov benchmarks

coverage:
	python -m pytest --cov=sources --cov-report=html --cov-report=xml --benchmark-disable

build: clean
	python -m build

dist: build
	python -m twine check dist/*
	python -m twine upload dist/*