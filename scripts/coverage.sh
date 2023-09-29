#!/bin/bash

python -m pytest --cov-report xml:coverage.xml --cov sources --cov-fail-under 0 --cov-append $@

exit