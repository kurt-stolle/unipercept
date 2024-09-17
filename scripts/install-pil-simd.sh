#!/bin/bash

pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
