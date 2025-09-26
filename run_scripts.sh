#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)  # Add current folder to python PATH
clear
python src/scripts/generate_samples.py