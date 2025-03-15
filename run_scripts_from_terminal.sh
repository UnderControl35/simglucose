#!/bin/bash

echo "Activating Anaconda environment..."

# Initialize Conda (needed if not already initialized in your shell)
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust path if necessary

# Activate the Anaconda environment (change 'myenv' to your environment name)
conda activate simbg
echo "Conda Path: $(which conda)"
echo "Python Path: $(which python)"

echo "Running Python scripts with arguments..."

python3 ./online-dt/odt_experiment.py
#python3 script1.py arg1 arg2
#python3 script2.py "$@"  # Pass all arguments to script2.py

echo "Done All Scripts!"
