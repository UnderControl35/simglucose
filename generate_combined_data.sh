#!/bin/bash

echo "Activating Anaconda environment..."

# Initialize Conda (needed if not already initialized in your shell)
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust path if necessary

# Activate the Anaconda environment (change 'myenv' to your environment name)
conda activate simbg
echo "Conda Path: $(which conda)"
echo "Python Path: $(which python)"

echo "Running Python scripts with arguments..."

python ./dataset/T1DatasetAnalysis/trajectory_creator.py --noisy True
python ./dataset/T1DatasetAnalysis/trajectory_creator.py --noisy False
python ./dataset/T1DatasetAnalysis/data_analysis_noisy.py --method BB --patient adolescent#001 --index 0
python ./dataset/T1DatasetAnalysis/data_analysis_noisy.py --method PID --patient adolescent#001 --index 0
python ./dataset/T1DatasetAnalysis/data_analysis_noisy.py --method PPO --patient adolescent#001 --index 0
#python3 script2.py "$@"  # Pass all arguments to script2.py

echo "Datasets has been created noisy and noiseless!"
