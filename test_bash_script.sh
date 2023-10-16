#!/bin/bash
#
#SBATCH --job-name=richard_test
#SBATCH --partition=gpu
#SBATCH --output=output.txt
#
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-user=richard.ky@sjsu.edu
#SBATCH --mail-type=BEGIN,END

module load python3

python file_write_test.py
