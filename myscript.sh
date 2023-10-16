#!/bin/bash
#
#SBATCH --job-name=fasta
#SBATCH --partition=gpu
#SBATCH --output=output.txt
#
#SBATCH --cpus-per-task=56
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-user=richard.ky@sjsu.edu
#SBATCH --mail-type=BEGIN,END

module load python3

python prott5_predictions.py 'Small Peptide Hits'
