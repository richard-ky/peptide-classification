#!/bin/bash
#
#SBATCH --job-name=fasta
#SBATCH --partition=gpu
#SBATCH --output=output.txt
#
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-user=richard.ky@sjsu.edu
#SBATCH --mail-type=BEGIN,END

module load anaconda
conda run -n fastaenv python prott5_predictions.py 'Small Peptide Hits'
