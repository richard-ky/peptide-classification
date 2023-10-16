#!/bin/bash
#
#SBATCH --job-name=fasta
#SBATCH --partition=condo
#SBATCH --output=output.txt
#
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-user=richard.ky@sjsu.edu
#SBATCH --mail-type=BEGIN,END

module load anaconda
module load cuda/12.2
conda run -n fastaenv python prott5_predictions.py 'Small Peptide Hits'
