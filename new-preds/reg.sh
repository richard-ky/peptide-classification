#!/bin/bash
#
#SBATCH --job-name=bacteria-preds
#SBATCH --partition=condo
#
#SBATCH --cpus-per-task=4
#SBATCH --time=4-0
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-user=richard.ky@sjsu.edu
#SBATCH --mail-type=BEGIN,END

module load anaconda
module load cuda/12.2
conda run -n fastaenv python reg_script.py
