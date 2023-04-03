#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --exclude=linux[1-50]
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --output=/usr/xtmp/jl888/FasterRisk/jupyter.log

conda activate FasterRisk

cat /etc/hosts
jupyter notebook --no-browser --ip=0.0.0.0 --port=9999