#!/bin/bash

#SBATCH -J us_0_25 # job name
#SBATCH -o sbatch_output_log/output_%x_%j.out # standard output and error log
#SBATCH -p 2080ti # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:1
#SBATCH  --nodes=1
#SBATCH  --ntasks=4

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date

sh us_0_25.sh

date
