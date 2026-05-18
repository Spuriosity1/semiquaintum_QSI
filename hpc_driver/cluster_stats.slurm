#!/bin/bash

#SBATCH -J clstat
#SBATCH --nodes=1
#! number of MPI tasks
#SBATCH --ntasks=1
#SBATCH --mail-user=alaric@zedat.fu-berlin.de   # Email to which notifications will be sent 
#! How much wallclock time will be required?
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=END
#SBATCH --output=hpc_driver/track/%x.%A_%a.out
#SBATCH --error=hpc_driver/track/%x.%A_%a.error
#SBATCH --chdir=/scratch/alaric/semiquaintum_QSI
#SBATCH --array=0-127
##SBATCH --constraint='uarch:x86-64-v3|uarch:x86-64-v4'

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load intel-oneapi/no-python

L=20

ds=$(( $SLURM_ARRAY_TASK_ID + 1000 ))

for p in 0.01 0.02 0.03 0.04 0.05; do
for cdef in nn24 nn2_mf; do
    CMD="build/percol/sq_pyrochlore_dump $L $p -o ../out/percol_CMC/ -s $ds --nsweep $sweep --cluster_def $cdef"
    echo "$CMD"
    $CMD
done
done
