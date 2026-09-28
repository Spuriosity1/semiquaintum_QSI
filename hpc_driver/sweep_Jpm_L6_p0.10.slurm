#!/bin/bash

#SBATCH -J L6_j0p10
#SBATCH --nodes=1
#! number of MPI tasks
#SBATCH --ntasks=1
#SBATCH --mail-user=alaric@zedat.fu-berlin.de   # Email to which notifications will be sent 
#! How much wallclock time will be required?
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=END
#SBATCH --output=hpc_driver/track/%x.%A_%a.out
#SBATCH --error=hpc_driver/track/%x.%A_%a.error
#SBATCH --chdir=/scratch/alaric/semiquaintum_QSI
#SBATCH --array=0-1024
##SBATCH --constraint='uarch:x86-64-v3|uarch:x86-64-v4'

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load intel-oneapi/no-python


export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


echo -e "Working in `pwd`.\n"

JOBID=$SLURM_JOB_ID
TASKID=$SLURM_ARRAY_TASK_ID

OUT_PATH="../out/sq_qsi"
EXE_DIR="build/percol/"

echo -e "TaskID: $TASKID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

#if [ "$SLURM_JOB_NODELIST" ]; then
#        #! Create a machine file:
#        export NODEFILE=`generate_pbs_nodefile`
#        cat $NODEFILE | uniq > machine.file.$JOBID
#        echo -e "\nNodes allocated:\n================"
#        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
#fi
#

L=6
DILUTION=0.10
Jzz=1.0
N_DISORDER_REALISATIONS=8
moves="CQBRW"
sweep=256
samp=128

mkdir -p $OUT_PATH

# disorder realisation number
ds=$(( $SLURM_ARRAY_TASK_ID % $N_DISORDER_REALISATIONS + 200 ))
# Monte Carlo seed
ms=$(( $SLURM_ARRAY_TASK_ID / $N_DISORDER_REALISATIONS + 3000 ))

for Jpm in -0.050 0.000 0.050 0.100 0.200; do

Jxx=$Jpm
Jyy=$Jpm

CMD="$EXE_DIR/sq_pyrochlore $L $DILUTION -o $OUT_PATH --Thot 100 --Tcold 0.01 --Jzz $Jzz --Jxx $Jxx --Jyy $Jyy --nsweep $sweep --nburn ${burn} -s $ds -m $ms --moves $moves --prefix bm_an_${moves}_sw${sweep}_samp${samp} --n_replicas 1 --nsamp ${samp}"
echo "$CMD"
$CMD
echo "done"

done
