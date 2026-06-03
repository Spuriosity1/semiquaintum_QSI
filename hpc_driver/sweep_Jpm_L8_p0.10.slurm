#!/bin/bash

#SBATCH -J L8_j0p10
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
#SBATCH --array=1-5120
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


X=$(( $TASKID / 5 ))


L=8
DILUTION=0.10
Jzz=1.0
N_DISORDER_REALISATIONS=8
moves="CQBRW"
sweep=1024
samp=128
tcold=0.01

mkdir -p $OUT_PATH

# disorder realisation number
ds=$(( $X % $N_DISORDER_REALISATIONS + 900 ))
# Monte Carlo seed
ms=$(( $X / $N_DISORDER_REALISATIONS + 9000 ))

JpmArr=(-0.050 0.000 0.050 0.100 0.200)
Jpm=${JpmArr[ $(($TASKID % 5)) ]}

Jxx=$Jpm
Jyy=$Jpm

CMD="$EXE_DIR/sq_pyrochlore $L $DILUTION -o $OUT_PATH --Thot 100 --Tcold ${tcold} --Jzz $Jzz --Jxx $Jxx --Jyy $Jyy --nsweep $sweep --nburn ${burn} -s $ds -m $ms --moves $moves --prefix run_Tc${tcold}_${moves}_sw${sweep}_samp${samp} --n_replicas 1 --nsamp ${samp}"
echo "$CMD"
$CMD

