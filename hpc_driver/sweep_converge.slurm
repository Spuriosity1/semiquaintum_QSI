#!/bin/bash

#SBATCH -J TEST_nsw
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
#SBATCH --output=hpc_driver/track/c_sweep.%A_%a.out
#SBATCH --error=hpc_driver/track/c_sweep.%A_%a.error
#SBATCH --chdir=/scratch/alaric/semiquaintum_QSI
#SBATCH --array=0-19
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

OUT_PATH="../out/sq_qsi/bench"
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

L=4
DILUTION=0.05
Jzz=1.0
N_DISORDER_REALISATIONS=1
moves="CQBRW"
n_replicas=100

mkdir -p $OUT_PATH

# disorder realisation number
ds=0
# Monte Carlo seed
ms=0

Jpm=0.05

Jxx=$Jpm
Jyy=$Jpm

burnvals=(64 128 256 512)
sweepvals=(8 16 32 64 128)

burn_id=$(( $SLURM_ARRAY_TASK_ID % 4 ))
sweep_id=$(( $SLURM_ARRAY_TASK_ID / 4 ))

burn=${burnvals[${burn_id}]}
sweep=256
replica_swaps=${sweepvals[${sweep_id}]}

CMD="$EXE_DIR/sq_pyrochlore $L $DILUTION -o $OUT_PATH --Thot 100 --Tcold 0.01 --Jzz $Jzz --Jxx $Jxx --Jyy $Jyy --nsweep $sweep --nburn ${burn} -s $ds -m $ms --moves $moves --prefix bm_r${n_replicas}_rs${replica_swaps}_${moves}_sw${sweep}_burn${burn} --n_replicas ${n_replicas} --n_replica_swaps ${replica_swaps} --ignore_ssf"
echo "$CMD"
$CMD
