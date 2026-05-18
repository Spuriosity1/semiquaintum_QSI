#!/bin/bash

#SBATCH -J bmark
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


export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


echo -e "Working in `pwd`.\n"

JOBID=$SLURM_JOB_ID
TASKID=$SLURM_ARRAY_TASK_ID

OUT_PATH="../tmp/bmark"
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

Jzz=1.0
Jxx=0.05
Jyy=0.05

mkdir -p $OUT_PATH

ds=0
ms=$(( $SLURM_ARRAY_TASK_ID + 400 ))
moves="CQBRW"
L=$(( 2 * ( $SLURM_ARRAY_TASK_ID % 3 + 1)  ))

for p in 0 0.01 0.05 0.1; do
    CMD="$EXE_DIR/sq_pyrochlore --moves $moves $L $p -o $OUT_PATH --Thot 100 --Tcold 0.001 --Jzz $Jzz --Jxx $Jxx --Jyy $Jyy --nsweep 16 --nburn 512 -s $ds -m $ms --n_replicas 100 --n_replica_swaps 16 --prefix bm_r"
    echo "$CMD"
    time $CMD
done
