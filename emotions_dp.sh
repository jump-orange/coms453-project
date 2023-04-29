#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G 
#SBATCH --time=0-36:00:00 
#SBATCH --gres=gpu:1 #If you just need one gpu, you're done, if you need more you can change the number
#SBATCH --partition=gpu #specify the gpu partition
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --job-name="cifar10_baseline" 
#SBATCH --output=%J.out
#SBATCH --error=%J.err 



cd /work/LAS/meisam-lab/feifei/
source dp-project/bin/activate
cd project




SIGMA=(1.1 1.5 2 4)
C=(0.1 0.5 1 2 3)
LR_SCH=(constant cos)

for var1 in ${SIGMA[@]};
do
    for var2 in ${C[@]};
    do
        for var3 in ${LR_SCH[@]};
        do    
            python3 main.py -sigma ${var1} -C ${var2} -lr-schedule ${var3} -logdir ./log_dp/
        done
    done
done