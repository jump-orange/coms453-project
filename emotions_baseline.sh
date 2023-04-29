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


lr=(0.01 0.1 0.5 1)
batch=(256 512 1024)
net=(resnet8 resnet14 resnet20 CNN)

for var1 in ${lr[@]};
do
    for var2 in ${batch[@]};
    do
        for var3 in ${net[@]};
        do

            python3 main.py -lr ${var1} -batch ${var2} -net ${var3} -disable-dp True

        done
    done
done