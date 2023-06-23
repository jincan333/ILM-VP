#!/bin/sh

experiment_name='ablation_normal_ff_lr'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# datasets=("cifar100" "dtd" "flowers102" "ucf101" "food101" "gtsrb" "svhn" "eurosat" "oxfordpets" "stanfordcars" "sun397")
# datasets=("ucf101" "eurosat" "oxfordpets" "stanfordcars" "sun397") 
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# dataset=('cifar10' 'cifar100' 'svhn' 'mnist' 'flowers102' 'ucf101')
networks=('resnet18')
datasets=('cifar10')
epochs=100
seed=7
prune_modes=('normal')
prune_methods=('imp')

ff_lrs=(0.1 0.001)
gpus=(4 2)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!ff_lrs[@]};do
            log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${ff_lrs[k]}.log
                python ./core/vpns.py \
                    --experiment_name ${experiment_name} \
                    --dataset ${datasets[i]} \
                    --network ${networks[j]} \
                    --prune_method ${prune_methods[0]} \
                    --prune_mode ${prune_modes[0]} \
                    --ff_lr ${ff_lrs[k]} \
                    --gpu ${gpus[k]} \
                    --epochs ${epochs} \
                    --seed ${seed} \
                    > $log_filename 2>&1 &
        done
        wait
    done
    wait
done
