#!/bin/sh

experiment_name='normal'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# datasets=("cifar100" "dtd" "flowers102" "ucf101" "food101" "gtsrb" "svhn" "eurosat" "oxfordpets" "stanfordcars" "sun397")
# datasets=("ucf101" "eurosat" "oxfordpets" "stanfordcars" "sun397") 
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# dataset=('cifar10' 'cifar100' 'svhn' 'mnist' 'flowers102' 'ucf101')
networks=('resnet18')
datasets=('cifar100' 'svhn' 'mnist' 'flowers102')
epochs=100
seed=7

prune_modes=('normal')
prune_methods=('hydra' 'imp' 'omp' 'random' 'snip' 'synflow' 'grasp')
gpus=(3 2 1 0 3 2 1 0)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for l in ${!prune_methods[@]};do
                log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}.log
                    python ./core/vpns.py \
                        --experiment_name ${experiment_name} \
                        --dataset ${datasets[i]} \
                        --network ${networks[j]} \
                        --prune_method ${prune_methods[l]} \
                        --prune_mode ${prune_modes[k]} \
                        --gpu ${gpus[l]} \
                        --epochs ${epochs} \
                        --seed ${seed} \
                        > $log_filename 2>&1 &
            done
            wait
        done
        wait
    done
    wait
done
