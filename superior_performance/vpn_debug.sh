#!/bin/sh

experiment_name='vpn_debug'
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
epochs=2
seed=7
prune_modes=('vp_ff')

second_phases=('freeze_vp+ff' 'vp+ff_cotrain' 'ff_then_vp')
prune_methods=('hydra' 'snip' 'synflow')
gpus=(3 2 1)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!second_phases[@]};do
            for l in ${!prune_methods[@]};do
                log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${second_phases[k]}_${prune_methods[l]}.log
                    python ./core/vpns_new.py \
                        --experiment_name ${experiment_name} \
                        --dataset ${datasets[i]} \
                        --network ${networks[j]} \
                        --second_phase ${second_phases[k]} \
                        --prune_mode ${prune_modes[0]} \
                        --prune_method ${prune_methods[l]} \
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
