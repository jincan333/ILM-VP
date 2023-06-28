#!/bin/sh

experiment_name='ablation_vp_hydra_hydra_weight_decay'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# ['imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# dataset=('ucf101' 'cifar10' 'cifar100' 'svhn' 'mnist' 'flowers102')
networks=('resnet18')
epochs=50
seed=7
prune_modes=('vp_ff')
input_size=192
pad_size=16
density_list='1,0.1'
prune_methods=('hydra')
datasets=('cifar10')

hydra_weight_decays=(0.00001 0.00005 0.0001 0.001)
gpus=(5 4 3 2)
for j in ${!networks[@]};do
    for l in ${!prune_methods[@]};do
        for i in ${!datasets[@]};do
            for k in ${!hydra_weight_decays[@]};do
                log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[0]}_${prune_methods[l]}_${hydra_weight_decays[k]}_${seed}.log
                    python ./core/vpns.py \
                        --experiment_name ${experiment_name} \
                        --dataset ${datasets[i]} \
                        --network ${networks[j]} \
                        --prune_mode ${prune_modes[0]} \
                        --prune_method ${prune_methods[l]} \
                        --input_size ${input_size} \
                        --pad_size ${pad_size} \
                        --density_list ${density_list} \
                        --hydra_weight_decay ${hydra_weight_decays[k]} \
                        --gpu ${gpus[k]} \
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
