#!/bin/sh

experiment_name='ablation_hydra_no_nettrain'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# dataset=('ucf101' 'cifar10' 'cifar100' 'svhn' 'mnist' 'flowers102')
networks=('resnet18')
datasets=('cifar10')
epochs=50
seed=7
density_list=('1,0.2')

prune_methods=('hydra')
prompt_methods=('pad')

prune_modes=('vp_ff' 'normal')
gpus=(7 6)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for l in ${!prune_methods[@]};do
            for k in ${!prune_modes[@]};do
                log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_methods[l]}_${prune_modes[k]}.log
                    python ./core/vpns.py \
                        --experiment_name ${experiment_name} \
                        --dataset ${datasets[i]} \
                        --network ${networks[j]} \
                        --prune_mode ${prune_modes[k]} \
                        --prune_method ${prune_methods[l]} \
                        --prompt_method ${prompt_methods[0]} \
                        --density_list ${density_list[0]} \
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
