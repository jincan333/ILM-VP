#!/bin/sh

experiment_name='ablation_vp_ff_hydra_hydra_then_vp'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# dataset=('ucf101' 'cifar10' 'cifar100' 'svhn' 'mnist' 'flowers102')
networks=('resnet18')
datasets=('cifar10')
epochs=100
seed=7
prune_modes=('vp_ff')
prune_methods=('hydra')
prompt_methods=('pad')
density_list=('1,0.2,0.1,0.05')

gpus=(6)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for l in ${!prune_methods[@]};do           
            log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_methods[l]}.log
                python ./core/vpns.py \
                    --experiment_name ${experiment_name} \
                    --dataset ${datasets[i]} \
                    --network ${networks[j]} \
                    --prune_mode ${prune_modes[0]} \
                    --prune_method ${prune_methods[0]} \
                    --prompt_method ${prompt_methods[0]} \
                    --density_list ${density_list} \
                    --gpu ${gpus[k]} \
                    --epochs ${epochs} \
                    --seed ${seed} \
                    > $log_filename 2>&1 &
        done
        wait
    done
    wait
done
