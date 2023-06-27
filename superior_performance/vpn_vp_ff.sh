#!/bin/sh

experiment_name='vp_ff'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# ['imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# dataset=('ucf101' 'cifar10' 'cifar100' 'svhn' 'mnist' 'flowers102')
networks=('resnet18')
epochs=100
seed=7
prune_modes=('vp_ff')
input_size=128
pad_size=48
density_list='1,0.10,0.05,0.01'

prune_methods=('imp' 'omp')
datasets=('cifar10')
gpus=(7)
for j in ${!networks[@]};do
    for l in ${!prune_methods[@]};do
        for i in ${!datasets[@]};do
            log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_methods[l]}_${seed}.log
                python ./core/vpns.py \
                    --experiment_name ${experiment_name} \
                    --dataset ${datasets[i]} \
                    --network ${networks[j]} \
                    --prune_mode ${prune_modes[0]} \
                    --prune_method ${prune_methods[l]} \
                    --input_size ${input_size} \
                    --pad_size ${pad_size} \
                    --density_list ${density_list} \
                    --gpu ${gpus[i]} \
                    --epochs ${epochs} \
                    --seed ${seed} \
                    > $log_filename 2>&1 &
        done
        wait
    done
    wait
done
