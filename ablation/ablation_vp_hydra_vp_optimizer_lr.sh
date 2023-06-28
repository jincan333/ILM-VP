#!/bin/sh

experiment_name='ablation_vp_hydra_vp_optimizer_lr'
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
density_list='1,0.01'
prune_methods=('hydra')
datasets=('cifar10')

second_phases=('vp+ff_cotrain')
vp_optimizers=('adam')
vp_lrs=(0.01 0.001 0.0001)
gpus=(4 3 2)
for j in ${!networks[@]};do
    for l in ${!prune_methods[@]};do
        for i in ${!datasets[@]};do
            for k in ${!vp_lrs[@]};do
                log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[0]}_${prune_methods[l]}_${vp_optimizers[0]}_${vp_lrs[k]}_${seed}.log
                    python ./core/vpns_new.py \
                        --experiment_name ${experiment_name} \
                        --dataset ${datasets[i]} \
                        --network ${networks[j]} \
                        --prune_mode ${prune_modes[0]} \
                        --prune_method ${prune_methods[l]} \
                        --input_size ${input_size} \
                        --pad_size ${pad_size} \
                        --density_list ${density_list} \
                        --second_phase ${second_phases[0]} \
                        --vp_optimizer ${vp_optimizers[0]} \
                        --vp_lr ${vp_lrs[k]} \
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
