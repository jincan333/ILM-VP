#!/bin/sh

experiment_name='main_vpns'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# dataset=('ucf101' 'cifar10' 'cifar100' 'svhn' 'mnist' 'flowers102')
networks=('resnet18')
# datasets=('cifar100' 'flowers102' 'dtd' 'food101' 'oxfordpets')
datasets=('dtd' 'food101' 'oxfordpets')
epochs=120
prune_modes=('vp_ff')
prune_methods=('hydra')
density_list='1,0.1,0.01,0.001'
second_phases=('vp+ff_cotrain')

seeds=(7 9 17)
ff_optimizer='sgd'
ff_lr=0.01
gpus=(0 1 2)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for l in ${!prune_methods[@]};do
            for k in ${!second_phases[@]};do
                for m in ${!seeds[@]};do
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${second_phases[k]}_${prune_methods[l]}_${seeds[m]}_${ff_optimizer}_${ff_lr}.log
                        python ./core/vpns_new.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --second_phase ${second_phases[k]} \
                            --prune_mode ${prune_modes[0]} \
                            --prune_method ${prune_methods[l]} \
                            --density_list ${density_list} \
                            --ff_optimizer ${ff_optimizer} \
                            --ff_lr ${ff_lr} \
                            --gpu ${gpus[m]} \
                            --epochs ${epochs} \
                            --seed ${seeds[m]} \
                            > $log_filename 2>&1 &
                done
                wait
            done
            wait
        done
        wait
    done
    wait
done
