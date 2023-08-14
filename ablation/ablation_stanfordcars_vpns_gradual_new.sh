#!/bin/sh

experiment_name='ablation_stanfordcars_vpns_gradual_new'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# dataset=('ucf101' 'cifar10' 'cifar100' 'svhn' 'mnist' 'flowers102')
networks=('resnet18')
# datasets=('cifar100' 'flowers102' 'dtd' 'food101' 'oxfordpets')
datasets=('stanfordcars')
epochs=60
prune_modes=('vp_ff')
prune_methods=('hydra')
density_list='1,0.1'
second_phases=('vp+ff_cotrain')

seeds=(7)
ff_optimizer='adam'
ff_lr=0.001
vp_optimizer='sgd'
vp_lr=0.001
hydra_lr=0.0001
gpus=(2)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for l in ${!prune_methods[@]};do
            for k in ${!second_phases[@]};do
                for m in ${!seeds[@]};do
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${second_phases[k]}_${prune_methods[l]}_${seeds[m]}_${ff_optimizer}_${ff_lr}_${vp_optimizer}_${vp_lr}_${hydra_lr}.log
                        python ./core/vpns_gradual.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --second_phase ${second_phases[k]} \
                            --prune_mode ${prune_modes[0]} \
                            --prune_method ${prune_methods[l]} \
                            --density_list ${density_list} \
                            --ff_optimizer ${ff_optimizer} \
                            --ff_lr ${ff_lr} \
                            --vp_optimizer ${vp_optimizer} \
                            --vp_lr ${vp_lr} \
                            --hydra_lr ${hydra_lr} \
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
