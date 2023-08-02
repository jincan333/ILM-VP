#!/bin/sh

experiment_name='ablation_vp_hydra_ff_sgd_0.01'
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
prune_modes=('vp_ff')
prune_methods=('hydra')
density_list='1,0.01,0.001'
second_phases=('vp+ff_cotrain')

# input_sizes=(224 192 160 128 96 64 32)
#pad_sizes=(16 32 48 64 80 96 112)

ff_optimizer='sgd'
ff_lr=0.01
gpus=(5)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for l in ${!prune_methods[@]};do              
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_methods[l]}_${seed}.log
                        python ./core/vpns_new.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --prune_mode ${prune_modes[0]} \
                            --prune_method ${prune_methods[0]} \
                            --density_list ${density_list} \
                            --second_phase ${second_phases[0]} \
                            --ff_optimizer ${ff_optimizer} \
                            --ff_lr ${ff_lr} \
                            --gpu ${gpus[m]} \
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
    wait
done
