#!/bin/sh

experiment_name='ablation_vp_ff_hydra_sigmoid_vp_optimizer_lr'
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
input_size=192
pad_size=16
density_list='1,0.20'
prune_methods=('hydra')

vp_optimizers=('sgd')
vp_schedulers=('cosine')
vp_lrs=(0.01 0.001 0.0001)
gpus=(7 6 5)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_methods[@]};do
            for l in ${!vp_optimizers[@]};do
                for m in ${!vp_schedulers[@]};do
                    for n in ${!vp_lrs[@]};do
                        log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_methods[k]}_${vp_optimizers[l]}_${vp_schedulers[m]}_${vp_lrs[n]}.log
                            python ./core/vpns.py \
                                --experiment_name ${experiment_name} \
                                --dataset ${datasets[i]} \
                                --network ${networks[j]} \
                                --prune_mode ${prune_modes[0]} \
                                --prune_method ${prune_methods[k]} \
                                --input_size ${input_size} \
                                --pad_size ${pad_size} \
                                --density_list ${density_list} \
                                --vp_optimizer ${vp_optimizers[l]} \
                                --vp_scheduler ${vp_schedulers[m]} \
                                --vp_lr ${vp_lrs[n]} \
                                --gpu ${gpus[n]} \
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
    wait
done
