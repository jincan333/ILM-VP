#!/bin/sh

experiment_name='ablation_normal_hydra_lr_search'
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
prune_modes=('normal')
prune_methods=('hydra')
prompt_methods=('pad')
# input_sizes=(224 192 160 128 96 64 32)
input_sizes=(128)
# pad_sizes=(16 32 48 64 80 96 112)
pad_sizes=(48)

hydra_lrs=(0.00001 0.000001)
vp_lrs=(0.01)
gpus=(6 5)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for l in ${!prune_methods[@]};do
            for m in ${!vp_lrs[@]};do
                for k in ${!hydra_lrs[@]};do            
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_methods[l]}_${hydra_lrs[k]}_${vp_lrs[m]}.log
                        python ./core/vpns.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --prune_mode ${prune_modes[0]} \
                            --prune_method ${prune_methods[0]} \
                            --prompt_method ${prompt_methods[0]} \
                            --hydra_lr ${hydra_lrs[k]} \
                            --vp_lr ${vp_lrs[m]} \
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
    wait
done
