#!/bin/sh

experiment_name='ablation_vp_ff_hydra_vp_search'
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
prompt_methods=('pad')
hydra_lr=0.0001
vp_lr=0.01
ff_lr=0.01

# input_sizes=(224 192 160 128 96 64 32)
#pad_sizes=(16 32 48 64 80 96 112)
pad_sizes=(0)
input_sizes=(224 192 160 128)
gpus=(7 6 5 1)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for l in ${!prune_methods[@]};do
            for m in ${!pad_sizes[@]};do
                for k in ${!input_sizes[@]};do            
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_methods[l]}_${input_sizes[k]}_${pad_sizes[m]}.log
                        python ./core/vpns.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --prune_mode ${prune_modes[0]} \
                            --prune_method ${prune_methods[0]} \
                            --prompt_method ${prompt_methods[0]} \
                            --input_size ${input_sizes[k]} \
                            --pad_size ${pad_sizes[m]} \
                            --hydra_lr ${hydra_lr} \
                            --vp_lr ${vp_lr} \
                            --ff_lr ${ff_lr} \
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
