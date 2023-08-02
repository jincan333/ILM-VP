#!/bin/sh

experiment_name='vpns_dataset_transfer'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# datasets=("cifar100" "dtd" "flowers102" "ucf101" "food101" "gtsrb" "svhn" "eurosat" "oxfordpets" "stanfordcars" "sun397", "tiny_imagenet")
# datasets=("ucf101" "eurosat" "oxfordpets" "stanfordcars" "sun397") 
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# datasets=('cifar100' 'flowers102' 'dtd' 'food101' 'oxfordpets')
networks=('resnet18')
# datasets=('cifar100' 'flowers102' 'dtd' 'food101' 'oxfordpets')
datasets=('tiny_imagenet' 'cifar100')
epochs=120
# seed 7 9 17
density_list='1,0.1,0.01,0.001'
prune_modes=('normal')


ff_optimizer='sgd'
ff_lr=0.01
hydra_lr=0.0001
seeds=(7)
prune_methods=('omp')
gmp_T=1000
gpus=(5 4)
for j in ${!networks[@]};do
    for k in ${!prune_modes[@]};do
        for l in ${!prune_methods[@]};do
            for m in ${!seeds[@]};do
                for i in ${!datasets[@]};do
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_${seeds[m]}_${ff_optimizer}_${ff_lr}_${hydra_lr}.log
                        python ./core/vpns.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --prune_method ${prune_methods[l]} \
                            --gmp_T ${gmp_T} \
                            --prune_mode ${prune_modes[k]} \
                            --density_list ${density_list} \
                            --ff_optimizer ${ff_optimizer} \
                            --ff_lr ${ff_lr} \
                            --hydra_lr ${hydra_lr} \
                            --gpu ${gpus[i]} \
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
