#!/bin/sh

experiment_name='compressed_then_vp_more_more'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# datasets=("cifar100" "dtd" "flowers102" "ucf101" "food101" "gtsrb" "svhn" "eurosat" "oxfordpets" "stanfordcars" "sun397", "tiny_imagenet")
# datasets=("ucf101" "eurosat" "oxfordpets" "stanfordcars" "sun397") 
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'gmp']
# datasets=('cifar100' 'flowers102' 'dtd' 'food101' 'oxfordpets')
networks=('resnet18')
datasets=('cifar10')
epochs=120
# seed 7 9 17
density_list='1,0.20'
prune_modes=('weight+vp')

weight_optimizer='sgd'
weight_lr=0.01
seeds=(7)
prune_methods=('random' 'snip')
# prune_methods=('omp' 'random' 'snip' 'synflow')
gpus=(2 1)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for m in ${!seeds[@]};do
                for l in ${!prune_methods[@]};do
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_${seeds[m]}_${weight_optimizer}_${weight_lr}.log
                        nohup python -u ./vpns/normal_then_vp.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --prune_method ${prune_methods[l]} \
                            --prune_mode ${prune_modes[k]} \
                            --density_list ${density_list} \
                            --weight_optimizer ${weight_optimizer} \
                            --weight_lr ${weight_lr} \
                            --gpu ${gpus[l]} \
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
