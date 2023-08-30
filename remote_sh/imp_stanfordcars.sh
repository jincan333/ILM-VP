#!/bin/sh

experiment_name='imp_stanfordcars'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# datasets=("cifar100" "dtd" "flowers102" "ucf101" "food101" "gtsrb" "svhn" "eurosat" "oxfordpets" "stanfordcars" "sun397", "tiny_imagenet")
# datasets=("ucf101" "eurosat" "oxfordpets" "stanfordcars" "sun397") 
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'gmp']
# datasets=('cifar100' 'flowers102' 'dtd' 'food101' 'oxfordpets')
networks=('resnet18')
datasets=('stanfordcars')
epochs=120
# seed 7 9 17
# density_list='1,0.8000,0.6400,0.5120,0.4100,0.3280,0.2620,0.2097,0.1678,0.1342,0.1074,0.0859,0.0687,0.0550'
density_list='1,0.8000,0.6400,0.5120,0.4100,0.3280,0.2620,0.2097,0.1678,0.1342,0.1074,0.0859,0.0687,0.0550,0.0440,0.0350,0.0225,0.0180,0.0144,0.0115'
# 'weight', 'weight+vp'
prune_modes=('weight')

weight_optimizer='adam'
weight_lr=0.001
seeds=(7 9 17)
prune_methods=('imp')
gpus=(7 6 5)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for l in ${!prune_methods[@]};do
                for m in ${!seeds[@]};do
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_${seeds[m]}_${weight_optimizer}_${weight_lr}.log
                        nohup python ./vpns/normal.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --prune_method ${prune_methods[l]} \
                            --prune_mode ${prune_modes[k]} \
                            --density_list ${density_list} \
                            --weight_optimizer ${weight_optimizer} \
                            --weight_lr ${weight_lr} \
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
