#!/bin/sh

experiment_name='ablation_hydra_lm'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# datasets=("cifar100" "dtd" "flowers102" "ucf101" "food101" "gtsrb" "svhn" "eurosat" "oxfordpets" "stanfordcars" "sun397")
# datasets=("ucf101" "eurosat" "oxfordpets" "stanfordcars" "sun397") 
networks=('resnet18')
datasets=('cifar10')
epochs=50
seed=7
prune_modes=('normal')
prune_methods=('hydra')
hydra_lrs=(0.0001)

label_mapping_modes=('flm')
gpus=(7)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for l in ${!prune_methods[@]};do
                for m in ${!label_mapping_modes[@]};do
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_${label_mapping_modes[m]}_${hydra_lrs[0]}.log
                        python ./core/vpns.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --prune_method ${prune_methods[l]} \
                            --prune_mode ${prune_modes[k]} \
                            --hydra_lr ${hydra_lrs[0]} \
                            --label_mapping_mode ${label_mapping_modes[m]} \
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
