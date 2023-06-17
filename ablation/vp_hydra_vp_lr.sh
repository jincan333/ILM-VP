#!/bin/sh

experiment_name='ablation_vp_hydra_lr'
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
prune_modes=('vp_ff')
prune_methods=('hydra')

vp_lrs=(0.001 0.01)
gpus=(3 2)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for l in ${!prune_methods[@]};do
                for m in ${!vp_lrs[@]};do
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_lr_${vp_lrs[m]}.log
                        python ./core/vpns.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --prune_method ${prune_methods[l]} \
                            --prune_mode ${prune_modes[k]} \
                            --vp_lr ${vp_lrs[m]} \
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
