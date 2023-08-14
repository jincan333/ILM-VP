#!/bin/sh

experiment_name='ablation_method'
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
datasets=('cifar100')
epochs=120
# seed 7 9 17
prune_modes=('vp_ff')
prune_methods=('hydra')
density_list='1,0.1'
second_phases=('vp+ff_cotrain')


ff_optimizer='sgd'
ff_lr=0.05
vp_optimizer='adam'
vp_lr=0.001
hydra_optimizer='adam'
hydra_lr=0.0001
seeds=(7)
prune_methods=('hydra')
# gmp_T=1000

score_vp_ratios=(5)
gpus=(4)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for l in ${!prune_methods[@]};do
                for m in ${!seeds[@]};do
                    for n in ${!score_vp_ratios[@]};do
                        log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_${seeds[m]}_ratio${score_vp_ratios[n]}_${ff_optimizer}_${ff_lr}_${vp_optimizer}_${vp_lr}_${hydra_optimizer}_${hydra_lr}.log
                            python ./core/vpns_unstructured_bil.py \
                                --experiment_name ${experiment_name} \
                                --dataset ${datasets[i]} \
                                --network ${networks[j]} \
                                --second_phase ${second_phases[0]} \
                                --prune_mode ${prune_modes[0]} \
                                --prune_method ${prune_methods[0]} \
                                --prune_mode ${prune_modes[k]} \
                                --density_list ${density_list} \
                                --ff_optimizer ${ff_optimizer} \
                                --ff_lr ${ff_lr} \
                                --vp_optimizer ${vp_optimizer} \
                                --vp_lr ${vp_lr} \
                                --hydra_optimizer ${hydra_optimizer} \
                                --hydra_lr ${hydra_lr} \
                                --score_vp_ratio ${score_vp_ratios[n]} \
                                --gpu ${gpus[n]} \
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
    wait
done
