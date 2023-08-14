#!/bin/sh

experiment_name='ablation_vpns_hydra'
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
epochs=1
# seed 7 9 17
# prune_modes=['score+vp_weight', 'weight+vp_score', 'score+vp_weight+vp','score_weight']

density_list='1,0.80,0.50,0.10'

score_vp_ratios=(5)
weight_optimizer='sgd'
weight_lr=0.05
vp_optimizer='adam'
vp_lr=0.001
score_optimizer='adam'
score_lr=0.0001
seeds=(7)
# gmp_T=1000

prune_modes=('score+vp_weight')
prune_methods=('vpns')
global_vp_data=1
gpus=(0)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for l in ${!prune_methods[@]};do
                for m in ${!seeds[@]};do
                    for n in ${!score_vp_ratios[@]};do
                        log_filename=${foler_name}/${global_vp_data}_${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_${seeds[m]}_ratio${score_vp_ratios[n]}_${weight_optimizer}_${weight_lr}_${vp_optimizer}_${vp_lr}_${score_optimizer}_${score_lr}.log
                            python ./vpns/vpns_unstructured_two_phase.py \
                                --experiment_name ${experiment_name} \
                                --dataset ${datasets[i]} \
                                --network ${networks[j]} \
                                --global_vp_data ${global_vp_data} \
                                --prune_method ${prune_methods[0]} \
                                --prune_mode ${prune_modes[k]} \
                                --density_list ${density_list} \
                                --weight_optimizer ${weight_optimizer} \
                                --weight_lr ${weight_lr} \
                                --vp_optimizer ${vp_optimizer} \
                                --vp_lr ${vp_lr} \
                                --score_optimizer ${score_optimizer} \
                                --score_lr ${score_lr} \
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
