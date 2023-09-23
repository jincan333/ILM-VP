#!/bin/sh

experiment_name='vpns_vgg_transfer_search'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# datasets=("cifar100" "dtd" "flowers102" "ucf101" "food101" "gtsrb" "svhn" "eurosat" "oxfordpets" "stanfordcars" "sun397", "tiny_imagenet")
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# datasets=('cifar100' 'flowers102' 'dtd' 'food101' 'oxfordpets')
networks=('vgg')
datasets=('tiny_imagenet')
epochs=10
# seed 7 9 17
# prune_modes=['score+vp_weight', 'weight+vp_score', 'score+vp_weight+vp','score_weight']

# density_list='1,0.60'
density_list='1,0.20,0.10'

weight_optimizer='sgd'
weight_lr=0.01
weight_vp_optimizer=${weight_optimizer}

score_optimizer='sgd'
# score_lr=0.001
score_vp_optimizer=${score_optimizer}
score_vp_lr=${score_lr}
prune_modes=('score+vp_weight+vp')
prune_methods=('vpns')
global_vp_data=0
batch_size=256
# gmp_T=1000

seeds=(7 7 7)
gpus=(1 2 3)
score_lrs=(0.01 0.005 0.001)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for l in ${!prune_methods[@]};do
                for m in ${!gpus[@]};do  
                    score_lr=${score_lrs[m]}
                    weight_vp_lr=${weight_lr}
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_${seeds[m]}_${weight_optimizer}_${weight_lr}_${weight_vp_optimizer}_${weight_vp_lr}_${score_optimizer}_${score_lr}_${score_vp_optimizer}_${score_vp_lr}_${epochs}_${gpus[m]}.log
                    nohup python -u ./vpns/vpns_unstructured_explore.py \
                        --experiment_name ${experiment_name} \
                        --dataset ${datasets[i]} \
                        --network ${networks[j]} \
                        --prune_method ${prune_methods[l]} \
                        --prune_mode ${prune_modes[k]} \
                        --density_list ${density_list} \
                        --weight_optimizer ${weight_optimizer} \
                        --weight_lr ${weight_lr} \
                        --weight_vp_optimizer ${weight_vp_optimizer} \
                        --weight_vp_lr ${weight_vp_lr} \
                        --score_optimizer ${score_optimizer} \
                        --score_lr ${score_lr} \
                        --score_vp_optimizer ${score_vp_optimizer} \
                        --score_vp_lr ${score_vp_lr} \
                        --gpu ${gpus[m]} \
                        --epochs ${epochs} \
                        --batch_size ${batch_size} \
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
