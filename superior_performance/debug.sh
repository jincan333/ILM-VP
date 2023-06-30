#!/bin/sh

experiment_name='debug'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# datasets=['cifar10', 'cifar100', 'flowers102', 'dtd', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'tiny_imagenet', 'imagenet']
# stanfordcars
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
networks=('vgg')
datasets=('tiny_imagenet')
epochs=1
#7 9 17
seed=(7)
density_list='1,0.1'
prune_modes=('vp_ff')
prune_methods=('hydra')

gpus=(3)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for l in ${!prune_methods[@]};do
                for m in ${!seed[@]};do
                log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_${seed[m]}.log
                    python ./core/vpns.py \
                        --experiment_name ${experiment_name} \
                        --dataset ${datasets[i]} \
                        --network ${networks[j]} \
                        --prune_method ${prune_methods[l]} \
                        --prune_mode ${prune_modes[k]} \
                        --density_list ${density_list} \
                        --gpu ${gpus[m]} \
                        --epochs ${epochs} \
                        --seed ${seed[m]} \
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
