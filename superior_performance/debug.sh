#!/bin/sh

experiment_name='debug'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# datasets=['cifar10', 'cifar100', 'flowers102', 'dtd', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'tiny_imagenet', 'imagenet']
# stanfordcars
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
networks=('resnet18')
datasets=('dtd')
epochs=20
#7 9 17
seed=(7)
density_list='1,0.1,0.01,0.001'
prune_modes=('normal')
prune_methods=('gmp')
gmp_T=50

ff_optimizer='sgd'
ff_lr=0.01
gpus=(3)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for l in ${!prune_methods[@]};do
                for m in ${!seed[@]};do
                log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_${seed[m]}_${ff_optimizer}_${ff_lr}.log
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
