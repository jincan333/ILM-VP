#!/bin/sh
experiment_name='vpns_dataset_transfer'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# datasets=("cifar100" "dtd" "flowers102" "ucf101" "food101" "gtsrb" "svhn" "eurosat" "oxfordpets" "stanfordcars" "sun397")
# datasets=("ucf101" "eurosat" "oxfordpets" "stanfordcars" "sun397") 
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# dataset=('cifar10' 'cifar100' 'svhn' 'mnist' 'flowers102' 'ucf101')
networks=('resnet18')
datasets=('imagenet')
epochs=120
# seed 7 9 17
seed=(7)
density_list='1,0.1,0.01,0.001'
imagenet_path='/data/imagenet'

prune_modes=('normal')
prune_methods=('hydra')
gpus=(0)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for l in ${!prune_methods[@]};do
                for m in ${!seed[@]};do
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_${seed[m]}.log
                        python ./core/dataset_transfer.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --prune_method ${prune_methods[l]} \
                            --prune_mode ${prune_modes[k]} \
                            --density_list ${density_list} \
                            --gpu ${gpus[m]} \
                            --epochs ${epochs} \
                            --imagenet_path ${imagenet_path} \
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
