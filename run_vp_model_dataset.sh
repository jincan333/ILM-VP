#!/bin/sh

# parser.add_argument('--label_mapping_mode', type=str, default='ilm', help='label mapping methods: rlm, flm, ilm, None', choices=['flm', 'ilm', 'rlm', None, 'None'], required=True)
# parser.add_argument('--prompt_method', type=str, default='pad', help='None, expand, pad, fix, random', choices=['pad', 'fix', 'random', None, 'None'], required=True)
# parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use. Default: sgd. Options: sgd, adam.', choices=['sgd', 'adam'], required=True)
# parser.add_argument('--lr_scheduler', default='multistep', help='decreasing strategy. Default: cosine, multistep', choices=['cosine', 'multistep'], required=True)
# parser.add_argument('--prune_method', type=str, default='hydra', help='prune methods: imp, omp, grasp, hydra', choices=['imp', 'omp', 'grasp', 'hydra'], required=True)
# parser.add_argument('--is_finetune', default=0, type=int, choices=[0, 1], required=True)
# parser.add_argument('--network', default='resnet18', choices=["resnet18", "resnet50", "instagram"], required=True)
# parser.add_argument('--dataset', default="gtsrb", choices=["cifar10", "cifar100", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], required=True)

# parser.add_argument('--experiment_name', default='exp', type=str, help='name of experiment, the save directory will be save_dir+exp_name')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--input_size', type=int, default=128, help='image size before prompt, no more than 224', choices=[224, 192, 160, 128, 96, 64, 32])
# parser.add_argument('--pad_size', type=int, default=48, help='only for padprompt, no more than 112, parameters cnt 4*pad**2+896pad', choices=[0, 16, 32, 48, 64, 80, 96, 112])
# parser.add_argument('--mask_size', type=int, default=156, help='only for fixadd and randomadd, no more than 224, parameters cnt mask**2', choices=[115, 156, 183, 202, 214, 221, 224])
# parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
# parser.add_argument('--pruning_times', default=10, type=int, help='overall times of pruning')
# parser.add_argument('--density', type=float, default=0.80, help='The density of the overall sparse network.')
# parser.add_argument('--seed', default=7, type=int, help='random seed')
# parser.add_argument('--hydra_scaled_init', type=int, default=1, help='whether use scaled initialization for hydra or not.', choices=[0, 1])
# parser.add_argument('--flm_loc', type=str, default='pre', help='pre-train flm or after-prune flm.', choices=['pre', 'after'])
# parser.add_argument('--randomcrop', type=int, default=0, help='dataset randomcrop.', choices=[0, 1])
# parser.add_argument('--is_adjust_linear_head', type=int, default=0, choices=[0, 1])


# test dataset and model 

experiment_name='vp_model_dataset'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi

# ucf101   eurosat   oxfordpets   stanfordcars   sun397
# datasets=("cifar100" "dtd" "flowers102" "ucf101" "food101" "gtsrb" "svhn" "eurosat" "oxfordpets" "stanfordcars" "sun397")
# datasets=("ucf101" "eurosat" "oxfordpets" "stanfordcars" "sun397") 
networks=('resnet18' 'resnet50')
datasets=("dtd" "flowers102" "food101" "gtsrb" "svhn")
is_finetunes=(0)
label_mapping_modes=('ilm')
prune_methods=('imp' 'omp' 'hydra')
prompt_methods=('pad')
optimizers=('adam')
lr_schedulers=('multistep')
gpus=(0 1 2)
input_sizes=(128)
pad_sizes=(48)
pruning_times=10
epochs=200
seed=7
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for m in ${!is_finetunes[@]};do
            for n in ${!prompt_methods[@]};do
                for l in ${!prune_methods[@]};do
                    log_filename=${foler_name}/test_${datasets[i]}_${networks[j]}_${prune_methods[l]}_${is_finetunes[m]}_${prompt_methods[n]}.log
                    python ./experiments/main.py \
                        --experiment_name ${experiment_name} \
                        --dataset ${datasets[i]} \
                        --network ${networks[j]} \
                        --label_mapping_mode ${label_mapping_modes[0]} \
                        --prune_method ${prune_methods[l]} \
                        --is_finetune ${is_finetunes[m]} \
                        --prompt_method ${prompt_methods[n]} \
                        --optimizer ${optimizers[0]} \
                        --lr_scheduler ${lr_schedulers[0]} \
                        --gpu ${gpus[n]} \
                        --input_size ${input_sizes[0]} \
                        --pad_size ${pad_sizes[0]} \
                        --pruning_times ${pruning_times} \
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




# test prompt or prompt+finntune  

# experiment_name='test_prompt'
# foler_name=logs/${experiment_name}
# if [ ! -d ${foler_name} ]; then
#     mkdir -p ${foler_name}
# fi


# datasets=('cifar10')
# networks=('resnet18')
# is_finetunes=(0 1)
# label_mapping_modes=('ilm')
# prune_methods=('imp')
# prompt_methods=('pad' 'None')
# optimizers=('sgd')
# lr_schedulers=('cosine')
# gpus=(0 0 0 0 0 0)
# input_sizes=(128)
# pad_sizes=(48)
# pruning_times=2
# epochs=3
# seed=7
# for i in ${!datasets[@]};do
#     for j in ${!networks[@]};do
#         for m in ${!is_finetunes[@]};do
#             for k in ${!label_mapping_modes[@]};do
#                 for l in ${!prune_methods[@]};do
#                     log_filename=${foler_name}/test_${datasets[i]}_${networks[j]}_${label_mapping_modes[k]}_${prune_methods[l]}_${is_finetunes[m]}.log
#                     python ./experiments/main.py \
#                         --experiment_name ${experiment_name} \
#                         --dataset ${datasets[i]} \
#                         --network ${networks[j]} \
#                         --label_mapping_mode ${label_mapping_modes[k]} \
#                         --prune_method ${prune_methods[l]} \
#                         --is_finetune ${is_finetunes[m]} \
#                         --prompt_method ${prompt_methods[0]} \
#                         --optimizer ${optimizers[0]} \
#                         --lr_scheduler ${lr_schedulers[0]} \
#                         --gpu ${gpus[l]} \
#                         --input_size ${input_sizes[0]} \
#                         --pad_size ${pad_sizes[0]} \
#                         --pruning_times ${pruning_times} \
#                         --epochs ${epochs} \
#                         --seed ${seed} \
#                         > $log_filename 2>&1 &
#                 done
#                 wait
#             done
#             wait
#         done
#         wait
#     done
#     wait
# done




# test no prompt finntune or not finetuen

# experiment_name='test_no_prompt'
# foler_name=logs/${experiment_name}
# if [ ! -d ${foler_name} ]; then
#     mkdir -p ${foler_name}
# fi


# datasets=('cifar10')
# networks=('resnet18')
# is_finetunes=(0 1)
# label_mapping_modes=('flm' 'ilm')
# prune_methods=('hydra')

# prompt_methods=(None)
# optimizers=('sgd')
# lr_schedulers=('cosine')
# gpus=(5 6 7 0 1 2)
# input_sizes=(128)
# pad_sizes=(48)
# pruning_times=2
# epochs=2
# seed=7
# lr=0.01
# for i in ${!datasets[@]};do
#     for j in ${!networks[@]};do
#         for m in ${!is_finetunes[@]};do
#             for k in ${!label_mapping_modes[@]};do
#                 for l in ${!prune_methods[@]};do
#                     log_filename=${foler_name}/test_${datasets[i]}_${networks[j]}_${label_mapping_modes[k]}_${prune_methods[l]}_${is_finetunes[m]}.log
#                     python ./experiments/main.py \
#                         --experiment_name ${experiment_name} \
#                         --dataset ${datasets[i]} \
#                         --network ${networks[j]} \
#                         --label_mapping_mode ${label_mapping_modes[k]} \
#                         --prune_method ${prune_methods[l]} \
#                         --is_finetune ${is_finetunes[m]} \
#                         --prompt_method ${prompt_methods[0]} \
#                         --optimizer ${optimizers[0]} \
#                         --lr_scheduler ${lr_schedulers[0]} \
#                         --gpu ${gpus[l]} \
#                         --input_size ${input_sizes[0]} \
#                         --pad_size ${pad_sizes[0]} \
#                         --pruning_times ${pruning_times} \
#                         --epochs ${epochs} \
#                         --seed ${seed} \
#                         --lr ${lr} \
#                         > $log_filename 2>&1 &
#                 done
#                 wait
#             done
#             wait
#         done
#         wait
#     done
#     wait
# done