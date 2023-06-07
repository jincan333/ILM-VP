#!/bin/sh


# pad size experiment pad_sizes=(16 32 48 64 80 96 112) input_size=224 gpus=(0 0 0 0 0 0 0)
# exp_name='padsize_exp'
# pad_sizes=(16 32 48 64 80 96 112)
# gpus=(0 0 0 0 0 0 0)
# for i in ${!pad_sizes[@]}; 
# do
#     log_filename='${exp_name}_224_${pad_sizes[i]}.log'
#     python ./experiments/prompt_sparsity/lm_vp_prune.py --experiment_name $exp_name --gpu ${gpus[i]} --input_size 224 --pad_size ${pad_sizes[i]} --epochs 200 > $log_filename 2>&1
# done


# input size experiment input_sizes=(224 192 160 128 96 64 32) pad_size=(16 32 48 64 80 96 112) gpus=(7 6 5 4 3 2 1)
# input_sizes=(192 160 128 96 64 32)
# gpus=(7 6 5 4 3 2)
# pad_sizes=(64 80 96 112)
# for pad_size in ${pad_sizes[@]};do
#     for i in ${!input_sizes[@]};do
#         log_filename=inputsize_${input_sizes[i]}_${pad_size}.log
#         python ./experiments/prompt_sparsity/lm_vp_prune.py --experiment_name inputsize_exp --gpu ${gpus[i]} --input_size ${input_sizes[i]} --pad_size ${pad_size} --epochs 200 > $log_filename 2>&1 &
#     done
#     wait
# done


# prompt location experiment
# experiment_name='prompt_loc_exp'
# prune_methods=('imp')
# label_mapping_modes=('flm')
# gpus=(7 5 1)
# prompt_methods=('pad' 'fix' 'random')
# input_sizes=(224)
# pad_sizes=(32)
# mask_sizes=(156)
# pruning_times=10
# epochs=200
# for i in ${!prompt_methods[@]};do
#     log_filename=${experiment_name}_${prompt_methods[i]}.log
#     python ./experiments/prompt_sparsity/lm_vp_prune.py --experiment_name ${experiment_name} --prune_method imp --label_mapping_mode flm --gpu ${gpus[i]} --prompt_method ${prompt_methods[i]} --input_size 224 --pad_size 32 --mask_size 156 --pruning_times ${pruning_times} --epochs ${epochs} > $log_filename 2>&1 &
# done


# visual prompt prune experiment
# experiment_name='prompt_prune_exp'
# prune_methods=('imp' 'omp' 'grasp' 'hydra')
# label_mapping_modes=('flm' 'ilm')
# gpus=(7 6 5 4 3 2 1 0)
# prompt_method='pad'
# input_size=160
# pad_size=32
# pruning_times=10
# epochs=200
# for i in ${!prune_methods[@]};do
#     for j in ${!label_mapping_modes[@]};do
#         log_filename=${experiment_name}_${prune_methods[i]}_${label_mapping_modes[j]}.log
#         python ./experiments/prompt_sparsity/lm_vp_prune.py --experiment_name ${experiment_name} --prune_method ${prune_methods[i]} --label_mapping_mode ${label_mapping_modes[j]} --gpu ${gpus[2*i+j]} --prompt_method ${prompt_method} --input_size ${input_size} --pad_size ${pad_size} --pruning_times ${pruning_times} --epochs ${epochs} > $log_filename 2>&1 &
#     done
# done


# padsize=0 and without tune performance
# experiment_name='pad_0_exp'
# prune_methods=('imp' 'omp' 'grasp')
# label_mapping_modes=('flm' 'ilm')
# gpus=(7 6 5 4 3 2 1 0)
# prompt_method='pad'
# input_size=224
# pad_size=0
# pruning_times=1
# epochs=1
# for i in ${!prune_methods[@]};do
#     for j in ${!label_mapping_modes[@]};do
#         log_filename=${experiment_name}_${prune_methods[i]}_${label_mapping_modes[j]}.log
#         python ./experiments/prompt_sparsity/lm_vp_prune.py --experiment_name ${experiment_name} --prune_method ${prune_methods[i]} --label_mapping_mode ${label_mapping_modes[j]} --gpu ${gpus[2*i+j]} --prompt_method ${prompt_method} --input_size ${input_size} --pad_size ${pad_size} --pruning_times ${pruning_times} --epochs ${epochs} > $log_filename 2>&1 &
#     done
# done


###  tricks experiments  ###
# experiment_name='tricks_exp_eval_use_train'
# prune_methods=('imp')
# label_mapping_modes=('flm')
# gpus=(1 6 5 4 3 2 1 0)
# prompt_method='expand'
# input_size=32
# pruning_times=1
# epochs=200
# for i in ${!prune_methods[@]};do
#     for j in ${!label_mapping_modes[@]};do
#         log_filename=${experiment_name}_${prune_methods[i]}_${label_mapping_modes[j]}.log
#         python ./experiments/prompt_sparsity/lm_vp_prune.py --experiment_name ${experiment_name} --prune_method ${prune_methods[i]} --label_mapping_mode ${label_mapping_modes[j]} --gpu ${gpus[2*i+j]} --prompt_method ${prompt_method} --input_size ${input_size} --pruning_times ${pruning_times} --epochs ${epochs} > $log_filename 2>&1 &
#     done
# done


# experiment_name='tricks_exp_flm_after_prune'
# prune_methods=('omp' 'grasp')
# label_mapping_modes=('flm')
# gpus=(3 0)
# prompt_method='pad'
# input_size=160
# pad_size=32
# pruning_times=10
# epochs=200
# for i in ${!prune_methods[@]};do
#     for j in ${!label_mapping_modes[@]};do
#         log_filename=${experiment_name}_${prune_methods[i]}.log
#         python ./experiments/prompt_sparsity/lm_vp_prune.py --experiment_name ${experiment_name} --prune_method ${prune_methods[i]} --label_mapping_mode ${label_mapping_modes[j]} --gpu ${gpus[i]} --prompt_method ${prompt_method} --input_size ${input_size} --pad_size ${pad_size} --pruning_times ${pruning_times} --epochs ${epochs} > $log_filename 2>&1 &
#     done
# done


# experiment_name='tricks_exp_flm_after_prune_notune'
# prune_methods=('imp' 'omp' 'grasp')
# label_mapping_modes=('flm')
# gpus=(7 5 1)
# prompt_methods=('pad')
# input_sizes=(224)
# pad_sizes=(0)
# mask_sizes=(156)
# pruning_times=10
# epochs=1
# for i in ${!prune_methods[@]};do
#     log_filename=${experiment_name}_${prune_methods[i]}.log
#     python ./experiments/prompt_sparsity/lm_vp_prune.py --experiment_name ${experiment_name} --prune_method ${prune_methods[i]} --label_mapping_mode flm --gpu ${gpus[i]} --prompt_method ${prompt_methods[0]} --input_size ${input_sizes[0]} --pad_size ${pad_sizes[0]} --mask_size ${mask_sizes[0]} --pruning_times ${pruning_times} --epochs ${epochs} > $log_filename 2>&1 &
# done



# parser.add_argument('--experiment_name', default='inputsize_exp', type=str, help='name of experiment, the save directory will be save_dir+exp_name')
# parser.add_argument('--gpu', type=int, default=4, help='gpu device id')
# parser.add_argument('--label_mapping_mode', type=str, default='flm', help='label mapping methods: rlm, flm, ilm, None')
# parser.add_argument('--prompt_method', type=str, default='pad', help='None, expand, pad, fix, random')
# parser.add_argument('--input_size', type=int, default=192, help='image size before prompt, no more than 224', choices=[224, 192, 160, 128, 96, 64, 32])
# parser.add_argument('--pad_size', type=int, default=112, help='only for padprompt, no more than 112, parameters cnt 4*pad**2+896pad', choices=[16, 32, 48, 64, 80, 96, 112])
# parser.add_argument('--mask_size', type=int, default=96, help='only for fixadd and randomadd, no more than 224, parameters cnt mask**2', choices=[115, 156, 183, 202, 214, 221, 224])
# parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
# parser.add_argument('--lr_scheduler', default='multistep', help='decreasing strategy. Default: cosine, multistep')
# parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
# parser.add_argument('--prune_method', type=str, default='imp', help='prune methods: imp, omp, grasp, hydra')
# parser.add_argument('--pruning_times', default=10, type=int, help='overall times of pruning')
# parser.add_argument('--density', type=float, default=0.80, help='The density of the overall sparse network.')