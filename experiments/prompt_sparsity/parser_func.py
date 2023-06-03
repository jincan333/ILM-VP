import pickle


def add_args(parser):
    # Frequently change
    parser.add_argument('--experiment_name', default='inputsize_exp', type=str, help='name of experiment, the save directory will be save_dir+exp_name')
    parser.add_argument('--gpu', type=int, default=4, help='gpu device id')
    parser.add_argument('--label_mapping_mode', type=str, default='flm', help='label mapping methods: rlm, flm, ilm, None')
    parser.add_argument('--prompt_method', type=str, default='pad', help='None, expand, pad, fix, random')
    parser.add_argument('--input_size', type=int, default=192, help='image size before prompt, no more than 224', choices=[224, 192, 160, 128, 96, 64, 32])
    parser.add_argument('--pad_size', type=int, default=112, help='only for padprompt, no more than 112, parameters cnt 4*pad**2+896pad', choices=[16, 32, 48, 64, 80, 96, 112])
    parser.add_argument('--mask_size', type=int, default=96, help='only for fixadd and randomadd, no more than 224, parameters cnt mask**2', choices=[115, 156, 183, 202, 214, 221, 224])
    parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--lr_scheduler', default='multistep', help='decreasing strategy. Default: cosine, multistep')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--prune_method', type=str, default='imp', help='prune methods: imp, omp, grasp, hydra')
    parser.add_argument('--pruning_times', default=10, type=int, help='overall times of pruning')
    parser.add_argument('--density', type=float, default=0.80, help='The density of the overall sparse network.')

    ##################################### General setting ############################################
    parser.add_argument('--save_dir', help='The directory used to save the trained models', default='results', type=str)
    # parser.add_argument('--experiment_name', default='test', type=str, help='name of experiment, the save directory will be save_dir+exp_name')
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    # parser.add_argument('--gpu', type=int, default=6, help='gpu device id')
    parser.add_argument('--workers', type=int, default=2, help='number of workers in dataloader')
    parser.add_argument('--resume_checkpoint', default='', help="resume checkpoint path")
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency')

    ##################################### Dataset #################################################
    parser.add_argument('--data', type=str, default='dataset', help='location of the data corpus')
    parser.add_argument('--dataset', default="cifar10")

    ##################################### Architecture ############################################
    parser.add_argument('--network', default='resnet18')
    # parser.add_argument('--label_mapping_mode', type=str, default='flm', help='label mapping methods: rlm, flm, ilm, None')
    parser.add_argument('--label_mapping_interval', type=int, default=1, help='in ilm, the interval of epoch to implement label mapping')
    parser.add_argument('--is_adjust_linear_head', type=bool, default=False, help='whether adjust the linear head or not')

    ##################################### Visual Prompt ############################################
    # parser.add_argument('--prompt_method', type=str, default='pad', help='None, expand, pad, fix, random')
    # parser.add_argument('--input_size', type=int, default=32, help='image size before prompt, only work for exapnd')
    parser.add_argument('--output_size', type=int, default=224, help='image size after prompt, fix to 224')
    # parser.add_argument('--pad_size', type=int, default=96, help='pad size of padprompt, no more than 112, parameters cnt 4*pad**2+896pad')
    # parser.add_argument('--mask_size', type=int, default=96, help='mask size of fixadd and randomadd, no more than 224, parameters cnt mask**2')
    
    ##################################### Training setting #################################################
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    # parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    # parser.add_argument('--lr_scheduler', default='multistep', help='decreasing strategy. Default: cosine, multistep')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    # parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
    parser.add_argument('--decreasing_step', default=[0.5,0.72], type = list, help='decreasing strategy')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N', help='extend training time by multiplier times')

    ##################################### Pruning setting #################################################
    # parser.add_argument('--prune_method', type=str, default='hydra', help='prune methods: imp, omp, grasp, hydra')
    # parser.add_argument('--pruning_times', default=10, type=int, help='overall times of pruning')
    # parser.add_argument('--density', type=float, default=0.80, help='The density of the overall sparse network.')
    parser.add_argument('--fix', default='true', action='store_true', help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death_rate', type=float, default=0.50, help='The pruning rate / death rate used for dynamic sparse training (not used in this paper).')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay_schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--scaled', action='store_true', help='scale the initialization by 1/density')
    # IMP
    parser.add_argument('--imp_prune_type', default='pt', type=str, help='IMP type (lt, pt or rewind_lt)')
    parser.add_argument('--imp_random_prune', action='store_true', help='whether using random prune')
    parser.add_argument('--imp_rewind_epoch', default=3, type=int, help='rewind checkpoint')
    # Hydra
    parser.add_argument('--layer_type',choices=['subnet','dense'],default='subnet')
    parser.add_argument('--ChannelPrune', choices=['kernel','channel','weight','inputchannel'],default='weight')
    parser.add_argument('--init_type',default='kaiming_normal')
    parser.add_argument('--scaled_score_init',default=True)
    parser.add_argument('--k',default=0.80)
    parser.add_argument('--exp_mode',default='prune')
    parser.add_argument('--freeze_bn',default=False)
    parser.add_argument('--scores_init_type',default=None, choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"))



def save_args(args, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(args, file)


def load_args(file_path):
    with open(file_path, 'rb') as file:
        load_args = pickle.load(file)
    return load_args