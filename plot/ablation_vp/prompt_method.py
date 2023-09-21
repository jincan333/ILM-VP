import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker 

def winning_ticket_gap(unpruned_acc, avg_acc_list, num_sparsities):
    x = np.arange(num_sparsities)
    x_density = 1-  0.8 ** x
    print("winning ticket acc gap:",np.max(avg_acc_list) - unpruned_acc)
    print("Achived at the sparsity of {}".format(x_density[np.argmax(avg_acc_list)]))
    return 0 

def last_winning_ticket_sparsity(unpruned_acc, avg_acc_list, num_sparsities, err_list):
    x = np.arange(num_sparsities)
    x_density = 1-  0.8 ** x
    print("last winning ticket sparsity:", x_density[avg_acc_list - unpruned_acc >= 0])
    print("Accuracy:", avg_acc_list[avg_acc_list - unpruned_acc >= 0] )
    print("Std:", err_list[avg_acc_list - unpruned_acc >= 0] )
    return 0

def extract_y_err(y_dense, performance_str):
    if '(' in performance_str or ')' in performance_str:
        y = re.findall('(\d+\.\d+)(?=\()', performance_str)
        err = re.findall('(?<=\()(\d+\.\d+)', performance_str)
        y = [float(num) for num in y]
        err = [float(num) for num in err]       
    else:
        y = [float(i) for i in performance_str.split()]
        err = [0] * len(y)

    y.insert(0, y_dense)
    err.insert(0, 0)
    return np.array(y), np.array(err)

if __name__ == "__main__":
    # num = 14
    # # x_grid = np.array(range(num))
    # step = 1
    # index = np.arange(0, num, step)
    # x = np.arange(num)[index]
    # x_density = 100 - 100 * (0.8 ** x)
    # x_grid = x_density
    # x_density_list = ['{:.2f}'.format(value) for value in x_density]

    # num = 14
    # x_grid = np.array(range(num))
    # step = 1
    # index = np.arange(0, num, step)
    # x = np.arange(num)[index]
    # x_density = 100 - 100 * (0.8 ** x)
    # x_grid = x_density
    # x_density_list = ['{:.2f}'.format(value) for value in x_density]

    # y_PFTT_time = np.insert(np.array([180 for i in range(num - 1)]), 0, 0)
    # y_BiP_time = np.insert(np.array([225 for i in range(num - 1)]), 0, 0)
    # y_hydra_time = np.insert(np.array([136 for i in range(num - 1)]), 0, 0)
    # y_IMP_time = np.array([115.2 * i for i in range(num)])
    # y_OMP_time = np.insert(np.array([115.2 for i in range(num - 1)]), 0, 0)
    # y_Grasp_time = np.insert(np.array([120 for i in range(num - 1)]), 0, 0)
    
    # 7, 11; 9, 20
    title = 'Prompt Method'
    num, imp_num = 7, 11
    y_dense = 82.10
    y_min, y_max = 80.4,83.6
    
    # 10, 20
    x_sparsity_list = np.array([0, 40, 50, 60, 70, 80, 90, 95, 99][:num])
    x_grid = x_sparsity_list
    x_LTH_sparsity_list = np.array([0, 20.00, 36.00, 48.80, 59.00, 67.20, 73.80, 79.03, 83.22, 86.58, 89.26, 91.41, 93.13, 94.50, 95.60, 96.50, 97.75, 98.20, 98.56, 98.85][:imp_num])


    pad    = '83.47 	83.29 	83.29 	82.98 	82.17 	80.87 '
    fix    = '82.98 	83.08 	83.00 	82.93 	81.93 	80.55 '
    random = '82.93 	83.13 	82.98 	82.87 	82.02 	81.07 '
    y_pad, y_pad_err = extract_y_err(y_dense, pad)
    y_fix, y_fix_err = extract_y_err(y_dense, fix)
    y_random, y_random_err = extract_y_err(y_dense, random)


    y_best = np.max(y_pad) 

    x_label = "Sparsity (%)"
    y_label = "Test Accuracy (%)"

    # Canvas setting
    width = 14
    height = 12
    plt.figure(figsize=(width, height))

    sns.set_theme()
    plt.grid(visible=True, which='major', linestyle='-', linewidth=4)
    plt.grid(visible=True, which='minor')
    plt.minorticks_on()
    plt.rcParams['font.serif'] = 'Times New Roman'

    markersize = 20
    linewidth = 4
    markevery = 1
    fontsize = 50
    alpha = 0.7

    # Color Palette
    best_color = 'green'
    best_alpha = 1.0
    dense_color = 'black'
    dense_alpha = 1.0
    SynFlow_color = 'red'
    SynFlow_alpha = 0.9
    SynFlow_prompt_color = 'red'
    SynFlow_prompt_alpha = 0.9
    VPNs_color = 'green'
    VPNs_alpha = 0.9

    pad_color = 'green'
    pad_alpha = 0.9

    fix_color = '#343a40'
    fix_alpha = alpha

    random_color = '#168aad'
    random_alpha = alpha


    BiP_color = 'hotpink'
    BiP_alpha = alpha
    HYDRA_color = 'blue'
    HYDRA_alpha = alpha - 0.1

    HYDRA_prompt_color = 'blue'
    HYDRA_prompt_alpha = alpha - 0.1
    LTH_color = '#dda15e'
    LTH_alpha = 0.9
    LTH_VP_color = '#bc6c25'
    LTH_VP_alpha = 0.9
    OMP_color = 'darkolivegreen'
    OMP_alpha = alpha
    OMP_prompt_color = 'darkolivegreen'
    OMP_prompt_alpha = alpha
    GraSP_color = 'purple'
    GraSP_alpha = alpha
    SNIP_color = 'darkorange'
    SNIP_alpha = alpha
    SNIP_prompt_color = 'darkorange'
    SNIP_prompt_alpha = alpha
    Random_color = 'violet'
    Random_alpha = alpha
    Random_prompt_color = 'violet'
    Random_prompt_alpha = alpha
    Random_VP_color = 'violet'
    Random_VP_alpha = alpha


    fill_in_alpha = 0.2

    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    # plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    # # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    # plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    l_dense = plt.axhline(y=y_dense, color=dense_color, linestyle='--', linewidth=3, label="Dense")


    # l_OMP = plt.plot(x_grid, y_OMP, color=OMP_color, marker='v', markevery=markevery, linestyle='-', linewidth=linewidth,
    #                 markersize=markersize+1, label="OMP", alpha=OMP_alpha)
    # plt.fill_between(x_grid, y_OMP - y_OMP_err, y_OMP + y_OMP_err, color=OMP_color, alpha=fill_in_alpha)

    # l_LTH = plt.plot(x_LTH_sparsity_list, y_LTH, color=LTH_color, marker='*', markevery=markevery, linestyle='-', linewidth=linewidth,
    #                 markersize=markersize+7, label="LTH", alpha=LTH_alpha)
    # plt.fill_between(x_LTH_sparsity_list, y_LTH - y_LTH_err, y_LTH + y_LTH_err, color=LTH_color, alpha=fill_in_alpha)

    # l_Random = plt.plot(x_grid, y_Random, color=Random_color, marker='s', markevery=markevery, linestyle='-',
    #         linewidth=linewidth,
    #         markersize=markersize-5, label="Random", alpha=Random_alpha)
    # plt.fill_between(x_grid, y_Random - y_Random_err, y_Random + y_Random_err, color=Random_color, alpha=fill_in_alpha)


    # l_Random_VP = plt.plot(x_grid, y_Random_VP, color=Random_VP_color, marker='*', markevery=markevery, linestyle='-',
    #         linewidth=linewidth,
    #         markersize=markersize+7, label="VPNs w. Random", alpha=Random_VP_alpha)
    # plt.fill_between(x_grid, y_Random_VP - y_Random_VP_err, y_Random_VP + y_Random_VP_err, color=Random_VP_color, alpha=fill_in_alpha)

    # l_BiP = plt.plot(x_grid, y_BiP, color=BiP_color, marker='v', markevery=markevery, linestyle='-', linewidth=linewidth,
    #                 markersize=markersize+1, label="BiP", alpha=BiP_alpha)
    # plt.fill_between(x_grid, y_BiP - y_BiP_err, y_BiP + y_BiP_err, color=BiP_color, alpha=fill_in_alpha)


    # l_SNIP = plt.plot(x_grid, y_SNIP, color=SNIP_color, marker='o', markevery=markevery, linestyle='-', linewidth=linewidth,
    #                 markersize=markersize, label="SNIP", alpha=SNIP_alpha)
    # plt.fill_between(x_grid, y_SNIP - y_SNIP_err, y_SNIP + y_SNIP_err, color=SNIP_color, alpha=fill_in_alpha)

    # l_GraSP = plt.plot(x_grid, y_GraSP, color=GraSP_color, marker='s', markevery=markevery, linestyle='-', linewidth=linewidth,
    #                 markersize=markersize-5, label="GraSP", alpha=GraSP_alpha)
    # plt.fill_between(x_grid, y_GraSP - y_GraSP_err, y_GraSP + y_GraSP_err, color=GraSP_color, alpha=fill_in_alpha)


    # l_SynFlow = plt.plot(x_grid, y_SynFlow, color=SynFlow_color, marker='*', markevery=markevery, linestyle='-',
    #     linewidth=linewidth,
    #     markersize=markersize+7, label="SynFlow", alpha=SynFlow_alpha)
    # plt.fill_between(x_grid, y_SynFlow - y_SynFlow_err, y_SynFlow + y_SynFlow_err, color=SynFlow_color, alpha=fill_in_alpha)


    # l_HYDRA = plt.plot(x_grid, y_HYDRA, color=HYDRA_color, marker='v', markevery=markevery, linestyle='-',
    #     linewidth=linewidth,
    #     markersize=markersize+1, label="HYDRA", alpha=HYDRA_alpha)
    # plt.fill_between(x_grid, y_HYDRA - y_HYDRA_err, y_HYDRA + y_HYDRA_err, color=HYDRA_color, alpha=fill_in_alpha)


    # l_VPNs = plt.plot(x_grid, y_VPNs, color=VPNs_color, marker='*', markevery=markevery, linestyle='-', linewidth=linewidth,
    #                 markersize=markersize+7, label="VPNs", alpha=VPNs_alpha)
    # plt.fill_between(x_grid, y_VPNs - y_VPNs_err, y_VPNs + y_VPNs_err, color=VPNs_color, alpha=fill_in_alpha)



    l_random = plt.plot(x_grid, y_random, color=random_color, marker='o', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Random Prompt", alpha=random_alpha)
    plt.fill_between(x_grid, y_random - y_random_err, y_random + y_random_err, color=random_color, alpha=fill_in_alpha)

    l_fix = plt.plot(x_grid, y_fix, color=fix_color, marker='o', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Fix Prompt", alpha=fix_alpha)
    plt.fill_between(x_grid, y_fix - y_fix_err, y_fix + y_fix_err, color=fix_color, alpha=fill_in_alpha)

    l_pad = plt.plot(x_grid, y_pad, color=pad_color, marker='*', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize+7, label="VPNs (Pad Prompt)", alpha=pad_alpha)
    plt.fill_between(x_grid, y_pad - y_pad_err, y_pad + y_pad_err, color=pad_color, alpha=fill_in_alpha)


    lbest = plt.axhline(y=y_best, color=best_color, linestyle='--', linewidth=3, alpha=best_alpha,
                        label="Our Best")

    # dense_line = Line2D([0], [0], color=dense_color, lw=3, linestyle='--')
    # custom_lines = [dense_line,
    #                 Line2D([0], [0], color=OMP_color, lw=5),
    #                 Line2D([0], [0], color=Random_color, lw=5),
    #                 Line2D([0], [0], color=SNIP_color, lw=5),
    #                 Line2D([0], [0], color=SynFlow_color, lw=5),
    #                 Line2D([0], [0], color=HYDRA_color, lw=5)]

    # custom_markers = [Line2D([0], [0], marker='*', color='black', markerfacecolor='green', markersize=markersize+10),
    #                 Line2D([0], [0], marker='o', color='black', markerfacecolor='green', markersize=markersize+2)]

    # Then you can use these custom_lines to create your legend
    # legend1 = plt.legend(custom_lines, ['Dense', 'OMP', 'Random', 'SNIP', 'SynFlow', 'HYDRA'], loc='lower left', bbox_to_anchor=(0, 0.16), fontsize=fontsize - 8, fancybox=True, shadow=False, framealpha=0, borderpad=0.3)
    # plt.gca().add_artist(legend1)  # gca = "get current axis"

    # legend2 = plt.legend(custom_markers, ['Current Method', 'Post-pruning Prompt'], loc='lower left', fontsize=fontsize - 8, fancybox=True, shadow=False, framealpha=0, borderpad=0.3)
    # plt.gca().add_artist(legend2)


    plt.ylim([y_min, y_max])
    plt.xlim(0, 100)

    plt.legend(fontsize=fontsize - 8, loc=3, fancybox=True, shadow=False, framealpha=0, borderpad=0.3)
    plt.xlabel(x_label, fontsize=fontsize-2)
    plt.ylabel(y_label, fontsize=fontsize-2)
    plt.xticks(x_grid, x_sparsity_list, rotation=0, fontsize=fontsize-2)
    plt.xscale("linear")
    ax = plt.gca()  # Get the current Axes instance on the current figure
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.yticks(fontsize=fontsize-2)

    plt.title(title, fontsize=fontsize-2)
    plt.tight_layout()
    # plt.twinx()
    # y_time_label = "Time Consumption (min)"
    # linewidth = 2
    # time_linestyle = '-.'
    # tIMP = plt.plot(x_grid, y_IMP_time, color=IMP_color, alpha=IMP_alpha, label="IMP", linestyle=time_linestyle,
    #                 linewidth=linewidth)
    # tBiP = plt.plot(x_grid, y_BiP_time, color=BiP_color, alpha=BiP_alpha, label="BiP", linestyle=time_linestyle,
    #                  linewidth=linewidth)
    # tPFTT = plt.plot(x_grid, y_PFTT_time, color=PFTT_color, alpha=PFTT_alpha, label="PFTT", linestyle=time_linestyle,
    #                  linewidth=linewidth)
    # tHydra = plt.plot(x_grid, y_hydra_time, color=hydra_color, alpha=hydra_alpha, label="Hydra Global",
    #                   linestyle=time_linestyle, linewidth=linewidth)
    # tGrasp = plt.plot(x_grid, y_Grasp_time, color=Grasp_color, alpha=Grasp_alpha, label="Grasp",
    #                   linestyle=time_linestyle, linewidth=linewidth)
    # tOMP = plt.plot(x_grid, y_OMP_time, color=OMP_color, alpha=OMP_alpha + 0., label="OMP", linestyle=time_linestyle,
    #                 linewidth=linewidth)
    #
    # plt.xlabel(x_label, fontsize=fontsize)
    # plt.ylabel(y_time_label, fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.ylim(0, (int(max(y_IMP_time) / 100) + 1) * 100)
    plt.savefig(f"pic/ablation_vp/{title}.pdf")
    plt.show()
    plt.close()










