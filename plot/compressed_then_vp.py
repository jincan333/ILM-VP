import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from matplotlib.lines import Line2D 

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
    title = 'Post-pruning Prompt'
    num, imp_num = 7, 11
    y_dense = 96.09
    y_min, y_max = 85,96.5
    
    # 10, 20
    x_sparsity_list = np.array([0, 40, 50, 60, 70, 80, 90, 95, 99][:num])
    x_grid = x_sparsity_list
    x_IMP_sparsity_list = np.array([0, 20.00, 36.00, 48.80, 59.00, 67.20, 73.80, 79.03, 83.22, 86.58, 89.26, 91.41, 93.13, 94.50, 95.60, 96.50, 97.75, 98.20, 98.56, 98.85][:imp_num])
    
    # y_IMP = np.array([y_dense,73.01,72.72,72.36,71.97,71.18,70.49,69.52,68.43,67.17,65.89])
    # y_IMP_err = np.array([0,0.10,0.16,0.10,0.17,0.39,0.21,0.25,0.39,0.33,0.18])

    OMP       ='95.98 	95.86 	95.89 	95.79 	95.76 	95.01 '
    Random    ='95.05 	94.19 	93.26 	92.06 	90.59 	89.75 '
    SNIP      ='95.90 	95.71 	95.39 	95.20 	94.72 	93.77 '
    SynFlow   ='95.69 	95.81 	95.51 	95.38 	94.94 	93.81 '
    HYDRA     ='96.00 	95.98 	95.86 	95.63 	95.31 	94.67 '

    OMP_prompt    ='86.11 	88.08 	89.30 	88.84 	78.56 	0'
    Random_prompt ='46.68 	42.64 	41.49 	40.60 	0 0	'
    SNIP_prompt   ='58.53 	53.61 	53.13 	53.54 	0 0	'
    SynFlow_prompt='56.36 	54.21 	54.14 	52.71 	51.50 	46.28 '
    HYDRA_prompt  ='90.28 	93.19 	94.39 	94.02 	93.88 	93.82 '


    y_OMP, y_OMP_err = extract_y_err(y_dense, OMP)
    y_OMP_prompt, y_OMP_prompt_err = extract_y_err(y_dense, OMP_prompt)
    y_Random, y_Random_err = extract_y_err(y_dense, Random)
    y_Random_prompt, y_Random_prompt_err = extract_y_err(y_dense, Random_prompt)
    y_SNIP, y_SNIP_err = extract_y_err(y_dense, SNIP)
    y_SNIP_prompt, y_SNIP_prompt_err = extract_y_err(y_dense, SNIP_prompt)
    y_SynFlow, y_SynFlow_err = extract_y_err(y_dense, SynFlow)
    y_SynFlow_prompt, y_SynFlow_prompt_err = extract_y_err(y_dense, SynFlow_prompt)
    y_HYDRA, y_HYDRA_err = extract_y_err(y_dense, HYDRA)
    y_HYDRA_prompt, y_HYDRA_prompt_err = extract_y_err(y_dense, HYDRA_prompt)

    # print("IMP winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_IMP, num)
    # print("Grasp winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_Grasp, num)
    # print("OMP winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_OMP, num)
    # print("Hydra winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_hydra, num)
    # print("BiP winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_BiP, num)

    # print("last IMP winning ticket:")
    # last_winning_ticket_sparsity(y_IMP[0], y_IMP, num, y_IMP_err)
    # print("last Grasp winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_Grasp, num, y_grasp_err)
    # print("last OMP winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_OMP, num, y_OMP_err)
    # print("last Hydra winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_hydra, num, y_hydra_err)
    # print("last BiP winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_BiP, num, y_BiP_err)


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
    linewidth = 2
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
    BiP_color = 'hotpink'
    BiP_alpha = alpha
    HYDRA_color = 'blue'
    HYDRA_alpha = alpha - 0.1

    HYDRA_prompt_color = 'blue'
    HYDRA_prompt_alpha = alpha - 0.1
    IMP_color = 'darkorange'
    IMP_alpha = alpha
    IMP_VP_color = 'red'
    IMP_VP_alpha = 0.9
    OMP_color = 'darkolivegreen'
    OMP_alpha = alpha
    OMP_prompt_color = 'darkolivegreen'
    OMP_prompt_alpha = alpha
    Grasp_color = 'purple'
    Grasp_alpha = alpha
    SNIP_color = 'gold'
    SNIP_alpha = alpha
    SNIP_prompt_color = 'gold'
    SNIP_prompt_alpha = alpha
    Random_color = 'violet'
    Random_alpha = alpha
    Random_prompt_color = 'violet'
    Random_prompt_alpha = alpha


    fill_in_alpha = 0.2

    # plt.rcParams['font.sans-serif'] = 'Times New Roman'
    # plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    # # Show the minor grid lines with very faint and almost transparent grey lines
    # plt.minorticks_on()
    # plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    l_dense = plt.axhline(y=y_dense, color=dense_color, linestyle='--', linewidth=3, label="Dense")


    l_OMP = plt.plot(x_grid, y_OMP, color=OMP_color, marker='*', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="OMP", alpha=OMP_alpha)
    plt.fill_between(x_grid, y_OMP - y_OMP_err, y_OMP + y_OMP_err, color=OMP_color, alpha=fill_in_alpha)

    l_OMP_prompt = plt.plot(x_grid, y_OMP_prompt, color=OMP_prompt_color, marker='o', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="OMP prompt", alpha=OMP_prompt_alpha)
    plt.fill_between(x_grid, y_OMP_prompt - y_OMP_prompt_err, y_OMP_prompt + y_OMP_prompt_err, color=OMP_prompt_color, alpha=fill_in_alpha)

    l_Random = plt.plot(x_grid, y_Random, color=Random_color, marker='*', markevery=markevery, linestyle='-',
            linewidth=linewidth,
            markersize=markersize, label="Random", alpha=Random_alpha)
    plt.fill_between(x_grid, y_Random - y_Random_err, y_Random + y_Random_err, color=Random_color, alpha=fill_in_alpha)

    l_Random_VP = plt.plot(x_grid, y_Random_prompt, color=Random_prompt_color, marker='o', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="Random prompt", alpha=Random_prompt_alpha)
    plt.fill_between(x_grid, y_Random_prompt - y_Random_prompt_err, y_Random_prompt + y_Random_prompt_err, color=Random_prompt_color, alpha=fill_in_alpha)


    l_SNIP = plt.plot(x_grid, y_SNIP, color=SNIP_color, marker='*', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="SNIP", alpha=SNIP_alpha)
    plt.fill_between(x_grid, y_SNIP - y_SNIP_err, y_SNIP + y_SNIP_err, color=SNIP_color, alpha=fill_in_alpha)

    l_SNIP_prompt = plt.plot(x_grid, y_SNIP_prompt, color=SNIP_prompt_color, marker='o', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="SNIP prompt", alpha=SNIP_prompt_alpha)
    plt.fill_between(x_grid, y_SNIP_prompt - y_SNIP_prompt_err, y_SNIP_prompt + y_SNIP_prompt_err, color=SNIP_prompt_color, alpha=fill_in_alpha)


    l_SynFlow = plt.plot(x_grid, y_SynFlow, color=SynFlow_color, marker='*', markevery=markevery, linestyle='-',
        linewidth=linewidth,
        markersize=markersize, label="SynFlow", alpha=SynFlow_alpha)
    plt.fill_between(x_grid, y_SynFlow - y_SynFlow_err, y_SynFlow + y_SynFlow_err, color=SynFlow_color, alpha=fill_in_alpha)

    l_SynFlow_prompt = plt.plot(x_grid, y_SynFlow_prompt, color=SynFlow_prompt_color, marker='o', markevery=markevery, linestyle='-',
        linewidth=linewidth,
        markersize=markersize, label="SynFlow prompt", alpha=SynFlow_prompt_alpha)
    plt.fill_between(x_grid, y_SynFlow_prompt - y_SynFlow_prompt_err, y_SynFlow_prompt + y_SynFlow_prompt_err, color=SynFlow_prompt_color, alpha=fill_in_alpha)


    l_HYDRA = plt.plot(x_grid, y_HYDRA, color=HYDRA_color, marker='*', markevery=markevery, linestyle='-',
        linewidth=linewidth,
        markersize=markersize, label="HYDRA", alpha=HYDRA_alpha)
    plt.fill_between(x_grid, y_HYDRA - y_HYDRA_err, y_HYDRA + y_HYDRA_err, color=HYDRA_color, alpha=fill_in_alpha)

    l_HYDRA_prompt = plt.plot(x_grid, y_HYDRA_prompt, color=HYDRA_prompt_color, marker='o', markevery=markevery, linestyle='-',
        linewidth=linewidth,
        markersize=markersize, label="HYDRA prompt", alpha=HYDRA_prompt_alpha)
    plt.fill_between(x_grid, y_HYDRA_prompt - y_HYDRA_prompt_err, y_HYDRA_prompt + y_HYDRA_prompt_err, color=HYDRA_prompt_color, alpha=fill_in_alpha)

    custom_lines = [Line2D([0], [0], color=OMP_color, lw=2),
                    Line2D([0], [0], color=Random_color, lw=2),
                    Line2D([0], [0], color=SNIP_color, lw=2),
                    Line2D([0], [0], color=SynFlow_color, lw=2),
                    Line2D([0], [0], color=HYDRA_color, lw=2)]

    custom_markers = [Line2D([0], [0], marker='*', color='black', markerfacecolor='green', markersize=18),
                    Line2D([0], [0], marker='o', color='black', markerfacecolor='green', markersize=15)]

    # Then you can use these custom_lines to create your legend
    legend1 = plt.legend(custom_lines, ['OMP', 'Random', 'SNIP', 'SynFlow', 'HYDRA'], loc='lower right', bbox_to_anchor=(1.0, 0.14), fontsize=fontsize - 22, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
    plt.gca().add_artist(legend1)  # gca = "get current axis"

    legend2 = plt.legend(custom_markers, ['Current Method', 'Post-pruning Prompt'], loc='lower right', fontsize=fontsize - 22, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
    plt.gca().add_artist(legend2)


    plt.ylim([y_min, y_max])
    plt.xlim(0, 100)

    # plt.legend(fontsize=fontsize - 20, loc=4, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(x_grid, x_sparsity_list, rotation=0, fontsize=fontsize)
    plt.xscale("linear")
    plt.yticks(fontsize=fontsize)

    plt.title(title, fontsize=fontsize)
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
    plt.savefig(f"pic/compressed_then_vp/{title}.pdf")
    plt.show()
    plt.close()










