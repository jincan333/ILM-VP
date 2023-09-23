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
    title = 'Post-pruning Prompt with Zero-shot'
    num, imp_num = 7, 11
    y_dense = 96.09
    y_min, y_max = 10,99
    
    # 10, 20
    x_sparsity_list = np.array([0, 40, 50, 60, 70, 80, 90, 95, 99][:num])
    x_grid = x_sparsity_list
    x_IMP_sparsity_list = np.array([0, 20.00, 36.00, 48.80, 59.00, 67.20, 73.80, 79.03, 83.22, 86.58, 89.26, 91.41, 93.13, 94.50, 95.60, 96.50, 97.75, 98.20, 98.56, 98.85][:imp_num])
    
    # y_IMP = np.array([y_dense,73.01,72.72,72.36,71.97,71.18,70.49,69.52,68.43,67.17,65.89])
    # y_IMP_err = np.array([0,0.10,0.16,0.10,0.17,0.39,0.21,0.25,0.39,0.33,0.18])

    OMP       ='71.17 	69.03 	56.54 	31.59 	18.16 	14.67 '
    Random    ='15.11 	13.84 	10.02 	10.00 	10.00 	10.00 '
    SNIP      ='25.80 	20.28 	13.51 	19.61 	10.19 	10.02 '
    SynFlow   ='22.70 	17.96 	25.37 	14.54 	15.28 	10.00 '
    HYDRA     ='94.24 	94.47 	94.42 	94.23 	94.23 	94.02 '

    OMP_prompt    ='88.74 	88.13 	85.92 	84.02 	80.43 	78.23 '
    Random_prompt ='46.68 	42.64 	41.49 	40.60 	38.06 	37.98 '
    SNIP_prompt   ='58.53 	53.61 	53.13 	53.54 	48.11 	44.17 '
    SynFlow_prompt='56.36 	54.21 	54.14 	52.71 	51.50 	46.28 '
    HYDRA_prompt  ='93.61 	93.47 	94.39 	94.02 	93.88 	93.82 '


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
    SNIP_color = 'darkorange'
    SNIP_alpha = alpha
    SNIP_prompt_color = 'darkorange'
    SNIP_prompt_alpha = alpha
    Random_color = 'violet'
    Random_alpha = alpha
    Random_prompt_color = 'violet'
    Random_prompt_alpha = alpha


    fill_in_alpha = 0.2

    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    # plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    # # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    # plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    l_dense = plt.axhline(y=y_dense, color=dense_color, linestyle='--', linewidth=3, label="Dense")


    l_OMP = plt.plot(x_grid, y_OMP, color=OMP_color, marker='*', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize+7, label="OMP", alpha=OMP_alpha)
    plt.fill_between(x_grid, y_OMP - y_OMP_err, y_OMP + y_OMP_err, color=OMP_color, alpha=fill_in_alpha)

    l_OMP_prompt = plt.plot(x_grid, y_OMP_prompt, color=OMP_prompt_color, marker='o', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="OMP prompt", alpha=OMP_prompt_alpha)
    plt.fill_between(x_grid, y_OMP_prompt - y_OMP_prompt_err, y_OMP_prompt + y_OMP_prompt_err, color=OMP_prompt_color, alpha=fill_in_alpha)

    l_Random = plt.plot(x_grid, y_Random, color=Random_color, marker='*', markevery=markevery, linestyle='-',
            linewidth=linewidth,
            markersize=markersize+7, label="Random", alpha=Random_alpha)
    plt.fill_between(x_grid, y_Random - y_Random_err, y_Random + y_Random_err, color=Random_color, alpha=fill_in_alpha)

    l_Random_VP = plt.plot(x_grid, y_Random_prompt, color=Random_prompt_color, marker='o', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="Random prompt", alpha=Random_prompt_alpha)
    plt.fill_between(x_grid, y_Random_prompt - y_Random_prompt_err, y_Random_prompt + y_Random_prompt_err, color=Random_prompt_color, alpha=fill_in_alpha)


    l_SNIP = plt.plot(x_grid, y_SNIP, color=SNIP_color, marker='*', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize+7, label="SNIP", alpha=SNIP_alpha)
    plt.fill_between(x_grid, y_SNIP - y_SNIP_err, y_SNIP + y_SNIP_err, color=SNIP_color, alpha=fill_in_alpha)

    l_SNIP_prompt = plt.plot(x_grid, y_SNIP_prompt, color=SNIP_prompt_color, marker='o', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="SNIP prompt", alpha=SNIP_prompt_alpha)
    plt.fill_between(x_grid, y_SNIP_prompt - y_SNIP_prompt_err, y_SNIP_prompt + y_SNIP_prompt_err, color=SNIP_prompt_color, alpha=fill_in_alpha)


    l_SynFlow = plt.plot(x_grid, y_SynFlow, color=SynFlow_color, marker='*', markevery=markevery, linestyle='-',
        linewidth=linewidth,
        markersize=markersize+7, label="SynFlow", alpha=SynFlow_alpha)
    plt.fill_between(x_grid, y_SynFlow - y_SynFlow_err, y_SynFlow + y_SynFlow_err, color=SynFlow_color, alpha=fill_in_alpha)

    l_SynFlow_prompt = plt.plot(x_grid, y_SynFlow_prompt, color=SynFlow_prompt_color, marker='o', markevery=markevery, linestyle='-',
        linewidth=linewidth,
        markersize=markersize, label="SynFlow prompt", alpha=SynFlow_prompt_alpha)
    plt.fill_between(x_grid, y_SynFlow_prompt - y_SynFlow_prompt_err, y_SynFlow_prompt + y_SynFlow_prompt_err, color=SynFlow_prompt_color, alpha=fill_in_alpha)


    l_HYDRA = plt.plot(x_grid, y_HYDRA, color=HYDRA_color, marker='*', markevery=markevery, linestyle='-',
        linewidth=linewidth,
        markersize=markersize+7, label="HYDRA", alpha=HYDRA_alpha)
    plt.fill_between(x_grid, y_HYDRA - y_HYDRA_err, y_HYDRA + y_HYDRA_err, color=HYDRA_color, alpha=fill_in_alpha)

    l_HYDRA_prompt = plt.plot(x_grid, y_HYDRA_prompt, color=HYDRA_prompt_color, marker='o', markevery=markevery, linestyle='-',
        linewidth=linewidth,
        markersize=markersize, label="HYDRA prompt", alpha=HYDRA_prompt_alpha)
    plt.fill_between(x_grid, y_HYDRA_prompt - y_HYDRA_prompt_err, y_HYDRA_prompt + y_HYDRA_prompt_err, color=HYDRA_prompt_color, alpha=fill_in_alpha)

    dense_line = Line2D([0], [0], color=dense_color, lw=3, linestyle='--')
    custom_lines = [dense_line,
                    Line2D([0], [0], color=OMP_color, lw=5),
                    Line2D([0], [0], color=Random_color, lw=5),
                    Line2D([0], [0], color=SNIP_color, lw=5),
                    Line2D([0], [0], color=SynFlow_color, lw=5),
                    Line2D([0], [0], color=HYDRA_color, lw=5)]

    custom_markers = [Line2D([0], [0], marker='*', color='black', markerfacecolor='black', markersize=markersize+10),
                    Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markersize=markersize+2)]

    # Then you can use these custom_lines to create your legend
    # legend1 = plt.legend(custom_lines, ['Dense', 'OMP', 'Random', 'SNIP', 'SynFlow', 'HYDRA'], loc='lower left', bbox_to_anchor=(0, 0.16), fontsize=fontsize - 8, fancybox=True, shadow=False, framealpha=0, borderpad=0.3)
    # plt.gca().add_artist(legend1)  # gca = "get current axis"

    # legend2 = plt.legend(custom_markers, ['w.o. Post-pruning Prompt', 'w. Post-pruning Prompt'], loc='lower left', fontsize=fontsize - 8, fancybox=True, shadow=False, framealpha=0, borderpad=0.3)
    # plt.gca().add_artist(legend2)


    plt.ylim([y_min, y_max])
    plt.xlim(0, 100)

    # plt.legend(fontsize=fontsize - 20, loc=4, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
    plt.xlabel(x_label, fontsize=fontsize-2)
    plt.ylabel(y_label, fontsize=fontsize-2)
    plt.xticks(x_grid, x_sparsity_list, rotation=0, fontsize=fontsize-2)
    plt.xscale("linear")
    ax = plt.gca()  # Get the current Axes instance on the current figure
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.yticks(fontsize=fontsize-2)

    plt.title('(a) Zero-shot', fontsize=fontsize)
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










