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
    title = 'VPNs w. OMP, CIFAR100'
    num, imp_num = 7, 11
    y_dense = 82.10
    y_min, y_max = 77,83.5
    
    # 10, 20
    x_sparsity_list = np.array([0, 40, 50, 60, 70, 80, 90, 95, 99][:num])
    x_grid = x_sparsity_list
    x_LTH_sparsity_list = np.array([0, 20.00, 36.00, 48.80, 59.00, 67.20, 73.80, 79.03, 83.22, 86.58, 89.26, 91.41, 93.13, 94.50, 95.60, 96.50, 97.75, 98.20, 98.56, 98.85][:imp_num])

    # LTH    ='82.02(0.28)	81.72(0.37)	81.36(0.37)	81.16(0.27)	80.76(0.17)	80.35(0.16)	79.70(0.09)	79.23(0.19)	78.91(0.20)	78.50(0.08)'
    # SNIP   ='80.89(0.15)	79.42(1.26)	78.35(1.49)	76.53(1.87)	74.39(1.77)	70.22(1.62)				'
    # GraSP  ='77.30(0.53)	76.33(0.23)	75.34(0.57)	73.89(0.63)	71.26(0.21)	69.09(0.60)				'
    # SynFlow='80.47(0.16)	79.32(0.37)	78.78(0.24)	77.31(0.18)	75.65(0.45)	71.49(0.15)				'
    # Random ='78.57(0.32)	76.03(0.63)	73.94(0.21)	71.03(0.74)	68.35(0.08)	66.14(0.62)				'
    # Random_VP ='79.37 	76.84 	74.56 	70.93 	69.51 	67.20 '
    OMP    ='81.60(0.22)	81.43(0.31)	81.26(0.07)	81.09(0.14)	80.03(0.15)	77.65(0.42)				'
    OMP_VP    ='82.46 	82.29 	82.02 	81.81 	80.84 	78.53 '
    # BiP    ='81.89(0.14)	81.91(0.27)	82.17(0.15)	81.81(0.34)	80.78(0.29)	79.52(0.05)				'
    # HYDRA  ='81.59(0.17)	81.11(0.15)	80.88(0.27)	80.01(0.10)	78.86(0.15)	77.12(0.16)				'
    VPNs   ='83.31(0.20)	83.18(0.13)	83.16(0.21)	82.74(0.21)	82.00(0.16)	80.58(0.37)				'

    # y_LTH, y_LTH_err = extract_y_err(y_dense, LTH)
    # y_SNIP, y_SNIP_err = extract_y_err(y_dense, SNIP)
    # y_GraSP, y_GraSP_err = extract_y_err(y_dense, GraSP)
    # y_SynFlow, y_SynFlow_err = extract_y_err(y_dense, SynFlow)
    # y_Random, y_Random_err = extract_y_err(y_dense, Random)
    y_OMP, y_OMP_err = extract_y_err(y_dense, OMP)
    y_OMP_VP, y_OMP_VP_err = extract_y_err(y_dense, OMP_VP)
    y_OMP_err=np.array([0 for _ in y_OMP_err])
    # y_BiP, y_BiP_err = extract_y_err(y_dense, BiP)
    # y_HYDRA, y_HYDRA_err = extract_y_err(y_dense, HYDRA)
    y_VPNs, y_VPNs_err = extract_y_err(y_dense, VPNs)
    y_VPNs_err=np.array([0 for _ in y_VPNs_err])
    # y_Random_VP, y_Random_VP_err = extract_y_err(y_dense, Random_VP)

    y_best = np.max(y_VPNs)

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
    LTH_color = '#dda15e'
    LTH_alpha = 0.9
    LTH_VP_color = '#bc6c25'
    LTH_VP_alpha = 0.9
    OMP_color = 'darkolivegreen'
    OMP_alpha = alpha
    OMP_prompt_color = 'darkolivegreen'
    OMP_prompt_alpha = alpha
    OMP_VP_color = 'darkolivegreen'
    OMP_VP_alpha = alpha
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


    l_OMP = plt.plot(x_grid, y_OMP, color=OMP_color, marker='v', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize+1, label="OMP", alpha=OMP_alpha)
    plt.fill_between(x_grid, y_OMP - y_OMP_err, y_OMP + y_OMP_err, color=OMP_color, alpha=fill_in_alpha)

    l_OMP_VP = plt.plot(x_grid, y_OMP_VP, color=OMP_VP_color, marker='*', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize+7, label="VPNs w. OMP", alpha=OMP_VP_alpha)
    plt.fill_between(x_grid, y_OMP_VP - y_OMP_VP_err, y_OMP_VP + y_OMP_VP_err, color=OMP_VP_color, alpha=fill_in_alpha)

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


    l_VPNs = plt.plot(x_grid, y_VPNs, color=VPNs_color, marker='*', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize+7, label="VPNs", alpha=VPNs_alpha)
    plt.fill_between(x_grid, y_VPNs - y_VPNs_err, y_VPNs + y_VPNs_err, color=VPNs_color, alpha=fill_in_alpha)

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
    plt.savefig(f"pic/algorithm_transfer/{title}.pdf")
    plt.show()
    plt.close()










