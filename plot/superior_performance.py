import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

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
    y = re.findall('(\d+\.\d+)(?=\()', performance_str)
    err = re.findall('(?<=\()(\d+\.\d+)', performance_str)
    y = [float(num) for num in y]
    err = [float(num) for num in err]

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
    title = 'ResNet-18, StanfordCars'
    num, imp_num = 9, 20
    y_dense = 85.39
    y_min, y_max = 50,86.5
    
    # 10, 20
    x_sparsity_list = np.array([0, 40, 50, 60, 70, 80, 90, 95, 99][:num])
    x_grid = x_sparsity_list
    x_IMP_sparsity_list = np.array([0, 20.00, 36.00, 48.80, 59.00, 67.20, 73.80, 79.03, 83.22, 86.58, 89.26, 91.41, 93.13, 94.50, 95.60, 96.50, 97.75, 98.20, 98.56, 98.85][:imp_num])
    
    # y_IMP = np.array([y_dense,73.01,72.72,72.36,71.97,71.18,70.49,69.52,68.43,67.17,65.89])
    # y_IMP_err = np.array([0,0.10,0.16,0.10,0.17,0.39,0.21,0.25,0.39,0.33,0.18])

    IMP    ='77.24(1.21)	72.31(3.86)	67.64(2.28)	64.45(0.68)	62.09(1.11)	58.77(1.43)	58.24(1.15)	58.40(1.16)	58.59(1.29)	58.53(1.24)	58.40(1.05)	58.23(0.88)	58.01(0.76)	57.91(1.08)	57.39(1.01)	56.28(1.02)	55.57(0.91)	54.71(1.22)	53.28(0.97)'
    SNIP   ='50.57(2.84)	42.87(0.79)	43.79(0.51)	40.77(2.01)	41.33(1.80)	38.99(0.96)	36.52(2.38)	17.40(1.59)'
    GraSP  ='81.12(0.33)	79.03(0.51)	78.25(1.59)	76.96(1.18)	73.15(1.03)	70.29(0.18)	65.47(1.03)	35.61(1.53)'
    SynFlow='84.34(0.58)	84.75(0.72)	84.58(0.58)	84.16(0.36)	83.32(0.36)	79.91(2.26)	75.37(0.60)	50.52(0.27)'
    Random ='81.81(0.53)	79.44(1.08)	73.16(1.23)	67.05(0.88)	63.42(2.37)	52.37(0.54)	45.53(0.31)	15.88(1.80)'
    OMP    ='70.61(2.93)	68.25(5.08)	64.32(1.74)	63.51(1.72)	60.23(0.79)	58.98(1.27)	57.17(1.03)	42.70(1.17)'
    BiP    ='83.42(0.21)	83.46(0.48)	83.30(0.42)	83.06(0.29)	82.55(0.37)	80.79(0.42)	78.22(0.42)	56.82(2.45)'
    HYDRA  ='83.67(0.10)	83.88(0.30)	83.36(0.31)	83.12(0.25)	83.26(0.29)	82.60(0.26)	82.06(0.35)	63.70(0.01)'
    VPNs   ='85.69(0.21)	85.60(0.30)	85.79(0.05)	85.92(0.16)	86.02(0.14)	84.51(0.29)	82.55(0.80)	63.74(1.06)'

    y_IMP, y_IMP_err = extract_y_err(y_dense, IMP)
    y_SNIP, y_SNIP_err = extract_y_err(y_dense, SNIP)
    y_GraSP, y_GraSP_err = extract_y_err(y_dense, GraSP)
    y_SynFlow, y_SynFlow_err = extract_y_err(y_dense, SynFlow)
    y_Random, y_Random_err = extract_y_err(y_dense, Random)
    y_OMP, y_OMP_err = extract_y_err(y_dense, OMP)
    y_BiP, y_BiP_err = extract_y_err(y_dense, BiP)
    y_HYDRA, y_HYDRA_err = extract_y_err(y_dense, HYDRA)
    y_VPNs, y_VPNs_err = extract_y_err(y_dense, VPNs)

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
    VPNs_color = 'green'
    VPNs_alpha = 0.9
    BiP_color = 'hotpink'
    BiP_alpha = alpha
    HYDRA_color = 'blue'
    HYDRA_alpha = alpha - 0.1
    IMP_color = 'darkorange'
    IMP_alpha = alpha
    OMP_color = 'darkolivegreen'
    OMP_alpha = alpha
    Grasp_color = 'purple'
    Grasp_alpha = alpha
    SNIP_color = 'gold'
    SNIP_alpha = alpha
    Random_color = 'violet'
    Random_alpha = alpha


    fill_in_alpha = 0.2

    # plt.rcParams['font.sans-serif'] = 'Times New Roman'
    # plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    # # Show the minor grid lines with very faint and almost transparent grey lines
    # plt.minorticks_on()
    # plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    l_dense = plt.axhline(y=y_dense, color=dense_color, linestyle='--', linewidth=3, label="Dense")

    l_HYDRA = plt.plot(x_grid, y_HYDRA, color=HYDRA_color, marker='v', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Hydra", alpha=HYDRA_alpha)
    plt.fill_between(x_grid, y_HYDRA - y_HYDRA_err, y_HYDRA + y_HYDRA_err, color=HYDRA_color, alpha=fill_in_alpha)

    l_IMP = plt.plot(x_IMP_sparsity_list, y_IMP, color=IMP_color, marker='o', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="IMP", alpha=IMP_alpha)
    plt.fill_between(x_IMP_sparsity_list, y_IMP - y_IMP_err, y_IMP + y_IMP_err, color=IMP_color, alpha=fill_in_alpha)

    l_OMP = plt.plot(x_grid, y_OMP, color=OMP_color, marker='s', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="OMP", alpha=OMP_alpha)
    plt.fill_between(x_grid, y_OMP - y_OMP_err, y_OMP + y_OMP_err, color=OMP_color, alpha=fill_in_alpha)

    l_GraSP = plt.plot(x_grid, y_GraSP, color=Grasp_color, marker='o', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Grasp", alpha=Grasp_alpha)
    plt.fill_between(x_grid, y_GraSP - y_GraSP_err, y_GraSP + y_GraSP_err, color=Grasp_color, alpha=fill_in_alpha)

    l_BiP = plt.plot(x_grid, y_BiP, color=BiP_color, marker='v', markevery=markevery, linestyle='-',
                    linewidth=linewidth,
                    markersize=markersize + 4, label="BiP", alpha=BiP_alpha)
    plt.fill_between(x_grid, y_BiP - y_BiP_err, y_BiP + y_BiP_err, color=BiP_color, alpha=fill_in_alpha)

    l_SynFlow = plt.plot(x_grid, y_SynFlow, color=SynFlow_color, marker='v', markevery=markevery, linestyle='-',
                linewidth=linewidth,
                markersize=markersize + 4, label="SynFlow", alpha=SynFlow_alpha)
    plt.fill_between(x_grid, y_SynFlow - y_SynFlow_err, y_SynFlow + y_SynFlow_err, color=SynFlow_color, alpha=fill_in_alpha)

    l_SNIP = plt.plot(x_grid, y_SNIP, color=SNIP_color, marker='o', markevery=markevery, linestyle='-',
                linewidth=linewidth,
                markersize=markersize + 4, label="SNIP", alpha=SNIP_alpha)
    plt.fill_between(x_grid, y_SNIP - y_SNIP_err, y_SNIP + y_SNIP_err, color=SNIP_color, alpha=fill_in_alpha)

    l_Random = plt.plot(x_grid, y_Random, color=Random_color, marker='o', markevery=markevery, linestyle='-',
            linewidth=linewidth,
            markersize=markersize + 4, label="Random", alpha=Random_alpha)
    plt.fill_between(x_grid, y_Random - y_Random_err, y_Random + y_Random_err, color=Random_color, alpha=fill_in_alpha)

    l_VPNs = plt.plot(x_grid, y_VPNs, color=VPNs_color, marker='*', markevery=markevery, linestyle='-',
        linewidth=linewidth,
        markersize=markersize + 4, label="VPNs", alpha=VPNs_alpha)
    plt.fill_between(x_grid, y_VPNs - y_VPNs_err, y_VPNs + y_VPNs_err, color=VPNs_color, alpha=fill_in_alpha)

    lbest = plt.axhline(y=y_best, color=best_color, linestyle='--', linewidth=3, alpha=best_alpha,
                        label="Best Winning Ticket")
    # lPFTT = plt.plot(x_grid, y_PFTT, color=PFTT_color, marker='*', markevery=markevery, linestyle='-',
    #                  linewidth=linewidth,
    #                  markersize=markersize + 4, label="PFTT", alpha=PFTT_alpha)
    # plt.fill_between(x_grid, y_PFTT - y_PFTT_err, y_PFTT + y_PFTT_err, color=PFTT_color, alpha=fill_in_alpha)

    plt.ylim([y_min, y_max])
    plt.xlim(0, 100)

    # plt.legend(fontsize=fontsize - 20, loc=3, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
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
    plt.savefig(f"pic/superior_performance/{title}.pdf")
    plt.show()
    plt.close()










