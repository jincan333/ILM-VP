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
    title = 'ResNet-18, CIFAR100'
    num, imp_num = 7, 11
    y_dense = 83.59
    y_min, y_max = 60,84.5
    
    # 10, 20
    x_sparsity_list = np.array([0, 40, 50, 60, 70, 80, 90, 95, 99][:num])
    x_grid = x_sparsity_list
    x_IMP_sparsity_list = np.array([0, 20.00, 36.00, 48.80, 59.00, 67.20, 73.80, 79.03, 83.22, 86.58, 89.26, 91.41, 93.13, 94.50, 95.60, 96.50, 97.75, 98.20, 98.56, 98.85][:imp_num])
    
    # y_IMP = np.array([y_dense,73.01,72.72,72.36,71.97,71.18,70.49,69.52,68.43,67.17,65.89])
    # y_IMP_err = np.array([0,0.10,0.16,0.10,0.17,0.39,0.21,0.25,0.39,0.33,0.18])

    Slim      ='83.05 	82.58 	81.08 	77.40 	66.69 	4.58'
    DepGraph  ='80.02 	79.14 	75.62 	61.78 	6.36 	3.67'
    ABCpruner ='80.02 	79.14 	75.62 	61.78 	6.36 	3.67'
    VPNs      ='83.85 	82.86 	81.96 	80.47 	78.92 	74.10'

    y_Slim, y_Slim_err = extract_y_err(y_dense, Slim)
    y_DepGraph, y_DepGraph_err = extract_y_err(y_dense, DepGraph)
    y_ABCpruner, y_ABCpruner_err = extract_y_err(y_dense, ABCpruner)
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
    VPNs_color = 'green'
    VPNs_alpha = 0.9
    # SynFlow_color = 'red'
    # SynFlow_alpha = 0.9
    # BiP_color = 'hotpink'
    # BiP_alpha = alpha
    # HYDRA_color = 'blue'
    # HYDRA_alpha = alpha - 0.1
    # IMP_color = 'darkorange'
    # IMP_alpha = alpha
    # OMP_color = 'darkolivegreen'
    # OMP_alpha = alpha
    # Grasp_color = 'purple'
    # Grasp_alpha = alpha
    # SNIP_color = 'gold'
    # SNIP_alpha = alpha
    # Random_color = 'violet'
    # Random_alpha = alpha
    Slim_color = 'violet'
    Slim_alpha = alpha
    DepGraph_color = 'darkolivegreen'
    DepGraph_alpha = alpha
    ABCpruner_color = 'blue'
    ABCpruner_alpha = alpha - 0.1
    fill_in_alpha = 0.2

    # plt.rcParams['font.sans-serif'] = 'Times New Roman'
    # plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    # # Show the minor grid lines with very faint and almost transparent grey lines
    # plt.minorticks_on()
    # plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    l_dense = plt.axhline(y=y_dense, color=dense_color, linestyle='--', linewidth=3, label="Dense")

    l_Slim = plt.plot(x_grid, y_Slim, color=Slim_color, marker='v', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Slim", alpha=Slim_alpha)
    plt.fill_between(x_grid, y_Slim - y_Slim_err, y_Slim + y_Slim_err, color=Slim_color, alpha=fill_in_alpha)

    l_DepGraph = plt.plot(x_grid, y_DepGraph, color=DepGraph_color, marker='s', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="DepGraph", alpha=DepGraph_alpha)
    plt.fill_between(x_grid, y_DepGraph - y_DepGraph_err, y_DepGraph + y_DepGraph_err, color=DepGraph_color, alpha=fill_in_alpha)

    l_ABCpruner = plt.plot(x_grid, y_ABCpruner, color=ABCpruner_color, marker='s', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="ABCpruner", alpha=ABCpruner_alpha)
    plt.fill_between(x_grid, y_ABCpruner - y_ABCpruner_err, y_ABCpruner + y_ABCpruner_err, color=ABCpruner_color, alpha=fill_in_alpha)

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

    plt.legend(fontsize=fontsize - 15, loc=3, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
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
    plt.savefig(f"pic/structural/{title}.pdf")
    plt.show()
    plt.close()










