import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

if __name__ == "__main__":
    num = 14
    # x_grid = np.array(range(num))
    step = 1
    index = np.arange(0, num, step)
    x = np.arange(num)[index]
    x_density = 100 - 100 * (0.8 ** x)
    x_grid = x_density
    x_density_list = ['{:.2f}'.format(value) for value in x_density]

    num = 14
    x_grid = np.array(range(num))
    step = 1
    index = np.arange(0, num, step)
    x = np.arange(num)[index]
    x_density = 100 - 100 * (0.8 ** x)
    x_grid = x_density
    x_density_list = ['{:.2f}'.format(value) for value in x_density]

    y_PFTT_time = np.insert(np.array([180 for i in range(num - 1)]), 0, 0)
    y_BiP_time = np.insert(np.array([225 for i in range(num - 1)]), 0, 0)
    y_hydra_time = np.insert(np.array([136 for i in range(num - 1)]), 0, 0)
    y_IMP_time = np.array([115.2 * i for i in range(num)])
    y_OMP_time = np.insert(np.array([115.2 for i in range(num - 1)]), 0, 0)
    y_Grasp_time = np.insert(np.array([120 for i in range(num - 1)]), 0, 0)

    y_PFTT = np.array(
        [93.65, 94.09, 94.29, 94.09, 94.14, 94.21, 94.00, 94.28, 94.13, 94.09, 94.01, 93.77, 93.70, 93.68, 93.33, 88.14,
         10.00, 10.00, 10.00, 10.00, 10.00, 10.00
         ])[:num]
    y_PFTT_err = np.array(
        [0.00, 0.02, 0.10, 0.05, 0.07, 0.04, 0.12, 0.02, 0.05, 0.01, 0.10, 0.13, 0.08, 0.12, 0.14, 0.79, 0.00, 0.00,
         0.00, 0.00, 0.00, 0.00
         ])[:num]

    y_BiP = np.array(
        [
            93.65, 94.25, 94.16, 94.31, 94.21, 94.13, 94.02, 93.94, 93.95, 93.92, 93.89, 93.75, 93.52, 93.62, 93.44,
            85.58, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00
        ])[:num]
    y_BiP_err = np.array(
        [

            0.00, 0.04, 0.18, 0.11, 0.25, 0.21, 0.03, 0.07, 0.19, 0.21, 0.09, 0.15, 0.20, 0.23, 0.24, 4.01, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00
        ])[:num]

    y_hydra = np.array(
        [
            93.65, 93.94, 93.86, 93.95, 93.80, 93.82, 93.80, 93.83, 93.65, 93.73, 93.39, 93.52, 93.26, 93.37, 93.24,
            37.73, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00
        ])[:num]
    y_hydra_err = np.array(
        [

            0.00, 0.07, 0.08, 0.11, 0.03, 0.12, 0.08, 0.09, 0.12, 0.03, 0.28, 0.27, 0.13, 0.24, 0.27, 48.03, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00
        ])[:num]

    y_IMP = np.array(
        [

            93.65, 93.79, 93.81, 93.75, 93.69, 93.65, 93.57, 93.49, 93.6, 93.38, 93.68, 93.5, 93.32, 93.5

        ])[:num]
    y_IMP_err = np.array(
        [

            0.00, 0.06, 0.05, 0.05, 0.25, 0.14, 0.19, 0.06, 0.19, 0.26, 0.05, 0.23, 0.32, 0.23, 0.11, 0.07, 0.12, 0.35,
            0.53, 0.41, 0.71, 2.33

        ])[:num]

    y_OMP = np.array(
        [

            93.65, 93.79, 93.64, 93.22, 93.56, 93.15, 93.30, 93.34, 93.40, 93.17, 93.33, 93.45, 92.78, 93.50
        ])[:num]
    y_OMP_err = np.array(
        [
            0.00, 0.06, 0.31, 0.02, 0.06, 0.22, 0.28, 0.34, 0.34, 0.42, 0.12, 0.27, 0.34, 0.66, 0.53, 0.34, 0.30, 0.15,
            0.20, 0.38, 0.51, 1.02
        ])[:num]

    y_Grasp = np.array(
        [
            93.65, 93.39, 93.44, 93.12, 93.20, 93.24, 93.08, 93.02, 92.83, 92.71, 92.59, 92.80, 92.60, 92.57, 92.47,
            10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00
        ])[:num]
    y_grasp_err = np.array(
        [
            0.00, 0.23, 0.17, 0.19, 0.31, 0.20, 0.20, 0.07, 0.17, 0.36, 0.07, 0.12, 0.06, 0.09, 0.22, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00
        ])[:num]

    dense = y_Grasp[0]
    best = y_BiP[3]

    print("IMP winning ticket gap:")
    winning_ticket_gap(y_IMP[0], y_IMP, num)
    print("Grasp winning ticket gap:")
    winning_ticket_gap(y_IMP[0], y_Grasp, num)
    print("OMP winning ticket gap:")
    winning_ticket_gap(y_IMP[0], y_OMP, num)
    print("Hydra winning ticket gap:")
    winning_ticket_gap(y_IMP[0], y_hydra, num)
    print("BiP winning ticket gap:")
    winning_ticket_gap(y_IMP[0], y_BiP, num)

    print("last IMP winning ticket:")
    last_winning_ticket_sparsity(y_IMP[0], y_IMP, num, y_IMP_err)
    print("last Grasp winning ticket gap:")
    last_winning_ticket_sparsity(y_IMP[0], y_Grasp, num, y_grasp_err)
    print("last OMP winning ticket gap:")
    last_winning_ticket_sparsity(y_IMP[0], y_OMP, num, y_OMP_err)
    print("last Hydra winning ticket gap:")
    last_winning_ticket_sparsity(y_IMP[0], y_hydra, num, y_hydra_err)
    print("last BiP winning ticket gap:")
    last_winning_ticket_sparsity(y_IMP[0], y_BiP, num, y_BiP_err)

    y_min = 92.0
    # y_max = int(max(max(y_BiP), max(y_IMP))) + 1
    y_max = 94.5
    x_label = "Pruning Ratio (%)"
    y_label = "Test Accuracy (%)"

    # Canvas setting
    width = 14
    height = 12
    plt.figure(figsize=(width, height))



    sns.set_theme()
    plt.grid(visible=True, which='major', linestyle='-', linewidth=4)
    plt.grid(visible=True, which='minor')
    plt.minorticks_on()
    plt.rcParams['font.sans-serif'] = 'Times New Roman'

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
    PFTT_color = 'red'
    PFTT_alpha = 0.9
    BiP_color = 'green'
    BiP_alpha = 0.9
    hydra_color = 'blue'
    hydra_alpha = alpha - 0.1
    IMP_color = 'darkorange'
    IMP_alpha = alpha
    OMP_color = 'darkolivegreen'
    OMP_alpha = alpha
    Grasp_color = 'purple'
    Grasp_alpha = alpha

    fill_in_alpha = 0.2

    # plt.rcParams['font.sans-serif'] = 'Times New Roman'
    # plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    # # Show the minor grid lines with very faint and almost transparent grey lines
    # plt.minorticks_on()
    # plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    ldense = plt.axhline(y=dense, color=dense_color, linestyle='--', linewidth=3, label="Dense Model")

    lhydra = plt.plot(x_grid, y_hydra, color=hydra_color, marker='s', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Hydra", alpha=hydra_alpha)
    plt.fill_between(x_grid, y_hydra - y_hydra_err, y_hydra + y_hydra_err, color=hydra_color, alpha=fill_in_alpha)

    lIMP = plt.plot(x_grid, y_IMP, color=IMP_color, marker='s', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="IMP", alpha=IMP_alpha)
    plt.fill_between(x_grid, y_IMP - y_IMP_err, y_IMP + y_IMP_err, color=IMP_color, alpha=fill_in_alpha)

    lOMP = plt.plot(x_grid, y_OMP, color=OMP_color, marker='v', markevery=markevery, linestyle='-', linewidth=linewidth,
                    markersize=markersize, label="OMP", alpha=OMP_alpha)
    plt.fill_between(x_grid, y_OMP - y_OMP_err, y_OMP + y_OMP_err, color=OMP_color, alpha=fill_in_alpha)

    lGrasp = plt.plot(x_grid, y_Grasp, color=Grasp_color, marker='o', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Grasp", alpha=Grasp_alpha)
    plt.fill_between(x_grid, y_Grasp - y_grasp_err, y_Grasp + y_grasp_err, color=Grasp_color, alpha=fill_in_alpha)

    lBiP = plt.plot(x_grid, y_BiP, color=BiP_color, marker='*', markevery=markevery, linestyle='-',
                    linewidth=linewidth,
                    markersize=markersize + 4, label="BiP", alpha=BiP_alpha)
    plt.fill_between(x_grid, y_BiP - y_BiP_err, y_BiP + y_BiP_err, color=BiP_color, alpha=fill_in_alpha)

    lbest = plt.axhline(y=best, color=best_color, linestyle='--', linewidth=3, alpha=best_alpha,
                        label="Best Winning Ticket")
    # lPFTT = plt.plot(x_grid, y_PFTT, color=PFTT_color, marker='*', markevery=markevery, linestyle='-',
    #                  linewidth=linewidth,
    #                  markersize=markersize + 4, label="PFTT", alpha=PFTT_alpha)
    # plt.fill_between(x_grid, y_PFTT - y_PFTT_err, y_PFTT + y_PFTT_err, color=PFTT_color, alpha=fill_in_alpha)

    plt.ylim([y_min, y_max])
    plt.xlim(0, 100)

    # plt.legend(fontsize=fontsize - 16, loc=3, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(x_grid, x_density_list, rotation=0, fontsize=fontsize)
    plt.xscale("linear")
    plt.yticks(fontsize=fontsize)

    plt.title("CIFAR-10, VGG-16", fontsize=fontsize)
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
    plt.savefig("graphs/Global_Pruning_on_CIFAR-10_VGG16.pdf")
    plt.show()
    plt.close()










