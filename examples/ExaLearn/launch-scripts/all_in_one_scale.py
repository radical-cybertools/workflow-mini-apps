import os, sys, subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import radical.utils as ru
import radical.pilot as rp
import radical.analytics as ra

def plot_horizontal_line(x_start, x_end, y_value, color):
    plt.hlines(y=y_value, xmin=x_start, xmax=x_end, color=color)

def plot_multiple_horizontal_line(x_start_list, x_end_list, read_list, write_list):
    for i in range(len(x_start_list)):
        plot_horizontal_line(x_start_list[i], x_end_list[i], read_list[i], 'red')
        plot_horizontal_line(x_start_list[i], x_end_list[i], write_list[i], 'blue')

def make_plot(title, x_start_list, x_end_list, read_list, write_list, filename):
    plt.figure(figsize=(8, 4))
    plt.xlabel('Time')
    plt.ylabel('IO read/write in GB')
    plt.title(title)

    plot_multiple_horizontal_line(x_start_list, x_end_list, read_list, write_list)

    plt.grid(True)
    legend_labels = ['write', 'read']
    legend_handles = [plt.Line2D([0], [0], color='blue', label='write'),
                   plt.Line2D([0], [0], color='red', label='read')]
    plt.legend(handles=legend_handles, labels=legend_labels)

    plt.savefig(filename)

log_dict = {
        "ss_n_4":   "re.session.polaris-login-01.twang3.019583.0002",
        "ss_n_8":   "re.session.polaris-login-01.twang3.019583.0003",
        "ss_n_16":  "re.session.polaris-login-01.twang3.019583.0006",
        "ss_n_32":  "re.session.polaris-login-04.twang3.019584.0005"
}

state_exec_start = {ru.EVENT: 'exec_start'}
state_exec_stop = {ru.EVENT: 'exec_stop'}

ra_root = "/eagle/RECUP/twang/miniapp/exalearn-original/workflow-mini-apps/ExaLearn/launch-scripts/"

for ss_n in [4, 8, 16, 32]:
    setup = "ss_n_{}".format(ss_n)
    print("setup = ", setup)
    rct_sandbox = log_dict[setup]
    sp = ra_root + rct_sandbox
    session = ra.Session(sp, 'radical.pilot')
    pilots  = session.filter(etype='pilot', inplace=False)
    tasks   = session.filter(etype='task' , inplace=False)
    
    darshan_dir = "/eagle/RECUP/twang/miniapp/exalearn-original/workflow-mini-apps/ExaLearn/launch-scripts/darshan_log/" + setup + '/'

    task_all_start = []
    task_all_stop = []
    task_all_dur = []
    task_all_read = []
    task_all_write = []
    for task_id, task in enumerate(tasks.get()):
        ts_start = task.timestamps(event=state_exec_start)[0]
        ts_stop = task.timestamps(event=state_exec_stop)[0]
        if task_id == 0:
            ts_start_bias = ts_start
        ts_start = ts_start - ts_start_bias
        ts_stop = ts_stop - ts_start_bias

        task_all_start.append(ts_start)
        task_all_stop.append(ts_stop)
        task_all_dur.append(ts_stop - ts_start)
#        print(setup, "  task_{}".format(task_id), "   start = {}, stop = {}, dur = {}".format(ts_start, ts_stop, ts_stop - ts_start))
        for io_mode in ['w', 'r']:
            if task_id % 2 == 0:
                result = subprocess.run(["./get_io_bytes.sh {} {} {} {} {} | tail -n 1".format(rct_sandbox, darshan_dir, "sim", task_id // 2, io_mode)], shell=True, capture_output=True)
                output = int(result.stdout.strip().decode().split()[3]) / 1024.0 / 1024.0 / 1024.0
            if task_id % 2 == 1:
                result = subprocess.run(["./get_io_bytes.sh {} {} {} {} {} | tail -n 1".format(rct_sandbox, darshan_dir, "train", task_id // 2, io_mode)], shell=True, capture_output=True)
                output = int(result.stdout.strip().decode().split()[3]) / 1024.0 / 1024.0 / 1024.0
            if io_mode == 'w':
                task_all_write.append(output)
#                print(io_mode, "   ", output, "   ", end="")
            else:
                task_all_read.append(output)
#                print(io_mode, "   ", output, "   ")
    print(task_all_start)
    print(task_all_stop)
    print(task_all_dur)
    print(task_all_read)
    print("total read = ", sum(task_all_read))
    print(task_all_write)
    print("total write = ", sum(task_all_write))
#    make_plot(setup, task_all_start, task_all_stop, task_all_read, task_all_write, setup+'png')



