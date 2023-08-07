import tarfile

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import radical.utils as ru
import radical.pilot as rp
import radical.analytics as ra

import os, sys, shutil

plt.style.use(ra.get_mplstyle('radical_mpl'))

state_exec_start = {ru.EVENT: 'exec_start'}
state_exec_stop = {ru.EVENT: 'exec_stop'}

sp = "/eagle/RECUP/twang/miniapp/exalearn-original/real_work_polaris_cpu/code/re.session.polaris-login-01.twang3.019562.0003" 
session = ra.Session(sp, 'radical.pilot')
pilots  = session.filter(etype='pilot', inplace=False)
tasks   = session.filter(etype='task' , inplace=False)

tseries = {'exec_start': [],
           'exec_stop': []}

for task in tasks.get():
    ts_start = task.timestamps(event=state_exec_start)[0]
    ts_stop = task.timestamps(event=state_exec_stop)[0]
    tseries['exec_start'].append(ts_start)
    tseries['exec_stop'].append(ts_stop)

time_series = pd.DataFrame.from_dict(tseries)
print(time_series)
