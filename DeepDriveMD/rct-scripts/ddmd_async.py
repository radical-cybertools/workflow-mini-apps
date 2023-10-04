#!/usr/bin/env python3

# Changing the ratio among types of tasks at runtime.
# Algorithmically:
# - Start MD simulation tasks, use all the available resources
# - upon termination of an MD sim task:
#   - if the aggregation threshold is reached, kill a sim task and
#     launch an Aggregation task
#   - else, launch a new sim task
# - upon termination of an Aggregation task, launch a ML training task (possibly
#   killing some of the sim tasks if it requires more resource)
# - upon termination of an ML training task:
#   - if learning threshold is reached, launch an Agent task;
#   - else, launch a sim task
# - Upon termination of an Agent task, kill all the tasks and goto i.


import os
import time
import random
import signal
import threading as mt
import argparse, sys
import json

import radical.pilot as rp
import radical.utils as ru


# ------------------------------------------------------------------------------
#
class DDMD(object):

    # define task types (used as prefix on task-uid)
    TASK_MD_SIM    = 'md_sim'
    TASK_ML_TRAIN  = 'ml_train'
    TASK_SELECTION = 'selection'
    TASK_AGENT     = 'agent'
#    DEFAULT        = 'task'

    TASK_TYPES     = [TASK_MD_SIM, TASK_ML_TRAIN,TASK_SELECTION, TASK_AGENT]
#    TASK_TYPES      = [DEFAULT]

    # keep track of core usage
    cores_used     = 0
    gpus_used     = 0

    # --------------------------------------------------------------------------
    #
    def __init__(self):

        self.env_work_dir = os.getenv("MINI_APP_DeepDriveMD_DIR")
        if self.env_work_dir is None:
            print("Warning: Did not set up work_dir using env var, need to set it up in parser manually!")
        self.set_argparse()
        self.get_json()

        # control flow table
        self._protocol = {self.TASK_MD_SIM   : self._control_md_sim,
                          self.TASK_ML_TRAIN : self._control_ml_train,
                          self.TASK_SELECTION: self._control_selection,
                          self.TASK_AGENT    : self._control_agent}

        self._glyphs   = {self.TASK_MD_SIM   : '#',
                          self.TASK_SELECTION: '+',
                          self.TASK_ML_TRAIN : '=',
                          self.TASK_AGENT    : '*'}

        # bookkeeping
        self._selection     =  0
        self._selection_max =  1  # aggregation threshold

        self._trained        =  0
        self._trained_max    =  1  # training threshold

        self._mdSim         = 0
        self._mdSim_max     = self.args.num_sim

        self._agent         = 0
        self._agent_max     = 1

        self._cores          = 32 * self.args.num_nodes  # available resources
        self._cores_used     =  0

        self._gpus           = 4 * self.args.num_nodes  #available Gpu resources
        self._gpus_used      = 0

        self._lock           = mt.RLock()
        self._tasks          = {ttype: dict() for ttype in self.TASK_TYPES}
        self._final_tasks    = list()

        # silence RP reporter, use own
        os.environ['RADICAL_REPORT'] = 'false'
        self._rep = ru.Reporter('ddmd')
        self._rep.title('DDMD')

        # RP setup
        self._session = rp.Session()
        self._pmgr    = rp.PilotManager(session=self._session)
        self._tmgr    = rp.TaskManager(session=self._session)

        pdesc = rp.PilotDescription({
           'resource': 'anl.polaris',
#           'queue'   : 'debug',
            'queue'   : self.args.queue,
#           'queue'   : 'default',
            'runtime': 30, #MIN
            'cores'    : 32 * self.args.num_nodes,
            'gpus'    : 4 * self.args.num_nodes,
            'project' : self.args.project_id})
        print(pdesc)

        self._pilot = self._pmgr.submit_pilots(pdesc)

        self._tmgr.add_pilots(self._pilot)
        self._tmgr.register_callback(self._checked_state_cb)


    def set_argparse(self):
        parser = argparse.ArgumentParser(description="DeepDriveMD_miniapp_EnTK_async")

        parser.add_argument('--num_phases', type=int, default=3,
                        help='number of phases in the workflow')
        parser.add_argument('--mat_size', type=int, default=5000,
                        help='the matrix with have size of mat_size * mat_size')
        parser.add_argument('--data_root_dir', default='./',
                        help='the root dir of gsas output data')
        parser.add_argument('--num_step', type=int, default=1000,
                        help='number of step in MD simulation')
        parser.add_argument('--num_epochs_train', type=int, default=150,
                        help='number of epochs in training task')
        parser.add_argument('--model_dir', default='./',
                        help='the directory where save and load model')
        parser.add_argument('--num_sample', type=int, default=500,
                        help='num of samples in matrix mult (training and agent)')
        parser.add_argument('--num_mult_train', type=int, default=4000,
                        help='number of matrix mult to perform in training task')
        parser.add_argument('--dense_dim_in', type=int, default=12544,
                        help='dim for most heavy dense layer, input')
        parser.add_argument('--dense_dim_out', type=int, default=128,
                        help='dim for most heavy dense layer, output')
        parser.add_argument('--preprocess_time_train', type=float, default=20.0,
                        help='time for doing preprocess in training')
        parser.add_argument('--preprocess_time_agent', type=float, default=10.0,
                        help='time for doing preprocess in agent')
        parser.add_argument('--num_epochs_agent', type=int, default=10,
                        help='number of epochs in agent task')
        parser.add_argument('--num_mult_agent', type=int, default=4000,
                        help='number of matrix mult to perform in agent task, inference')
        parser.add_argument('--num_mult_outlier', type=int, default=10,
                        help='number of matrix mult to perform in agent task, outlier')

        parser.add_argument('--project_id', required=True,
                        help='the project ID we used to launch the job')
        parser.add_argument('--queue', required=True,
                        help='the queue we used to submit the job')
        parser.add_argument('--work_dir', default=self.env_work_dir,
                        help='working dir, which is the dir of this repo')
        parser.add_argument('--num_sim', type=int, default=12,
                        help='number of tasks used for simulation')
        parser.add_argument('--num_nodes', type=int, default=3,
                        help='number of nodes used for simulation')
        parser.add_argument('--io_json_file', default="io_size.json",
                        help='the filename of json file for io size')

        args = parser.parse_args()
        self.args = args

    def get_json(self):
        json_file = "{}/launch-scripts/{}".format(self.args.work_dir, self.args.io_json_file)
        with open(json_file) as f:
            self.io_dict = json.load(f)

    # --------------------------------------------------------------------------
    #
    def __del__(self):

        self.close()


    # --------------------------------------------------------------------------
    #
    def close(self):

        if self._session is not None:
            self._session.close()
            self._session = None


    # --------------------------------------------------------------------------
    #
    def dump(self, task=None, msg=''):
        '''
        dump a representation of current task set to stdout
        '''

        # this assumes one core per task

        self._rep.plain('<<|')

        idle = self._cores

        n     = len(self._tasks[self.TASK_MD_SIM])
        idle -= n
        self._rep.ok('%s' % self._glyphs[self.TASK_MD_SIM] * n)

        n     = len(self._tasks[self.TASK_SELECTION])
        idle -= n
        self._rep.warn('%s' % self._glyphs[self.TASK_SELECTION] * n)

        n     = len(self._tasks[self.TASK_ML_TRAIN])
        idle -= n
        self._rep.error('%s' % self._glyphs[self.TASK_ML_TRAIN] * n)

        n     = len(self._tasks[self.TASK_AGENT])
        idle -= n
        self._rep.info('%s' % self._glyphs[self.TASK_AGENT] * n)

        self._rep.plain('%s' % '-' * idle +
                        '| %4d [%4d]' % (self._cores_used, self._cores))

        if task and msg:
            self._rep.plain(' %-15s: %s\n' % (task.uid, msg))
        else:
            if task:
                msg = task
            self._rep.plain(' %-15s: %s\n' % (' ', msg))





    # --------------------------------------------------------------------------
    #
    def start(self):
        '''
        submit initial set of MD similation tasks
        '''

        self.dump('submit MD simulations')

        # reset bookkeeping
        self._iteration = 0
        self._selection = 0
        self._trained   = 0
        self._mdSim     = 0
        self._agent     = 0
        self._cores_used = 0
        self._tasks      = {ttype: dict() for ttype in self._protocol}


        self._iteration = 1
        self.run_sim(self.TASK_MD_SIM, n=self.args.num_sim)


#        # run initial batch of MD_SIM tasks (assume one core per task)
#        self._submit_task(self.TASK_MD_SIM, n=self._cores, g=self._gpus)
#
#        self.dump('started %s md sims' % self._cores)



    # --------------------------------------------------------------------------
    #
    def stop(self):

        os.kill(os.getpid(), signal.SIGKILL)
        os.kill(os.getpid(), signal.SIGTERM)


    # --------------------------------------------------------------------------
    #
    def _get_ttype(self, uid):
        '''
        get task type from task uid
        '''

        ttype = uid.split('.')[0]

        assert ttype in self.TASK_TYPES, 'unknown task type: %s' % uid
        return ttype


    # --------------------------------------------------------------------------
    #
    def _submit_task(self, ttype, n=1, g=0):
        '''
        submit 'n' new tasks of specified type

        NOTE: all tasks are uniform for now: they use a single core and sleep
              for a random number (0..3) of seconds.
        '''

        with self._lock:

            tds   = list()
            for _ in range(n):
                tds.append(rp.TaskDescription({
                         'uid'          : ru.generate_id(ttype),
                         'cpu_processes': 1,
                         'executable'   : '/bin/sh',
                         'arguments'    : ['-c', 'sleep %s; echo %s' %
                             (int(random.randint(0,30) / 10),
                              int(random.randint(0,10) /  1))]}))

            tasks  = self._tmgr.submit_tasks(tds)

            for task in tasks:
                self._register_task(task)


    # --------------------------------------------------------------------------
    #
    def _cancel_tasks(self, uids):
        '''
        cancel tasks with the given uids, and unregister them
        '''

        uids = ru.as_list(uids)

        # FIXME: does not work
        self._tmgr.cancel_tasks(uids)

        for uid in uids:
            ttype = self._get_ttype(uid)
            task  = self._tasks[ttype][uid]
            self.dump(task, 'cancel [%s]' % task.state)

            self._unregister_task(task)


    # --------------------------------------------------------------------------
    #
    def _register_task(self, task):
        '''
        add task to bookkeeping
        '''

        with self._lock:
            ttype = self._get_ttype(task.uid)
            self._tasks[ttype][task.uid] = task

            cores = task.description['cpu_processes'] \
                  * task.description['cpu_threads']
            self._cores_used += cores


    # --------------------------------------------------------------------------
    #
    def _unregister_task(self, task):
        '''
        remove completed task from bookkeeping
        '''

        with self._lock:

            ttype = self._get_ttype(task.uid)

            if task.uid not in self._tasks[ttype]:
                return

            # removed tasks dont consume cores
            cores = task.description['cpu_processes'] \
                  * task.description['cpu_threads']
            self._cores_used -= cores

            # remove task from bookkeeping
            self._final_tasks.append(task.uid)
            del self._tasks[ttype][task.uid]


    # --------------------------------------------------------------------------
    #
    def _state_cb(self, task, state):
        '''
        act on task state changes according to our protocol
        '''

        try:
            return self._checked_state_cb(task, state)
        except Exception as e:
            self._rep.error('\n\n---------\nexception caught: %s\n\n' % repr(e))
            self.stop()


    # --------------------------------------------------------------------------
    #
    def _checked_state_cb(self, task, state):

        # this cb will react on task state changes.  Specifically it will watch
        # out for task completion notification and react on them, depending on
        # the task type.

      # if state in [rp.TMGR_SCHEDULING] + rp.FINAL:
      #     self.dump(task, ' -> %s' % task.state)

        # ignore all non-final state transitions
        if state not in rp.FINAL:
            return

        # ignore tasks which were already
        if task.uid in self._final_tasks:
            return

        # lock bookkeeping
        with self._lock:

            # raise alarm on failing tasks (but continue anyway)
            if state == rp.FAILED:
                self._rep.error('task %s failed: %s' % (task.uid, task.stderr))
                self.stop()

            # control flow depends on ttype
            ttype  = self._get_ttype(task.uid)
            action = self._protocol[ttype]
            if not action:
                self._rep.exit('no action found for task %s' % task.uid)
            action(task)

            # remove final task from bookkeeping
            self._unregister_task(task)


    # --------------------------------------------------------------------------
    #
    def _control_md_sim(self, task):
        '''
        react on completed MD simulation task
        '''

        # - upon termination of an MD sim task:
        #   - if the ML threshold is reached,
        #     - launch an ML task
        #   - else
        #     - dont do anything
        self._mdSim += 1

        if self._mdSim >= self._mdSim_max:
            self._mdSim = 0
            self.dump(task, 'completed, start ML and next Sim')
            self.run_train(self.TASK_ML_TRAIN, 1)
            if iteration < self.arg.num_phases:
                iteration += 1
                self.run_sim(self.TASK_MD_SIM, n=self.args.num_sim)
        else:
            self.dump(task, 'completed, aggregation low  - start md sim')


    # --------------------------------------------------------------------------
    #
    def _control_selection(self, task):


        self._selection += 1
        if self._selection >= self._selection_max:
            self.dump(task, 'completed, Selection - start agent ')
            self._selection= 0
            self.run_agent(self.TASK_AGENT, n=1)
        else:
            self.dump(task, 'completed, Selection incomplete  ')




    # --------------------------------------------------------------------------
    #
    def _control_ml_train(self, task):
        '''
        react on completed ML training task
        '''
        # - upon termination of an ML training task:
        #   - if learning threshold is reached
        #     - launch an Agent task;
        #   - else
        #     - launch a sim task

        self._trained += 1
        if self._trained >= self._trained_max:
            self.dump(task, 'completed, training complete - start agent ')
            self._trained = 0
            self.run_selection(self.TASK_SELECTION, n=1)
        else:
            self.dump(task, 'completed, training incomplete  ')


    # --------------------------------------------------------------------------
    #
    def _control_agent(self, task):
        '''
        react on completed agent task
        '''
        # - Upon termination of an Agent task, kill all the tasks and goto i.
        try:
            self._agent += 1
        except:
            pass

        if self._agent >= self._agent_max:
            self._agent= 0
            self.dump(task, 'completed,  next Sim')
            if iteration < self.arg.num_phases:
                iteration += 1
                self.run_sim(self.TASK_MD_SIM, n=self.args.num_sim)
        else:
            self.dump(task, 'completed, aggregation low  - start md sim')



    def run_sim(self, ttype , n=1):

        with self._lock:
            tds   = list()
            for i in range(n):
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load PrgEnv-gnu","module load conda","conda activate /grand/CSC249ADCD08/twang/env/rct-recup-polaris","export HDF5_USE_FILE_LOCKING=FALSE"],
                         'uid'          : ru.generate_id(ttype),
                         'cpu_processes': 1,
                         'cpu_process_type' : None,
                         'cpu_threads'      : 8,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 1,
                         'gpu_process_type'  : rp.CUDA,
                         'executable'   : 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python',
                         'arguments'    : ['{}/Executables/simulation.py'.format(self.args.work_dir),
                               '--phase={}'.format(self._iteration-1),
                               '--task_idx={}'.format(i),
                               '--mat_size={}'.format(self.args.mat_size),
                               '--data_root_dir={}'.format(self.args.data_root_dir),
                               '--num_step={}'.format(self.args.num_step),
                               '--write_size={}'.format(self.io_dict["phase{}".format(self._iteration-1)]["sim"]["write"]),
                               '--read_size={}'.format(self.io_dict["phase{}".format(self._iteration-1)]["sim"]["read"])]}))

            tasks  = self._tmgr.submit_tasks(tds)

            for task in tasks:
                self._register_task(task)


    # This is for training, return a stage which has a single training task
    def run_train(self, n=1):

        with self._lock:
            tds   = list()
            for _ in range(n):
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load PrgEnv-gnu","module load conda","conda activate /grand/CSC249ADCD08/twang/env/rct-recup-polaris","export HDF5_USE_FILE_LOCKING=FALSE"],
                         'uid'          : ru.generate_id(ttype),
                         'cpu_processes': 1,
                         'cpu_process_type' : None,
                         'cpu_threads'      : 8,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 1,
                         'gpu_process_type'  : rp.CUDA,
                         'executable'   : 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python',
                         'arguments'    : [ '{}/Executables/training.py'.format(self.args.work_dir),
                               '--num_epochs={}'.format(self.args.num_epochs_train),
                               '--device=gpu',
                               '--phase={}'.format(self._iteration-1),
                               '--data_root_dir={}'.format(self.args.data_root_dir),
                               '--model_dir={}'.format(self.args.model_dir),
                               '--num_sample={}'.format(self.args.num_sample * (1 if (self._iteration - 1) == 0 else 2)),
                               '--num_mult={}'.format(self.args.num_mult_train),
                               '--dense_dim_in={}'.format(self.args.dense_dim_in),
                               '--dense_dim_out={}'.format(self.args.dense_dim_out),
                               '--mat_size={}'.format(self.args.mat_size),
                               '--preprocess_time={}'.format(self.args.preprocess_time_train),
                               '--write_size={}'.format(self.io_dict["phase{}".format(self._iteration-1)]["train"]["write"]),
                               '--read_size={}'.format(self.io_dict["phase{}".format(self._iteration-1)]["train"]["read"])]}))

            tasks  = self._tmgr.submit_tasks(tds)

            for task in tasks:
                self._register_task(task)


    # This is for model selection, return a stage which has a single training task
    def run_selection(self, n=1):

        with self._lock:
            tds   = list()
            for _ in range(n):
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load PrgEnv-gnu","module load conda","conda activate /grand/CSC249ADCD08/twang/env/rct-recup-polaris","export HDF5_USE_FILE_LOCKING=FALSE"],
                         'uid'          : ru.generate_id(ttype),
                         'cpu_processes': 1,
                         'cpu_process_type' : None,
                         'cpu_threads'      : 8,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 0,
                         'gpu_process_type'  : None,
                         'executable'   : 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python',
                         'arguments'    : [ '{}/Executables/selection.py'.format(self.args.work_dir),
                       '--phase={}'.format(self._iteration-1),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--write_size={}'.format(self.io_dict["phase{}".format(self._iteration-1)]["selection"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(self._iteration-1)]["selection"]["read"])]}))

            tasks  = self._tmgr.submit_tasks(tds)

            for task in tasks:
                self._register_task(task)


    # This is for agent, return a stage which has a single training task
    def run_agent(self, n=1):

        with self._lock:
            tds   = list()
            for _ in range(n):
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load PrgEnv-gnu","module load conda","conda activate /grand/CSC249ADCD08/twang/env/rct-recup-polaris","export HDF5_USE_FILE_LOCKING=FALSE"],
                         'uid'          : ru.generate_id(ttype),
                         'cpu_processes': 1,
                         'cpu_process_type' : None,
                         'cpu_threads'      : self.args.num_sim,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 1,
                         'gpu_process_type'  : rp.CUDA,
                         'executable'   : 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python',
                         'arguments'    : [ '{}/Executables/agent.py'.format(self.args.work_dir),
                       '--num_epochs={}'.format(self.args.num_epochs_agent),
                       '--device=gpu',
                       '--phase={}'.format(self._iteration-1),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--num_sample={}'.format(self.args.num_sample),
                       '--num_mult={}'.format(self.args.num_mult_agent),
                       '--num_mult_outlier={}'.format(self.args.num_mult_outlier),
                       '--dense_dim_in={}'.format(self.args.dense_dim_in),
                       '--dense_dim_out={}'.format(self.args.dense_dim_out),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--preprocess_time={}'.format(self.args.preprocess_time_agent),
                       '--write_size={}'.format(self.io_dict["phase{}".format(self._iteration-1)]["agent"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(self._iteration-1)]["agent"]["read"])]}))

            tasks  = self._tmgr.submit_tasks(tds)

            for task in tasks:
                self._register_task(task)



# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    ddmd = DDMD()

    try:
        ddmd.start()

        while True:
          # ddmd.dump()
            time.sleep(1)

    finally:
        ddmd.close()


# ------------------------------------------------------------------------------
