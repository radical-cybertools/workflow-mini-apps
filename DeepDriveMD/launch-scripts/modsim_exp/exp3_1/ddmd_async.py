from radical import entk
import os
import argparse, sys, math
import radical.pilot as rp
import radical.utils as ru
import json
import math
from radical.entk.tools import (cache_darshan_env,
                                with_darshan,
                                enable_darshan,
                                get_provenance_graph)



class MVP(object):

    def __init__(self):
        self.env_work_dir = os.getenv("MINI_APP_DeepDriveMD_DIR")
        if self.env_work_dir is None:
            print("Warning: Did not set up work_dir using env var, need to set it up in parser manually!")
        self.set_argparse()
        self.get_json()
        self.am = entk.AppManager()

        if self.args.enable_darshan:
            cache_darshan_env(
                darshan_runtime_root='/home/twang3/libraries/darshan/',
                env={'PATH': "/home/twang3/libraries/darshan/bin:$PATH"}
                )


    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

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
        parser.add_argument('--conda_env', default=None,
                        help='the conda env where numpy/cupy installed, if not specified, no env will be loaded')
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
        parser.add_argument('--enable_darshan', action='store_true',
                        help='enable darshan analyze')
        parser.add_argument('--project_id', required=True,
                        help='the project ID we used to launch the job')
        parser.add_argument('--queue', required=True,
                        help='the queue we used to submit the job')
        parser.add_argument('--work_dir', default=self.env_work_dir,
                        help='working dir, which is the dir of this repo')
        parser.add_argument('--num_sim', type=int, default=12,
                        help='number of tasks used for simulation')
        parser.add_argument('--num_nodes', type=int, default=2,
                        help='number of nodes used for simulation')
        parser.add_argument('--io_json_file', default="io_size.json",
                        help='the filename of json file for io size')

        args = parser.parse_args()
        self.args = args

    def get_json(self):
        json_file = "{}/launch-scripts/{}".format(self.args.work_dir, self.args.io_json_file)
        with open(json_file) as f:
            self.io_dict = json.load(f)

    # This is for simulation, return a stage which has many sim task
    def run_sim(self, phase_idx):

        s = entk.Stage()
        for i in range(self.args.num_sim):
            t = entk.Task()
            t.pre_exec = [
                    "module use /soft/modulefiles",
                    "module load conda/2024-04-29",
                    "export HDF5_USE_FILE_LOCKING=FALSE"
                    ]
            if self.args.conda_env is not None:
                t.pre_exec.append("conda activate {}".format(self.args.conda_env))
            t.executable = 'python'
            t.arguments = ['{}/Executables/simulation.py'.format(self.args.work_dir),
                           '--phase={}'.format(phase_idx),
                           '--mat_size={}'.format(self.args.mat_size),
                           '--data_root_dir={}'.format(self.args.data_root_dir),
                           '--num_step={}'.format(self.args.num_step),
                           '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["sim"]["write"]),
                           '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["sim"]["read"]),
                           '--instance_index={}'.format(i)]
            t.post_exec = []
            t.cpu_reqs = {
                 'cpu_processes'    : 1,
                 'cpu_process_type' : None,
                 'cpu_threads'      : 8,
                 'cpu_thread_type'  : rp.OpenMP
                 }
            t.gpu_reqs = {
                 'gpu_processes'     : 1,
                 'gpu_process_type'  : rp.CUDA
                 }

            if self.args.enable_darshan:
                s.add_tasks(enable_darshan(t))
            else:
                s.add_tasks(t)

        return s


    # This is for training, return a training task
    def run_train(self, phase_idx):

        t = entk.Task()
        t.pre_exec = [
                "module use /soft/modulefiles",
                'module load conda/2024-04-29',
                "export HDF5_USE_FILE_LOCKING=FALSE"
                ]
        if self.args.conda_env is not None:
            t.pre_exec.append("conda activate {}".format(self.args.conda_env))

        t.executable = 'python'
        t.arguments = ['{}/Executables/training.py'.format(self.args.work_dir),
                       '--num_epochs={}'.format(self.args.num_epochs_train),
                       '--device=gpu',
                       '--phase={}'.format(phase_idx),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--num_sample={}'.format(self.args.num_sample * (1 if phase_idx == 0 else 2)),
                       '--num_mult={}'.format(self.args.num_mult_train),
                       '--dense_dim_in={}'.format(self.args.dense_dim_in),
                       '--dense_dim_out={}'.format(self.args.dense_dim_out),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--preprocess_time={}'.format(self.args.preprocess_time_train),
                       '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["train"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["train"]["read"]),
                       '--instance_index={}'.format(0)]
        t.post_exec = []
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 8,
            'cpu_thread_type'   : rp.OpenMP
                }
        t.gpu_reqs = {
            'gpu_processes'     : 1,
            'gpu_process_type'  : rp.CUDA
                }

        if self.args.enable_darshan:
            return enable_darshan(t)
        else:
            return t

    # This is for model selection, return a stage which has a single training task
    def run_selection(self, phase_idx):

        t = entk.Task()
        t.pre_exec = [
                "module use /soft/modulefiles",
                'module load conda/2024-04-29',
                "export HDF5_USE_FILE_LOCKING=FALSE"
                ]
        if self.args.conda_env is not None:
            t.pre_exec.append("conda activate {}".format(self.args.conda_env))

        t.executable = 'python'
        t.arguments = ['{}/Executables/selection.py'.format(self.args.work_dir),
                       '--phase={}'.format(phase_idx),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["selection"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["selection"]["read"]),
                       '--instance_index={}'.format(0)]
        t.post_exec = []
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 8,
            'cpu_thread_type'   : rp.OpenMP
                }

        if self.args.enable_darshan:
            return enable_darshan(t)
        else:
            return t

    # This is for agent, return a stage which has a single training task
    def run_agent(self, phase_idx):

        t = entk.Task()
        t.pre_exec = [
                "module use /soft/modulefiles",
                'module load conda/2024-04-29',
                "export HDF5_USE_FILE_LOCKING=FALSE"
                ]
        if self.args.conda_env is not None:
            t.pre_exec.append("conda activate {}".format(self.args.conda_env))

        t.executable = 'python'
        t.arguments = ['{}/Executables/agent.py'.format(self.args.work_dir),
                       '--num_epochs={}'.format(self.args.num_epochs_agent),
                       '--device=gpu',
                       '--phase={}'.format(phase_idx),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--num_sample={}'.format(self.args.num_sample),
                       '--num_mult={}'.format(self.args.num_mult_agent),
                       '--num_mult_outlier={}'.format(self.args.num_mult_outlier),
                       '--dense_dim_in={}'.format(self.args.dense_dim_in),
                       '--dense_dim_out={}'.format(self.args.dense_dim_out),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--preprocess_time={}'.format(self.args.preprocess_time_agent),
                       '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["agent"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["agent"]["read"]),
                       '--instance_index={}'.format(0)]
        t.post_exec = []
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : self.args.num_sim,
            'cpu_thread_type'   : rp.OpenMP
                }
        t.gpu_reqs = {
            'gpu_processes'     : 1,
            'gpu_process_type'  : rp.CUDA
                }

        if self.args.enable_darshan:
            return enable_darshan(t)
        else:
            return t

    def generate_pipeline(self):

        p = entk.Pipeline()
        
        s1 = self.run_sim(0)
        s2 = self.run_sim(1)
        s2.add_tasks(self.run_train(0))
        s3 = entk.Stage()
        s3.add_tasks(self.run_selection(0))
        s4 = entk.Stage()
        s4.add_tasks(self.run_agent(0))
        s5 = self.run_sim(2)
        s5.add_tasks(self.run_train(1))
        s6 = entk.Stage()
        s6.add_tasks(self.run_selection(1))
        s7 = entk.Stage()
        s7.add_tasks(self.run_agent(1))
        s8 = entk.Stage()
        s8.add_tasks(self.run_train(2))
        s9 = entk.Stage()
        s9.add_tasks(self.run_selection(2))
        s10 = entk.Stage()
        s10.add_tasks(self.run_agent(2))

        p.add_stages(s1)
        p.add_stages(s2)
        p.add_stages(s3)
        p.add_stages(s4)
        p.add_stages(s5)
        p.add_stages(s6)
        p.add_stages(s7)
        p.add_stages(s8)
        p.add_stages(s9)
        p.add_stages(s10)
        return p

    def run_workflow(self):
        p = self.generate_pipeline()
        self.am.workflow = [p]
        self.am.run()


if __name__ == "__main__":

    mvp = MVP()
    mvp.set_resource(res_desc = {
        'resource': 'anl.polaris',
#        'queue'   : 'debug',
        'queue'   : mvp.args.queue,
#        'queue'   : 'default',
        'walltime': 45, #MIN
        'cpus'    : 32 * mvp.args.num_nodes,
        'gpus'    : 4 * mvp.args.num_nodes,
        'project' : mvp.args.project_id
        })
    mvp.run_workflow()
