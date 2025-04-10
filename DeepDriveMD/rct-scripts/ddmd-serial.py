#!/usr/bin/env python

import os
import json
import argparse
import math

from radical import entk
import radical.pilot as rp
import radical.utils as ru

class MVP(object):

    def __init__(self):
        self.env_work_dir = os.getenv("MINI_APP_DeepDriveMD_DIR")
        if self.env_work_dir is None:
            print("Warning: Did not set up work_dir using env var, need to set it up in parser manually!")
        self.parse_args()
        self.load_config()
        self.get_io_dict()
        self.am = entk.AppManager()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="DeepDriveMD_miniapp_EnTK_serial")
        parser.add_argument('--config', default="config.json",
                            help='Path to the system configuration file (JSON format)')
        # Common arguments
        parser.add_argument('--num_phases', type=int, default=3,
                            help='number of phases in the workflow')
        parser.add_argument('--mat_size', type=int, default=5000,
                            help='the matrix will have size mat_size * mat_size')
        parser.add_argument('--data_root_dir', default='./',
                            help='the root directory of gsas output data')
        parser.add_argument('--num_step', type=int, default=1000,
                            help='number of steps in MD simulation')
        parser.add_argument('--num_epochs_train', type=int, default=150,
                            help='number of epochs in training task')
        parser.add_argument('--model_dir', default='./',
                            help='the directory where models are saved/loaded')
        parser.add_argument('--conda_env', default=None,
                            help='if specified, a conda env to activate (if needed)')
        parser.add_argument('--num_sample', type=int, default=500,
                            help='number of samples in matrix multiplication (for training and agent)')
        parser.add_argument('--num_mult_train', type=int, default=4000,
                            help='number of matrix multiplications for training task')
        parser.add_argument('--dense_dim_in', type=int, default=12544,
                            help='input dimension for the heavy dense layer')
        parser.add_argument('--dense_dim_out', type=int, default=128,
                            help='output dimension for the heavy dense layer')
        parser.add_argument('--preprocess_time_train', type=float, default=20.0,
                            help='preprocess time for training')
        parser.add_argument('--preprocess_time_agent', type=float, default=10.0,
                            help='preprocess time for agent')
        parser.add_argument('--num_epochs_agent', type=int, default=10,
                            help='number of epochs in the agent task')
        parser.add_argument('--num_mult_agent', type=int, default=4000,
                            help='number of matrix mults in agent (inference)')
        parser.add_argument('--num_mult_outlier', type=int, default=10,
                            help='number of matrix mults in agent (outlier detection)')
        parser.add_argument('--enable_darshan', action='store_true',
                            help='enable darshan profiling if supported by config')
        parser.add_argument('--project_id', required=True,
                            help='project ID used to launch the job')
        parser.add_argument('--queue', required=True,
                            help='queue used to submit the job')
        parser.add_argument('--work_dir', default=self.env_work_dir,
                            help='working directory (root of the repo)')
        parser.add_argument('--num_sim', type=int, default=12,
                            help='number of simulation tasks per phase')
        parser.add_argument('--num_nodes', type=int, default=2,
                            help='number of nodes used for simulation')
        parser.add_argument('--io_json_file', default="io_size.json",
                            help='filename of JSON file for IO sizes')
        self.args = parser.parse_args()

    def load_config(self):
        with open(self.args.config, 'r') as f:
            self.config = json.load(f)

    def get_io_dict(self):
        json_file = os.path.join(self.args.work_dir, "launch-scripts", self.args.io_json_file)
        with open(json_file, 'r') as f:
            self.io_dict = json.load(f)

    def get_pre_exec(self, stage):
        stage_conf = self.config.get("stages", {}).get(stage, {})
        pre_exec = list(stage_conf["pre_exec"])
        if self.args.conda_env:
            pre_exec.append("conda activate {}".format(self.args.conda_env))
        return pre_exec

    def get_executable(self, stage):
        stage_conf = self.config.get("stages", {}).get(stage, {})
        if self.args.enable_darshan:
            if "executable_with_darshan" not in stage_conf:
                raise ValueError(f"Error: 'executable_with_darshan' not found for stage '{stage}' in the configuration!")
            return stage_conf["executable_with_darshan"]
        else:
            if "executable" not in stage_conf:
                raise ValueError(f"Error: 'executable' not found for stage '{stage}' in the configuration!")
            return stage_conf["executable"]

    def get_cpu_reqs(self, stage):
        stage_conf = self.config.get("stages", {}).get(stage, {})
        threads = stage_conf.get("cpu_threads", 1)
        if stage == "agent":
            if threads != self.args.num_sim:
                raise ValueError(
                    "For the agent stage, the 'cpu_threads' value ({}) must be equal to "
                    "num_sim ({}) as specified on the command-line.".format(threads, self.args.num_sim)
                )
        
        return {
            'cpu_processes': stage_conf.get("cpu_processes", 1),
            'cpu_process_type': None,
            'cpu_threads': threads,
            'cpu_thread_type': rp.OpenMP
        }

    def get_gpu_reqs(self, stage):
        stage_conf = self.config.get("stages", {}).get(stage, {})
        gpu_processes = stage_conf.get("gpu_processes")
        if gpu_processes is not None:
            reqs = {'gpu_processes': gpu_processes}
            # Only include gpu_process_type if it is provided in the config.
            gpu_type = stage_conf.get("gpu_process_type")
            if gpu_type is not None:
                reqs['gpu_process_type'] = gpu_type
            return reqs
        else:
            return {}

    def run_sim(self, phase_idx):
        s = entk.Stage()
        for i in range(self.args.num_sim):
            t = entk.Task()
            t.pre_exec = self.get_pre_exec("sim")
            t.executable = self.get_executable("sim")
            t.arguments = [
                os.path.join(self.args.work_dir, "Executables", "simulation.py"),
                '--phase={}'.format(phase_idx),
                '--task_idx={}'.format(i),
                '--mat_size={}'.format(self.args.mat_size),
                '--data_root_dir={}'.format(self.args.data_root_dir),
                '--num_step={}'.format(self.args.num_step),
                '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["sim"]["write"]),
                '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["sim"]["read"]),
                '--instance_index={}'.format(i)
            ]
            t.post_exec = []
            t.cpu_reqs = self.get_cpu_reqs("sim")
            t.gpu_reqs = self.get_gpu_reqs("sim")
            s.add_tasks(t)
        return s

    def run_train(self, phase_idx):
        s = entk.Stage()
        t = entk.Task()
        t.pre_exec = self.get_pre_exec("train")
        t.executable = self.get_executable("train")
        t.arguments = [
            os.path.join(self.args.work_dir, "Executables", "training.py"),
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
            '--instance_index={}'.format(0)
        ]
        t.post_exec = []
        t.cpu_reqs = self.get_cpu_reqs("train")
        t.gpu_reqs = self.get_gpu_reqs("train")
        s.add_tasks(t)
        return s

    def run_selection(self, phase_idx):
        s = entk.Stage()
        t = entk.Task()
        t.pre_exec = self.get_pre_exec("selection")
        t.executable = self.get_executable("selection")
        t.arguments = [
            os.path.join(self.args.work_dir, "Executables", "selection.py"),
            '--phase={}'.format(phase_idx),
            '--mat_size={}'.format(self.args.mat_size),
            '--data_root_dir={}'.format(self.args.data_root_dir),
            '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["selection"]["write"]),
            '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["selection"]["read"]),
            '--instance_index={}'.format(0)
        ]
        t.post_exec = []
        t.cpu_reqs = self.get_cpu_reqs("selection")
        t.gpu_reqs = self.get_gpu_reqs("selection")
        s.add_tasks(t)
        return s

    def run_agent(self, phase_idx):
        s = entk.Stage()
        t = entk.Task()
        t.pre_exec = self.get_pre_exec("agent")
        t.executable = self.get_executable("agent")
        t.arguments = [
            os.path.join(self.args.work_dir, "Executables", "agent.py"),
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
            '--instance_index={}'.format(0)
        ]
        t.post_exec = []
        t.cpu_reqs = self.get_cpu_reqs("agent")
        t.gpu_reqs = self.get_gpu_reqs("agent")
        s.add_tasks(t)
        return s

    def generate_pipeline(self):
        p = entk.Pipeline()
        for phase in range(int(self.args.num_phases)):
            p.add_stages(self.run_sim(phase))
            p.add_stages(self.run_train(phase))
            p.add_stages(self.run_selection(phase))
            p.add_stages(self.run_agent(phase))
        return p

    def run_workflow(self):
        p = self.generate_pipeline()
        self.am.workflow = [p]
        self.am.run()

    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

if __name__ == "__main__":
    mvp = MVP()
    res_config = mvp.config.get("resource", {})
    num_nodes = mvp.args.num_nodes
    resource_desc = {
        'resource': res_config.get("resource"),
        'queue': mvp.args.queue,
        'walltime': res_config.get("walltime"),
        'cpus': res_config.get("cpus_per_node", 1) * num_nodes,
        'gpus': res_config.get("gpus_per_node", 1) * num_nodes,
        'project': mvp.args.project_id
    }
    mvp.set_resource(res_desc=resource_desc)
    mvp.run_workflow()
