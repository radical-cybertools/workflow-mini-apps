from radical import entk
import os
import argparse, sys, math
import radical.pilot as rp
import radical.utils as ru

class MVP(object):

    def __init__(self):
        self.env_work_dir = os.getenv("MINI_APP_EXALEARN_DIR")
        if self.env_work_dir is None:
            print("Warning: Did not set up work_dir using env var, need to set it up in parser manually!")
        self.set_argparse()
        self.am = entk.AppManager()

    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="Exalearn_miniapp_EnTK_serial")

        parser.add_argument('--num_phases', type=int, default=0,
                        help='number of phases in the workflow')
        parser.add_argument('--num_epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
        parser.add_argument('--mat_size', type=int, default=3000,
                        help='the matrix with have size of mat_size * mat_size')
        parser.add_argument('--data_root_dir', default='./',
                        help='the root dir of gsas output data')
        parser.add_argument('--model_dir', default='./',
                        help='the directory where save and load model')
        parser.add_argument('--project_id', required=True,
                        help='the project ID we used to launch the job')
        parser.add_argument('--work_dir', default=self.env_work_dir,
                        help='working dir, which is the dir of this repo')
        parser.add_argument('--num_mult', type=int, default=10,
                        help='number of matrix mult to perform, need to be larger than num_worker!')
        parser.add_argument('--sim_rank', type=int, default=1,
                        help='number of rank used for simulation. This is needed to determine the size of data in those files')
        parser.add_argument('--train_rank', type=int, default=1,
                        help='number of rank used for training.')
        parser.add_argument('--sim_inner_iter', type=int, default=10,
                        help='number of inner iter for each matrix mult in simulation app. Used to control sim workload size')
        parser.add_argument('--train_inner_iter', type=int, default=1,
                        help='inner iteration of mult and allreduce')
        parser.add_argument('--num_allreduce', type=int, default=1,
                        help='the number of allreduce op performed')
        parser.add_argument('--train_preprocess_time', type=float, default=5.0,
                        help='time for doing preprocess in training')
        parser.add_argument('--sim_read_size', type=int, default=0,
                        help='read bytes for all ranks in one sim task')
        parser.add_argument('--sim_write_size', type=int, default=0,
                        help='write bytes for all ranks in one sim task')
        parser.add_argument('--train_read_size', type=int, default=0,
                        help='read bytes for all ranks in one train task')

        args = parser.parse_args()
        self.args = args

    # This is for simulation, return a stage which has a single sim task
    def run_mpi_sweep_hdf5_py(self, phase_idx):

        t = entk.Task({'uid': ru.generate_id("sim")})
        t.pre_exec = [
                "module load PrgEnv-gnu",
                "module load conda",
                "conda activate /grand/CSC249ADCD08/twang/env/rct-recup-polaris",
                "export IBV_FORK_SAFE=1",
                "export HDF5_USE_FILE_LOCKING=FALSE"
                ]
        t.executable = 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/lib/python3.8/,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/bin/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/lib/python3.8/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/bin/,/home/twang3/g2full_theta/GSASII/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python'
        t.arguments = ['{}/Executables/simulation.py'.format(self.args.work_dir),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--phase={}'.format(phase_idx),
                       '--num_mult={}'.format(self.args.num_mult),
                       '--sim_inner_iter={}'.format(self.args.sim_inner_iter),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--write_size={}'.format(self.args.sim_write_size),
                       '--read_size={}'.format(self.args.sim_read_size)]
        t.post_exec = []
        t.cpu_reqs = {
             'cpu_processes'    : self.args.sim_rank,
             'cpu_process_type' : None,
             'cpu_threads'      : 1,
             'cpu_thread_type'  : rp.OpenMP
             }

        s = entk.Stage()
        s.add_tasks(t)
        return s


    # This is for training, return a stage which has a single training task
    def run_mtnetwork_training_horovod_py(self, phase_idx):

        t = entk.Task({'uid': ru.generate_id("train")})
        t.pre_exec = [
                "module load PrgEnv-gnu",
                'module load conda',
                "conda activate /grand/CSC249ADCD08/twang/env/rct-recup-polaris",
                "export IBV_FORK_SAFE=1",
                "export HDF5_USE_FILE_LOCKING=FALSE"
                ]
        t.executable = 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/lib/python3.8/,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/bin/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/lib/python3.8/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/bin/,/home/twang3/g2full_theta/GSASII/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python'
        t.arguments = ['{}/Executables/training.py'.format(self.args.work_dir),
                       '--num_epochs={}'.format(self.args.num_epochs),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--phase={}'.format(phase_idx),
                       '--num_mult={}'.format(self.args.num_mult),
                       '--train_inner_iter={}'.format(self.args.train_inner_iter),
                       '--num_allreduce={}'.format(self.args.num_allreduce),
                       '--sim_rank={}'.format(self.args.sim_rank),
                       '--device=gpu',
                       '--preprocess_time={}'.format(self.args.train_preprocess_time),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--read_size={}'.format(self.args.train_read_size)]
        t.post_exec = []
        t.cpu_reqs = {
            'cpu_processes'     : self.args.train_rank,
            'cpu_process_type'  : None,
            'cpu_threads'       : 8,
            'cpu_thread_type'   : rp.OpenMP
                }
        t.gpu_reqs = {
            'gpu_processes'     : 1,
            'gpu_process_type'  : rp.CUDA
                }

        s = entk.Stage()
        s.add_tasks(t)
        return s

    def generate_pipeline(self):

        p = entk.Pipeline()
        for phase in range(int(self.args.num_phases)):
            s1 = self.run_mpi_sweep_hdf5_py(phase)
            p.add_stages(s1)
            s2 = self.run_mtnetwork_training_horovod_py(phase)
            p.add_stages(s2)
        return p

    def run_workflow(self):
        p = self.generate_pipeline()
        self.am.workflow = [p]
        self.am.run()


if __name__ == "__main__":

    mvp = MVP()
    n_nodes = 4
    mvp.set_resource(res_desc = {
        'resource': 'anl.polaris',
#        'queue'   : 'debug',
        'queue'   : 'preemptable',
#        'queue'   : 'default',
        'walltime': 45, #MIN
        'cpus'    : 32 * n_nodes,
        'gpus'    : 4 * n_nodes,
        'project' : mvp.args.project_id
        })
    mvp.run_workflow()
