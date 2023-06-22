from radical import entk
import os
import argparse, sys, math
import radical.pilot as rp

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

        args = parser.parse_args()
        self.args = args

    # This is for simulation, return a stage which has a single sim task
    def run_mpi_sweep_hdf5_py(self, phase_idx):

        t = entk.Task()
        t.pre_exec = [
                "module load conda",
                "conda activate /grand/CSC249ADCD08/twang/env/rct-recup-polaris"
                ]
        t.executable = 'python'
        t.arguments = ['{}/Executables/simulation.py'.format(self.args.work_dir),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--phase={}'.format(phase_idx),
                       '--num_mult={}'.format(self.args.num_mult),
                       '--mat_size={}'.format(self.args.mat_size)]
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

        t = entk.Task()
        t.pre_exec = [
                'module load conda',
                "conda activate /grand/CSC249ADCD08/twang/env/rct-recup-polaris"
                ]
        t.executable = 'python'
        t.arguments = ['{}/Executables/training.py'.format(self.args.work_dir),
                       '--num_epochs={}'.format(self.args.num_epochs),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--phase={}'.format(phase_idx),
                       '--num_mult={}'.format(self.args.num_mult),
                       '--sim_rank={}'.format(self.args.sim_rank),
                       '--mat_size={}'.format(self.args.mat_size)]
        t.post_exec = []
        t.cpu_reqs = {
             'cpu_processes'    : self.args.train_rank,
             'cpu_process_type' : None,
             'cpu_threads'      : 8,
             'cpu_thread_type'  : rp.OpenMP
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
    n_nodes = 1
    mvp.set_resource(res_desc = {
        'resource': 'anl.polaris',
        'queue'   : 'preemptable',
#        'queue'   : 'default',
        'walltime': 30, #MIN
        'cpus'    : 32 * n_nodes,
        'gpus'    : 0 * n_nodes,
        'project' : mvp.args.project_id
        })
    mvp.run_workflow()
