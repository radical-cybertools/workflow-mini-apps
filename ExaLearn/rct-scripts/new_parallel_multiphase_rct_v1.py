from radical import entk
import os
import argparse, sys, math

class MVP(object):

    def __init__(self):
        self.set_argparse()
        self.am = entk.AppManager()

    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="Exalearn_miniapp_EnTK_parallel")

        parser.add_argument('--num_phase', type=int, default=0,
                        help='number of phases in the workflow')
        parser.add_argument('--mat_size', type=int, default=3000,
                        help='the matrix with have size of mat_size * mat_size')
        parser.add_argument('--data_root_dir', default='./',
                        help='the root dir of gsas output data')
        parser.add_argument('--exec_pattern', default='single-thread',
                        help='running pattern, can be single-thread, multi-thread or MPI')
        parser.add_argument('--model_dir', default='./',
                        help='the directory where save and load model')
        parser.add_argument('--project_id', required=True,
                        help='the project ID we used to launch the job')
        args = parser.parse_args()
        self.args = args

    #TODO we need to provide a generic execution for anyone to use
    #FIXME we need to use relative path.
    # This is for simulation, return a sim task
    def run_mpi_sweep_hdf5_py(self, phase_idx):

        t = entk.Task()
        t.pre_exec = [
                "module load conda/2021-09-22",
                "export OMP_NUM_THREADS=32"
                ]
        t.executable = 'python'
        t.arguments = ['/home/twang3/myWork/miniapp-exalearn/RECUP/mini-apps/ExaLearn-final/Executables/mini-mpi_sweep_hdf5_theta.py',
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--phase={}'.format(phase_idx),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--exec_pattern={}'.format(self.args.exec_pattern)]
        t.post_exec = []
        t.cpu_reqs = {
            'cpu_processes': 1,
            'cpu_process_type': None,
            'cpu_threads': 64,
            'cpu_thread_type': None
            }

        return t


    # This is for training, return a training task
    def run_mtnetwork_training_horovod_py(self, phase_idx):

        t = entk.Task()
        t.pre_exec = [
                'module load conda/2021-09-22',
                'export OMP_NUM_THREADS=32'
                ]
        t.executable = 'python'
        t.arguments = ['/home/twang3/myWork/miniapp-exalearn/RECUP/mini-apps/ExaLearn-final/Executables/test_training.py',
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--phase={}'.format(phase_idx),
                       '--mat_size={}'.format(self.args.mat_size)]
        t.post_exec = []
        t.cpu_reqs = {
             'cpu_processes'    : 1,
             'cpu_process_type' : None,
             'cpu_threads'      : 64,
             'cpu_thread_type'  : None
             }

        return t

    def generate_pipeline(self):

        p = entk.Pipeline()

        s0 = entk.Stage()
        t0 = self.run_mpi_sweep_hdf5_py(0)
        s0.add_tasks(t0)
        p.add_stages(s0)

        for phase in range(1, int(self.args.num_phase)):
            s = entk.Stage()
            ta = self.run_mpi_sweep_hdf5_py(phase)
            tb = self.run_mtnetwork_training_horovod_py(phase-1)
            s.add_tasks(ta)
            s.add_tasks(tb)
            p.add_stages(s)

        sf = entk.Stage()
        tf = self.run_mtnetwork_training_horovod_py(int(self.args.num_phase) - 1)
        sf.add_tasks(tf)
        p.add_stages(sf)

        return p

    def run_workflow(self):
        p = self.generate_pipeline()
        self.am.workflow = [p]
        self.am.run()


if __name__ == "__main__":
#TODO maybe we can use arguments to ask for resource, walltime, cpu, gpu too
    mvp = MVP()
    n_nodes = 2
    mvp.set_resource(res_desc = {
        'resource': 'anl.theta',
        'queue'   : 'debug-flat-quad',
#        'queue'   : 'default',
        'walltime': 60, #MIN
        'cpus'    : 64 * n_nodes,
        'gpus'    : 0 * n_nodes,
        'project' : mvp.args.project_id
        })
    mvp.run_workflow()
