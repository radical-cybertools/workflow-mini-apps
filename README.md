# RADICAL Workflow Mini-Apps
Workflow Mini-Apps provides [small, self-contained representations of scientific workflows (or mini-apps)](https://arxiv.org/abs/2403.18073) for developing workflows.
Each mini-app is a simplified version of a complex scientific workflow, capturing its key tasks, data flow, and performance characteristics without the deployment challenges of the full application.
Workflow Mini-apps can be scaled and configured without application specific deployment challenges and constraints​.

Workflow Mini-app facilitate experimentation and helps understand workflow (distinct from application) performance.

There are 2 example Workflow Mini-apps:

- Scalable Adaptive Learning (ExaLearn)

- DeepDriveMD

### Installation
1). Install rct. Please make sure to use conda env approach since we also need an env that has cupy/h5py/mpi4py

2). Install darshan. Please make sure to modify the darshan code as explained so that it can be used to collect info. Also don't forget to install darshan-util

3). Set the environment, a sample script is shown below:

```
#/bin/bash

module load cray-hdf5/1.12.1.3
module load conda
conda activate env/rct-recup-polaris

which python
python -V

export RADICAL_PILOT_DBURL=$(cat useful_script/rp_dburl_polaris)
echo $RADICAL_PILOT_DBURL

export RADICAL_LOG_LVL=DEBUG
export RADICAL_PROFILE=TRUE
export RADICAL_SMT=1

export PATH=/home/$USER/libraries/darshan/bin:$PATH
```

Here "env/rct-recup-polaris" is the conda env with rct, and "/home/$USER/libraries/darshan/bin" is where darshan is installed. "RADICAL_PILOT_DBURL" is not necessary anymore with the latest rct.

4). Go to the specific mini-app sub-dir, then do source `source_me.sh` 

5). Go to launch-scripts to run the experiment. Before starting, make sure the parameters have been set up

6). Analyze the results. Some useful tools can be found in Analyze/
