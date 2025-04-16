Please see the published paper accompanying this work: https://doi.org/10.1109/CCGrid59990.2024.00059 
or: https://arxiv.org/abs/2403.18073

How to run workflow mini-app on Polaris:

1). Install rct. Please make sure to use conda env approach since we also need an env that has cupy/h5py/mpi4py

2). Install darshan. Please make sure to modify the darshan code as explained so that it can be used to collect info. Also don't forget to install darshan-util

3). I currently use the following script on the login node to setup the env:

########################################################################################
```
#/bin/bash

module load cray-hdf5/1.12.1.3
module load conda
conda activate /grand/CSC249ADCD08/twang/env/rct-recup-polaris

which python
python -V

export RADICAL_PILOT_DBURL=$(cat /home/twang3/useful_script/rp_dburl_polaris)
echo $RADICAL_PILOT_DBURL

export RADICAL_LOG_LVL=DEBUG
export RADICAL_PROFILE=TRUE
export RADICAL_SMT=1

export PS1="[$CONDA_PREFIX] \u@\H:\w> "
export PATH=/home/twang3/libraries/darshan/bin:$PATH
```
########################################################################################

Here "/grand/CSC249ADCD08/twang/env/rct-recup-polaris" is the conda env where I install rct, and "/home/twang3/libraries/darshan/bin" is the place where I install darshan. "RADICAL_PILOT_DBURL" is not necessary anymore with the latest rct

4). Go to the specific mini-app sub-dir, then do source source_me.sh 

5). Go to launch-scripts to run the experiment. Before starting, make sure the parameters have been set up

6). Analyze the results. Some useful tools can be found in Analyze/



# RADICAL Workflow Mini-App
Mini-apps development repo

WF Mini-apps can be scaled and configured without application specific deployment challenges and constraints​

WF Mini-app facilitate experimentation, understand workflow (distinct from application) performance​

We develop 2 WF mini-apps:
 
 - WF1: Scalable Adaptive Learning Learning (ExaLearn)​
 
 - WF2: DeepDriveMD​


ExaLearn mini-app implementation:

- Task1: Data Generation
    - Does a random matrix multiplication and prints the result to a file
- Task2: ML
    - Reads matrices created by data generation and does a matrix multiplication
    - For the next step will run a simplified ML algorithm that has 1-2 layers.
-enTK implementations:
    - Serial: DG1 -> ML1 -> DG2 -> ML2 -> ... -> DGn -> MLn
    - Parallel: DG1-> (ML1, DG2) -> (ML2, DG3) -> ... -> (MLn-1, DGn) -> MLn
- Requirements:
    - \#Tasks: 8 ( 4 of each)
    - Currently static, will be adaptive
    - python executables
    - \# nodes: 128 (real app) Currently 1 for mini-app
    - \# cores: 64 per node
    - \# gpus: currently only CPU but will move to gpu soon (~128 per node)
    - Has input/output dependencies
    - Original ExaLearn Generates ~70GB of data during Data Generation and reads it during ML
    - Data generated and read in the same node. No multi-node data sharing
    - Systems: Currently Theta, will move to Polaris


