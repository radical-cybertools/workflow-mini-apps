# Workflow mini-app using MPI and Rhapsody with Dragon backend
This example goes over launching MPI tasks 
## First load the proper modules

### Load cray-mpich-abi
`$ module load cray-mpich-abi`

### Load cuda toolkit
`$ module load cudatoolkit`

### Load h5py with MPI support
`$ module load cray-hdf5`

### If using a venv, load the venv now
`$ source path_to_venv/bin/activate`

### Now you can launch using dragon
`$ dragon miniapp_mpi.py`

> [!NOTE]  
> DRAGON assumes it is launched via SLURM and will use SLURM environment variables.
> Trying to run DRAGON without these set will cause a crash or hang. 
> This can be manually fixed by running `export SLURM_JOB_NUM_NODES=1` (or however many nodes you would like).