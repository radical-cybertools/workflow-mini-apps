This workflow mini-app build for xGFabric Project

To execute the mini-app please follow the steps provided here

After setting the enviroment please install necessary libraries:
```sh
pip install mpi4py
```
 if GPU is needed please install necessaray cupy library. This step may change based on the system you are installing. 
```sh
pip install cupy
```

## How to Execute the Simulation

The simulation can be run directly using MPI with the following command:

```sh
mpirun -np <num_processes> python /path/to/Executables/simulation.py [options]
```

Where:
- `<num_processes>` is the number of MPI processes you want to use

### Command-line Options

The simulation script accepts several arguments:

- `--phase`: Current phase of workflow (default: 0)
- `--mat_size`: Matrix size (default: 3000)
- `--data_root_dir`: Root directory for data (default: './')
- `--num_mult`: Number of matrix multiplications (default: 10)
- `--sim_inner_iter`: Inner iterations for each matrix multiplication (default: 10)
- `--write_size`: Bytes to write to disk (default: -1, write data once)
- `--read_size`: Bytes to read from disk (default: 0)
- `--input_file`: Path to an input file to read
- `--read_ratio`: Ratio of file to read (default: 1.0)
- `--scale_matrix`: Flag to enable scaling matrix size based on input file size (default: disabled)

### Example Commands

Basic simulation:
```sh
mpirun -np 4 python Executables/simulation.py
```


Simulation with dynamic matrix sizing based on input file:
```sh
mpirun -np 4 python Executables/simulation.py --input_file /path/to/your/data.txt --scale_matrix --read_ratio 0.5 
```

Customizing matrix size and iterations:
```sh
mpirun -np 4 python Executables/simulation.py --mat_size 1000 --sim_inner_iter 5
```

## Using with Workflow Scripts

To run the simulutaion code using workflow scripts:

- You would need to have RADICAL-Ensemble Toolkit 
  - https://radicalentk.readthedocs.io/en/latest/

After installing EnTK please run workflow mini-app
`source source_me.sh`
`./launch-scripts/run_<basic/serial/parallel>_workflow.sh`

You can change the parameter in these script accordingly.

Read the arguments and provide them as needed. 
- The scripts on the rct-scripts are build for ALCF Theta machine if you are running on an other machine please edit "mvp.set_resource" 
- You can use local_host for running locally. 
- The Exalearn project can be found in https://github.com/GKNB/Exalearn_theta_polaris_rct, for detail.
