# ExaLearn Mini-App Implementation
Workflow mini-app build based on ExaLearn workflow. 

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


To execute the mini-app please follow the steps provided here

- You would need to have RADICAL-Ensemble Toolkit 
  - https://radicalentk.readthedocs.io/en/latest/

After installing EnTK please run workflow mini-app
`source source_me.sh`
`./launch-scripts/run_<basic/serial/parallel>_workflow.sh`

You can change the parameter in these script accordingly.

Read the arguments and provide them as needed. 

- The scripts on the rct-scripts are built for ALCF Theta machine; if you are running on an other machine please edit "mvp.set_resource" 
- You can use local_host for running locally. 
- The Exalearn project can be found in https://github.com/GKNB/Exalearn_theta_polaris_rct, for detail.
