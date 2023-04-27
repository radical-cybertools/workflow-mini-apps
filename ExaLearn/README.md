This workflow mini-app build based on ExaLearn workflow. 

To execute the mini-app please follow the steps provided here

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
