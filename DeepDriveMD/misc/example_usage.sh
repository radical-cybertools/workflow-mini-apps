#This is the example usage of get_task_execution_time.py and get_workflow_execution_time.py
#The task execution time is obtained from agent_staging_output.0000.prof file, and return as a dictionary with key/value pair being task_id/execution time
#The workflow execution time is obtained from bootstrap_0.prof, and return as a single value of execution time of the entire workflow

python get_task_execution_time.py -f ../launch-scripts/re.session.polaris-login-02.twang3.019992.0000/pilot.0000/agent_staging_output.0000.prof
python get_workflow_execution_time.py -f ../launch-scripts/re.session.polaris-login-02.twang3.019992.0000/pilot.0000/bootstrap_0.prof

