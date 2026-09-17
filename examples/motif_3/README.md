# Inverse Design Motif Workflow Mini-app
Workflow mini-app build based on an Inverse Design workflow. 

To execute the mini-app please follow these steps

### Load your modules
First load any modules you need for your environment. 
We did our experiments on Delta, so your needed modules may change.
Generally 

- `$ module load cudatoolkit/25.3_12.8`
- `$ module load python/3.13.5-gcc13.3.1`

### If using a venv, load the venv now

- `$ source path_to_venv/bin/activate`

If you have already installed wfMiniAPI, you can skip ahead to running the workflow.
### Installing wfMiniAPI and its dependencies
First clone the wfMiniAPI repository
- `$ git clone git@github.com:radical-cybertools/workflow-mini-apps.git`
- `$ cd workflow-mini-apps/`

Now before installing, we must patch the kernel.py to use the correct axpy kernel.
Edit `wfMiniAPI/src/wfMiniAPI/kernel.py` by uncommenting lines `305-313` and commenting lines `315-321`.
It should look like this:
```python
@annotate_kernel
def axpy_fuse(device, size):
    xp = get_device_module(device)
    x = xp.empty(size, dtype=xp.float32)
    y = xp.empty(size, dtype=xp.float32)
    if xp == np:
        y += 1.01 * x
    elif xp == cp:
        _axpy_fuse(1.01, x, y, size=size)

#_axpy_fuse_fast = cp.ElementwiseKernel(
#    'float32 alpha, raw float32 x',
#    'raw float32 y',               
#    'y[i] += alpha * x[i]',        
#    'axpy_fuse_kernel',
#    no_return=True                 
#)
```
There are also some dependencies which are not installed by `pip`.
You will need `cupy`, the specific version you need is determined by the CUDA version.
In our case, on DELTA we are using CUDA 12.8
- `$ pip install cupy-cuda12x`

Now we can install the wfMiniAPI
- `$ cd wfMiniAPI`
- `$ pip install .`

### Running the workflow
Navigate to the example motif and launch the workflow mini-app using Dragon
- `$ cd workflow-mini-apps/examples/motif_3`
- `$ dragon -s workflow_simple.py`
or
 `$ dragon -s workflow_parallel.py`

This will launch the workflows using Dragon with a single node.
Remove the `-s` to run the workflow with more than one node.