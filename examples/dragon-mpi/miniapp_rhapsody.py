import argparse, asyncio, yaml, random, logging, os
import time

from rhapsody.backends import DragonExecutionBackendV3

from radical.asyncflow import WorkflowEngine
from radical.asyncflow.logging import init_default_logger

from wfMiniAPI import kernel as kern

class Timer:
    def __init__(self): self.t0 = None
    def start(self): self.t0 = time.time(); return self
    def stop(self): return time.time() - self.t0

def load_cfg(path): 
    with open(path, "r") as f: return yaml.safe_load(f)

async def workflow(cfg):
    import multiprocessing as mp
    import mpi4py
    from mpi4py import MPI
    logger = logging.getLogger(__name__)
    init_default_logger(logging.DEBUG)

    mp.set_start_method("dragon")
    backend = await DragonExecutionBackendV3()
    flow = await WorkflowEngine.create(backend=backend)
    
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    kern.writeNonMPI(num_bytes=64, data_root_dir="./input")

    s1 = cfg["stage1"]
    @flow.function_task
    async def stage1(task_description={'ranks': s1['ranks'], 'type': 'mpi'}, *args):
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        import mpi4py
        from mpi4py import MPI
        s1 = cfg["stage1"]
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"Rank {rank} of {size} says: Hello from Stage 1!", flush=True)
        steps = s1["steps"]
        read_size = s1["read_size_bytes"]
        write_size = s1["write_size_bytes"]
        device = s1["device"]
        matmul_dim = s1["matmul_dim"]
        
        kern.readWithMPI(num_bytes=read_size, data_root_dir="./input")
        kern.generateRandomNumber(device=device, size=matmul_dim)
        for j in range(steps):
            kern.matMulSimple2D(device=device, size=matmul_dim)
        kern.writeWithMPI(num_bytes=write_size, data_root_dir="./output")
        logger.info(f"Finished stage 1...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()
        return rank
    
    s2 = cfg["stage2"]
    @flow.function_task
    async def stage2(task_description={'ranks': s2['ranks'], 'type': 'mpi'}, *args):
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        import mpi4py
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"Rank {rank} of {size} says: Hello from Stage 2!", flush=True)
        s2 = cfg["stage2"]
        steps = s2["steps"]
        read_size = s2["read_size_bytes"]
        write_size = s2["write_size_bytes"]
        device = s2["device"]
        matmul_dim = s2["matmul_dim"]

        kern.readWithMPI(num_bytes=read_size, data_root_dir="./input")
        kern.generateRandomNumber(device=device, size=matmul_dim)
        for _ in range(steps):
            kern.matMulSimple2D(device=device, size=matmul_dim)
        kern.writeWithMPI(num_bytes=write_size, data_root_dir="./output")
        logger.info(f"Finished stage 2...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()

    s3 = cfg["stage3"]
    @flow.function_task
    async def stage3(task_description={'ranks': s3['ranks'], 'type': 'mpi'}, *args):
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        import mpi4py
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"Rank {rank} of {size} says: Hello from Stage 3!", flush=True)
        s3 = cfg["stage3"]
        steps = s3["steps"]
        read_size = s3["read_size_bytes"]
        write_size = s3["write_size_bytes"]
        copy_size = s3["data_copy_size_bytes"]
        matmul_dim = s3["matmul_dim"]

        kern.readWithMPI(num_bytes=read_size, data_root_dir="./input")
        kern.dataCopyH2D(data_size=copy_size)
        for i in range(steps):
            kern.matMulSimple2D(device="gpu", size=matmul_dim)
            kern.matMulSimple2D(device="cpu", size=matmul_dim)
        kern.dataCopyH2D(data_size=copy_size)
        kern.writeWithMPI(num_bytes=write_size, data_root_dir="./output")
        logger.info(f"Finished stage 3...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()

    s4 = cfg["stage4"]
    @flow.function_task
    async def stage4(task_description={'ranks': s4['ranks'], 'type': 'mpi'}, *args):
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        import mpi4py
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"Rank {rank} of {size} says: Hello from Stage 4!", flush=True)
        s4 = cfg["stage4"]
        steps = s4["steps"]
        read_size = s4["read_size_bytes"]
        write_size = s4["write_size_bytes"]
        copy_size = s4["data_copy_size_bytes"]
        matmul_dim = s4["matmul_dim"]

        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        kern.readWithMPI(num_bytes=read_size, data_root_dir="./input")
        kern.dataCopyH2D(data_size=copy_size)
        for i in range(steps):
            kern.matMulSimple2D(device="gpu", size=matmul_dim)
            kern.matMulSimple2D(device="cpu", size=matmul_dim)
        kern.dataCopyH2D(data_size=copy_size)
        kern.writeWithMPI(num_bytes=write_size, data_root_dir="./output")
        logger.info(f"Finished stage 4...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()
    
    stage1_t = stage1()
    stage2_t = stage2(stage1_t)
    stage3_t = stage3(stage1_t)
    stage4_t = await stage4(stage2_t, stage3_t)

    await flow.shutdown()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MPI with Dragon Mini-App")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t = Timer().start()
    asyncio.run(workflow(cfg))
    print(f"DONE in {t.stop():.3f}s")
