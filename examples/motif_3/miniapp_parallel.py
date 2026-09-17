import argparse, asyncio, yaml, random, logging
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
    logger = logging.getLogger(__name__)
    init_default_logger(logging.DEBUG)
    
    mp.set_start_method("dragon")
    backend = await DragonExecutionBackendV3()
    flow = await WorkflowEngine.create(backend=backend)

    @flow.function_task
    async def simulate(_train=None):
        e = cfg["simulate"]
        steps = e["steps"]
        device = e["device"]
        read_size = e["read_size_bytes"]
        write_size = e["write_size_bytes"]
        matmul_dim = e["matmul_dim"]

        logger.info(f"Simulating with params...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        kern.generateRandomNumber(device=device, size=matmul_dim)
        for i in range(steps):
            kern.matMulSimple2D(device=device, size=matmul_dim)
        kern.writeNonMPI(num_bytes=write_size, data_root_dir="./")
        kern.readNonMPI(num_bytes=read_size, data_root_dir="./")
        logger.info(f"Finished simulating with params...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()

    @flow.function_task
    async def train(_sim, _train=None):
        e = cfg["training"]
        steps = e["steps"]
        device = e["device"]
        read_size = e["read_size_bytes"]
        write_size = e["write_size_bytes"]
        copy_size = e["data_copy_size_bytes"]
        matmul_dim = e["matmul_dim"]

        logger.info(f"Training model with evaluation results...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        kern.readNonMPI(num_bytes=read_size, data_root_dir="./")
        kern.dataCopyH2D(data_size=copy_size)
        for i in range(steps):
            kern.matMulSimple2D(device="gpu", size=matmul_dim)
            kern.matMulSimple2D(device="cpu", size=matmul_dim)
        kern.dataCopyH2D(data_size=copy_size)
        kern.writeNonMPI(num_bytes=write_size, data_root_dir="./")
        logger.info(f"Finished training model with evaluation results...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()

    simulate_t0= simulate()
    
    train_t1 = train(simulate_t0)
    simulate_t1 = simulate(simulate_t0)
    
    train_t2 = train(simulate_t1, train_t1)
    simulate_t2 = simulate(train_t1)
    
    train_t3 = await train(simulate_t2, train_t2)

    await flow.shutdown()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inverse Design (Motif 3) Mini-App")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t = Timer().start()
    asyncio.run(workflow(cfg))
    print(f"DONE in {t.stop():.3f}s")
