import argparse, asyncio, yaml, random, logging
import time
from pathlib import Path

from rhapsody.backends import DragonExecutionBackend

from radical.asyncflow import WorkflowEngine
from radical.asyncflow.logging import init_default_logger

from wfMiniAPI import kernel as kern
from wfMiniAPI.api import MiniApp

class Timer:
    def __init__(self): self.t0 = None
    def start(self): self.t0 = time.time(); return self
    def stop(self): return time.time() - self.t0

def load_cfg(path): 
    with open(path, "r") as f: return yaml.safe_load(f)

async def run_miniapp(cfg):
    import multiprocessing as mp
    logger = logging.getLogger(__name__)
    init_default_logger(logging.DEBUG)
    
    mp.set_start_method("dragon")
    backend = await DragonExecutionBackend()
    flow = await WorkflowEngine.create(backend=backend)
    telemetry = await flow.start_telemetry(checkpoint_path="telemetry")
    app = MiniApp(flow, telemetry)
    try:
        try:
            # every task reads ./data_0.h5 and writes its own file: create it for the largest read
            input_size = max(e["read_size_bytes"] for e in cfg.values() if "read_size_bytes" in e)
            kern.writeNonMPI(num_bytes=input_size, data_root_dir="./")
            async with app, flow.workflow_scope("motif_3_parallel"):
                await miniapp_workflow(cfg, flow, logger)
        finally:
            try:
                await flow.shutdown()
            finally:
                await telemetry.stop()
                app.verify()
    finally:
        # delete the .h5 files the run wrote, unless the config keeps them
        if not cfg.get("outputs", {}).get("keep"):
            for path in Path(".").glob("data_0*.h5"):
                path.unlink()

async def miniapp_workflow(cfg, flow, logger):

    @flow.function_task
    async def simulate(sim_idx, _train=None):
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
        kern.writeNonMPI(num_bytes=write_size, data_root_dir="./", filename_suffix=f"simulate_{sim_idx}")
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
        kern.writeNonMPI(num_bytes=write_size, data_root_dir="./", filename_suffix="train")
        logger.info(f"Finished training model with evaluation results...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()

    simulate_t0= simulate(0)
    
    train_t1 = train(simulate_t0)
    simulate_t1 = simulate(1, simulate_t0)
    
    train_t2 = train(simulate_t1, train_t1)
    simulate_t2 = simulate(2, train_t1)
    
    train_t3 = await train(simulate_t2, train_t2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inverse Design (Motif 3) Mini-App")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t = Timer().start()
    asyncio.run(run_miniapp(cfg))
    print(f"DONE in {t.stop():.3f}s")
