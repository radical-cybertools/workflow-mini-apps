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
    async def simulate(stage_idx, candidate):
        e = cfg["stages"][stage_idx]
        steps = e["steps"]
        device = e["device"]
        read_size = e["read_size_bytes"]
        write_size = e["write_size_bytes"]
        matmul_dim = e["matmul_dim"]

        logger.info(f"Stage {stage_idx}: simulating candidate {candidate}...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        kern.readNonMPI(num_bytes=read_size, data_root_dir="./")
        kern.generateRandomNumber(device=device, size=matmul_dim)
        for _ in range(steps):
            kern.matMulSimple2D(device=device, size=matmul_dim)
        kern.writeNonMPI(num_bytes=write_size, data_root_dir="./")
        logger.info(f"Stage {stage_idx}: finished simulating candidate {candidate}...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()

    @flow.function_task
    async def analyze(stage_idx, num_candidates):
        e = cfg["analysis"]
        steps = e["steps"]
        read_size = e["read_size_bytes"]
        write_size = e["write_size_bytes"]
        copy_size = e["data_copy_size_bytes"]
        matmul_dim = e["matmul_dim"]

        logger.info(f"Stage {stage_idx}: analyzing {num_candidates} candidates...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        kern.readNonMPI(num_bytes=read_size, data_root_dir="./")
        kern.dataCopyH2D(data_size=copy_size)
        for _ in range(steps):
            kern.matMulSimple2D(device="gpu", size=matmul_dim)
            kern.matMulSimple2D(device="cpu", size=matmul_dim)
        kern.dataCopyD2H(data_size=copy_size)
        kern.writeNonMPI(num_bytes=write_size, data_root_dir="./")
        logger.info(f"Stage {stage_idx}: finished analyzing candidates...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return [random.random() for _ in range(num_candidates)]

    stages = cfg["stages"]
    candidates = list(range(stages[0]["ensemble_size"]))

    for stage_idx in range(len(stages)):
        e = stages[stage_idx]

        sim_t = [simulate(stage_idx, c) for c in candidates]
        await asyncio.gather(*sim_t)

        scores = await analyze(stage_idx, len(candidates))

        threshold = e["filter_threshold"]
        candidates = [c for c, s in zip(candidates, scores) if s >= threshold]
        logger.info(f"Stage {stage_idx}: {len(candidates)} candidates passed the filter")

        if not candidates:
            logger.info(f"Stage {stage_idx}: no candidates left, stopping the pipeline")
            break

        if stage_idx + 1 < len(stages):
            candidates = candidates[: stages[stage_idx + 1]["ensemble_size"]]

    await flow.shutdown()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multistage Pipeline (Motif 2) Mini-App")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t = Timer().start()
    asyncio.run(workflow(cfg))
    print(f"DONE in {t.stop():.3f}s")
