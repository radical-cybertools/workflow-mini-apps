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
    async def simulate(i):
        e = cfg["simulation"]
        steps = e["steps"]
        device = e["device"]
        read_size = e["read_size_bytes"]
        write_size = e["write_size_bytes"]
        matmul_dim = e["matmul_dim"]

        logger.info(f"Generating training data with simulation {i}...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        kern.generateRandomNumber(device=device, size=matmul_dim)
        for _ in range(steps):
            kern.matMulSimple2D(device=device, size=matmul_dim)
        kern.writeNonMPI(num_bytes=write_size, data_root_dir="./")
        logger.info(f"Finished simulation {i}...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()

    @flow.function_task
    async def train(model_id, steps, matmul_dim):
        e = cfg["training"]
        read_size = e["read_size_bytes"]
        write_size = e["write_size_bytes"]
        copy_size = e["data_copy_size_bytes"]

        logger.info(f"Training model {model_id} (steps={steps}, dim={matmul_dim})...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        kern.readNonMPI(num_bytes=read_size, data_root_dir="./")
        kern.dataCopyH2D(data_size=copy_size)
        for _ in range(steps):
            kern.matMulSimple2D(device="gpu", size=matmul_dim)
            kern.matMulSimple2D(device="cpu", size=matmul_dim)
        kern.dataCopyD2H(data_size=copy_size)
        kern.writeNonMPI(num_bytes=write_size, data_root_dir="./")
        logger.info(f"Finished training model {model_id}...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()

    @flow.function_task
    async def analyze(num_models):
        e = cfg["analysis"]
        steps = e["steps"]
        read_size = e["read_size_bytes"]
        write_size = e["write_size_bytes"]
        copy_size = e["data_copy_size_bytes"]
        matmul_dim = e["matmul_dim"]

        logger.info(f"Analyzing accuracy of {num_models} models and data quality...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        kern.readNonMPI(num_bytes=read_size, data_root_dir="./")
        kern.dataCopyH2D(data_size=copy_size)
        for _ in range(steps):
            kern.matMulSimple2D(device="gpu", size=matmul_dim)
            kern.matMulSimple2D(device="cpu", size=matmul_dim)
        kern.dataCopyD2H(data_size=copy_size)
        kern.writeNonMPI(num_bytes=write_size, data_root_dir="./")
        logger.info(f"Finished analysis...")
        logger.info(time.strftime("%H:%M:%S", time.localtime()))
        return {"accuracy": [random.random() for _ in range(num_models)],
                "coverage": random.random()}

    ens = cfg["ensemble"]
    tr = cfg["training"]

    def sample_hyperparams():
        return {"steps": random.randint(tr["min_steps"], tr["max_steps"]),
                "matmul_dim": random.choice(tr["matmul_dims"])}

    # Initial data generation
    sim_count = ens["initial_simulations"]
    sim_t = [simulate(i) for i in range(sim_count)]
    await asyncio.gather(*sim_t)

    models = {m: sample_hyperparams() for m in range(ens["num_models"])}
    active = list(models.keys())

    for round_idx in range(ens["max_rounds"]):
        logger.info(f"Round {round_idx}: training models {active}")

        train_t = [train(m, models[m]["steps"], models[m]["matmul_dim"])
                   for m in active]
        await asyncio.gather(*train_t)

        result = await analyze(len(active))
        accuracy = result["accuracy"]
        coverage = result["coverage"]

        # Terminate low-accuracy models and respawn them with new configurations
        retrain = []
        for m, acc in zip(active, accuracy):
            if acc < ens["accuracy_threshold"]:
                models[m] = sample_hyperparams()
                retrain.append(m)
                logger.info(f"Round {round_idx}: model {m} terminated "
                            f"(accuracy {acc:.3f}), respawning with new config")
            else:
                logger.info(f"Round {round_idx}: model {m} converged "
                            f"(accuracy {acc:.3f})")
        active = retrain

        if not active:
            logger.info(f"Round {round_idx}: all models converged, stopping")
            break

        # Spawn new simulations when data coverage is insufficient
        if coverage < ens["coverage_threshold"]:
            n = ens["new_simulations_per_round"]
            logger.info(f"Round {round_idx}: coverage {coverage:.3f} too low, "
                        f"spawning {n} new simulations")
            sim_t = [simulate(sim_count + i) for i in range(n)]
            sim_count += n
            await asyncio.gather(*sim_t)

    await flow.shutdown()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Adaptive Training (Motif 6) Mini-App")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t = Timer().start()
    asyncio.run(workflow(cfg))
    print(f"DONE in {t.stop():.3f}s")
