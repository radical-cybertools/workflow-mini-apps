def triad_kernel(N_atoms, avg_neighbors, flops_per_pair, alpha=1.0000001, beta=0.0000001):
    """
    A batched AXPY/Triad on a 2D array:
      arr <- arr * alpha + beta
    Parameters:
    - N_atoms: number of "particles"
    - avg_neighbors: length of each vector slice
    - flops_per_pair: number of times to repeat the triad
    """
    arr = xp.ones((N_atoms, avg_neighbors), dtype=xp.float32)
    for _ in range(flops_per_pair):
        arr = arr * alpha + beta
    return arr

def reduction_kernel(arr):
    """
    Global reduction (sum) over all elements.
    """
    return arr.sum()

def run_miniapp(nsteps,
                N_atoms=10000,
                avg_neighbors=50,
                flops_per_pair=20,
                build_freq=1):
    """
    Run the mini-app for `nsteps`, calling the two kernels each step.
    Tunable parameters:
    - N_atoms, avg_neighbors, flops_per_pair control the work per kernel.
    - build_freq controls how often triad_kernel is called (can model neighbor-list rebuild frequency).
    """
    for step in range(nsteps):
        # Optionally skip triad kernel on some steps
        if step % build_freq == 0:
            arr = triad_kernel(N_atoms, avg_neighbors, flops_per_pair)
        # Always do reduction to mimic energy sum
        energy = reduction_kernel(arr)
        if step % max(nsteps // 5, 1) == 0:
            print(f"Step {step}: energy={energy:.3e}")
    # Return final energy (and free GPU memory if used)
    if gpu_enabled:
        xp.get_default_memory_pool().free_all_blocks()
    return energy

if __name__ == "__main__":
    # Example: 10 steps, 5k atoms, 60 neighbors, 30 flops per pair, triad every step
    run_miniapp(nsteps=10, N_atoms=5000, avg_neighbors=60, flops_per_pair=30, build_freq=1)

