mpiexec -n 4 --ppn 4 python ./simulation.py --phase 2 --mat_size 600 --num_mult 21
mpiexec -n 3 --ppn 3 python ./training.py --num_epochs 11 --device cpu --phase 2 --mat_size 600 --num_mult 21 --sim_rank 4
