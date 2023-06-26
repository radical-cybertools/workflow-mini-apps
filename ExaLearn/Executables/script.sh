mpiexec -n 4 --ppn 4 python ./simulation.py --phase 2 --mat_size 600 --num_mult 21 --inner_iter 1
mpiexec -n 3 --ppn 3 python ./training.py --num_epochs 11 --device gpu --phase 2 --mat_size 600 --num_mult 21 --sim_rank 4 --preprocess_time 5
python ./simulation.py --phase 2 --mat_size 3000 --inner_iter 24 --data_root_dir /eagle/RECUP/miniapp-Exalearn/test_miniapp_v2/serial_wf/data/ --mat_size 3000 --num_mult 10

