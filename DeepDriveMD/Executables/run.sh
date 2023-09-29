python simulation.py --phase 1 --mat_size 5000 --data_root_dir ./ --num_step 1000 --write_size 3500000 --read_size 6000000

python training.py --num_epochs 150 --device gpu --phase 1 --data_root_dir ./ --model_dir ./ --num_sample 500 --num_mult 4000 --dense_dim_in 12544 --dense_dim_out 128 --mat_size 5000 --preprocess_time 0 --read_size 4000000 --write_size 7000000

python selection.py --phase 1 --mat_size 3000 --data_root_dir ./ --write_size 0 --read_size 0

python agent.py --num_epochs 10 --device gpu --phase 1 --data_root_dir ./ --model_dir ./ --num_sample 500 --num_mult 2000 --num_mult_outlier 10 --dense_dim_in 12544 --dense_dim_out 128 --mat_size 5000 --read_size 4000000 --write_size 7000000
