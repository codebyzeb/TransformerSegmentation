host_ip=$(hostname --ip-address)
port_number=$(shuf -i 29510-49510 -n 1)
torchrun --nnodes 1 --rdzv_endpoint $host_ip:$port_number --nproc_per_node 1 train.py $@
