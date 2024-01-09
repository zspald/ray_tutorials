#!/bin/bash

#SBATCH --job-name=ray_train_clust
#SBATCH --output=out/ray_train_clust/slurm_%j.out
#SBATCH -e out/ray_train_clust/slurm_%j.err
#SBATCH -p gpu-common,scavenger-gpu

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=5

### Give all resources on each node to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ~/.bashrc
source activate ray

echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"

### Head address ###
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "All nodes: $nodes"

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
####################

### Ray Start on Head ###
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --block &

####################

### Ray Start on Workers ###

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --block &
    sleep 5
done

####################

### Run your program ###
python -u ray_distTrain_gpu.py -n "$SLURM_JOB_NUM_NODES"