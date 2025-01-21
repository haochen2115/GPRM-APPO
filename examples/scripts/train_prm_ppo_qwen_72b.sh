
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1

pip install transformers==4.44.2
pip install deepspeed==0.14.0
pip install accelerate==1.0.1
pip install ray[default]

mkdir -p /root/.cache/huggingface/hub
echo 1 > /root/.cache/huggingface/hub/version.txt

NODE_NAME=`echo $ILOGTAIL_PODNAME | awk -F 'ptjob-' '{print $2}'`
NODE_NAME=${NODE_NAME:-master-0}
output_dir=/path/to/your/output/dir
log_file=/path/to/your/log/file
cur_time=$(date "+%Y%m%d-%H%M%S")
echo $cur_time >> $log_file

set -x

# 先启动2个reward-model并记录下rm server的ip
# sh examples/scripts/serve_remote_gen_prm.sh &
# sh examples/scripts/serve_remote_dense_prm.sh & 
RM_SERVER_IP=0.0.0.0

# 启动ray服务的master节点和worker节点
MASTER_IP=0.0.0.0
# ray start --head --node-ip-address $MASTER_IP --num-gpus 8
# ray start --address $MASTER_IP:6379  --num-gpus 8

# 在master节点提交ray job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/training/code/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 4 \
   --ref_num_gpus_per_node 2 \
   --actor_num_nodes 8 \
   --actor_num_gpus_per_node 4 \
   --critic_num_nodes 4 \
   --critic_num_gpus_per_node 4 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 4 \
   --pretrain /path/to/your/sft/model \
   --remote_rm_url "http://${RM_SERVER_IP}:5001/get_gen_prm,http://${RM_SERVER_IP}:5002/get_dense_prm" \
   --save_path $output_dir \
   --ckpt_path $output_dir/ckpt \
   --save_steps 100 \
   --logging_steps 1 \
   --eval_steps 10 \
   --micro_train_batch_size 1 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 4096 \
   --generate_max_len 2048 \
   --max_samples 100000 \
   --advantage_estimator gae \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.05 \
   --prompt_data json@/path/to/your/data \
   --input_key input \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --save_steps -1 \
   --vllm_sync_backend nccl \
   --placeholder_token_id 151665 \
   --action_level token \
   2>&1 | tee $log_file
