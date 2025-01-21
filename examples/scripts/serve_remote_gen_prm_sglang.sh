# Deploy this on another machine
# CUDA_VISIBLE_DEVICE=0,1 \
# python -m sglang.launch_server \
#   --model-path /path/to/your/gen_prm/ \
#   --tp 2 \
#   --port 5001 \
#   --host 0.0.0.0 \
#   --trust-remote-code \

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m openrlhf.cli.serve_genprm_sglang \
    --reward_pretrain /path/to/your/gen_prm/ \
    --remote_url "http://0.0.0.0:5001/generate" \
    --max_len 8192 \
    --port 5001 \
    --placeholder_token "<|ки|>" \
    --reward_tokens "<|+|>" "<|-|>"
