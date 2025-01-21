CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m openrlhf.cli.serve_genprm_simple \
    --reward_pretrain /path/to/your/gen_prm/ \
    --max_len 8192 \
    --port 5001 \
    --placeholder_token "<|ки|>" \
    --reward_tokens "<|+|>" "<|-|>"

