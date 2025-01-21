CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m openrlhf.cli.serve_denseprm \
    --reward_pretrain /path/to/your/prm \
    --port 5002 \
    --bf16 \
    --flash_attn \
    --max_len 8192 \
    --batch_size 16 \
    --placeholder_token "<|ки|>" \
    --reward_tokens "<|+|>" "<|-|>"