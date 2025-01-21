# 继承actor类，新增generate_reward方法
from typing import Optional, Tuple, Union, List
import torch
from openrlhf.models import Actor

import argparse
import re

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

class DensePRM(Actor):
    def __init__(self, pretrain_or_model, use_flash_attention_2=False, bf16=True, load_in_4bit=False, lora_rank=0, lora_alpha=16, lora_dropout=0, target_modules=None, ds_config=None, device_map="auto", packing_samples=False, **kwargs) -> None:
        super().__init__(pretrain_or_model, use_flash_attention_2, bf16, load_in_4bit, lora_rank, lora_alpha, lora_dropout, target_modules, ds_config, device_map, packing_samples, **kwargs)
    
    @torch.no_grad()
    def process_forward(
        self,
        tokenizer,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        placeholder_token_id=12902,
        reward_token_ids=[648, 387]
    ) -> torch.Tensor:
        self.tokenizer = tokenizer
        outputs = self.model(input_ids)
        logits = outputs.logits[:,:,reward_token_ids]
        scores = logits.softmax(dim=-1)[:,:,0] 
        mask = (input_ids == placeholder_token_id)
        reward = torch.where(mask, scores, torch.tensor(0.0))
        print(f"""
        🐛[dense prm] 
        torch.sum(mask) = {torch.sum(mask)} 
        torch.sum(reward) = {torch.sum(reward)} 
        input_ids.shape = {input_ids.shape}
        """)
        # 对batch中的每个样本分别检查
        for idx, sample_mask in enumerate(mask):
            if torch.sum(sample_mask) < 5 or torch.sum(sample_mask) > 50:
                # 只解码出现问题的那个样本
                query = self.tokenizer.decode(input_ids[idx], skip_special_tokens=False)
                debug_info = f"""
                🐛[dense prm]
                🚨 第{idx}个样本的步骤小于5或大于50，请检查下输入是否正常！
                query = 
                {query}
                步骤数量：{torch.sum(sample_mask)}
                🚨 
                """
                print(debug_info)
        return (reward, outputs) if return_output else reward


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")

class RewardModelProxy:
    def __init__(self, args):
        self.reward_model = DensePRM(
            args.reward_pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
        )
        self.reward_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.placeholder_token_id = convert_token_to_id(args.placeholder_token, self.tokenizer)
        # 这里注意，good和bad的token_id在tokenizer里面的id顺序一定要保持是bad token是较小值，这里是为了保证不管输入是什么样子，都选择较小id的token作为bad_token_id
        self.reward_token_ids = sorted(
            [convert_token_to_id(token, self.tokenizer) for token in args.reward_tokens]
        )

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # remove pad_token
        # for i in range(len(queries)):
        #     queries[i] = (
        #         strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
        #         + self.tokenizer.eos_token
        #     )
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                # bs, len
                inputs = self.tokenize_fn(
                    queries[i : min(len(queries), i + batch_size)], device=self.reward_model.model.device
                )
                r = self.reward_model.process_forward(
                    self.tokenizer, 
                    inputs["input_ids"], inputs["attention_mask"],
                    placeholder_token_id = self.placeholder_token_id,
                    reward_token_ids = self.reward_token_ids
                )
                
                r = r.tolist()
                scores.extend(r)
        
        logger.info(f"scores[0]: {scores[0]}, len(scores[0]): {len(scores[0])}")
        
        return scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    # prm parameters
    parser.add_argument("--placeholder_token", type=str, default=None)
    parser.add_argument("--reward_tokens", type=str, nargs="*", default=None)

    args = parser.parse_args()

    # server
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_dense_prm")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
