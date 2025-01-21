import argparse
import re
from typing import List, Optional
import uvicorn
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

class GenPRM:
    def __init__(self, args) -> None:
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.reward_pretrain,
            trust_remote_code=True,
            use_fast=False
        )

        # 初始化模型
        self.model = AutoModelForCausalLM.from_pretrained(
            args.reward_pretrain,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        
        self.max_length = args.max_len
        self.placeholder_token = args.placeholder_token
        
    def prepare_prompt(self, text: str) -> str:
        return f"""<|im_start|>user
请你根据当前的问题和模型输出的中间步骤进行得分，注意只需要对最后一个步骤进行评价即可。你需要先输出判断的过程，再输出判断的3个标记[ <|+|> , <|-|> , <|o|> ]
{text}
请输出你的点评和标签：
<|im_end|>
<|im_start|>assistant
"""

    def process_response(self, response: str) -> float:
        """处理生成的响应,提取分数"""
        gen_part = response.split("<|im_start|>assistant")[-1]
        
        if "<|+|>" in gen_part:
            print(" ✅ ",gen_part)
            score = 1.0
        elif "<|-|>" in gen_part:
            print(" ❌ ",gen_part)
            score = -1.0
        else:
            print(" 🀄️ ",gen_part)
            score = 0.0
            
        return score

    def get_reward(self, queries: List[str]) -> List[List[float]]:
        """批量获取奖励分数,返回token级别的reward"""
        all_rewards = []
        
        # 将queries转换为token ids
        tokenized = self.tokenizer(
            queries,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        input_ids = tokenized["input_ids"].to(self.model.device)
        attention_mask = tokenized["attention_mask"].to(self.model.device)
        
        for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
            # 初始化token级别的reward
            reward = torch.zeros_like(ids, dtype=torch.float)
            
            # 找到所有placeholder_token的位置
            placeholder_id = self.tokenizer.convert_tokens_to_ids(self.placeholder_token)
            step_indices = (ids == placeholder_id).nonzero(as_tuple=True)[0]
            
            if len(step_indices) < 5 or len(step_indices) > 50:
                logger.warning(f"""🐛[gen prm]
                🚨 Steps count {len(step_indices)} is abnormal for query {i} :
                {queries[i]}
                🚨 
                """)
            
            # 对每个step位置生成评分
            for step_idx in step_indices:
                # 构建到当前step的prompt
                prefix = ids[:step_idx+1].tolist()
                curr_input = self.prepare_prompt(self.tokenizer.decode(prefix))
                
                # 转换输入
                inputs = self.tokenizer(curr_input, return_tensors="pt").to(self.model.device)
                
                # 生成评分
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.01,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                score = self.process_response(generated_text)
                
                # 在对应的token位置设置reward
                reward[step_idx] = score
                
            all_rewards.append(reward.tolist())
            logger.info(f"Query {i}: {len(step_indices)} steps, sum reward: {torch.sum(reward)}")
        
        logger.info(f"all_rewards[0]: {all_rewards[0]}, len(all_rewards[0]): {len(all_rewards[0])}")

        return all_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument("--reward_pretrain", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=2048)
    
    # Server params
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    
    # prm parameters
    parser.add_argument("--placeholder_token", type=str, default=None)
    parser.add_argument("--reward_tokens", type=str, nargs="*", default=None)
    
    args = parser.parse_args()
    
    # 初始化模型
    reward_model = GenPRM(args)
    
    # 创建FastAPI应用
    app = FastAPI()
    
    @app.post("/get_gen_prm")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query", [])
        rewards = reward_model.get_reward(queries)
        return JSONResponse({"rewards": rewards})

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
