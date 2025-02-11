import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from recbole.utils import set_color
import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
import re

# 设置设备参数
use_LLM = True
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息
model_path = '/root/haichao/modelscope_model/LLM-Research/'
model_name = 'Meta-Llama-3-8B-Instruct'
model_name_or_path = os.path.join(model_path, model_name)
prompt_files = 'ml-1m_LightGCN_prompt_top50_SelectionStrategy.RANDOM_forget.json'
max_retries = 5  # 最大重试次数

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)

def call_llm(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=CUDA_DEVICE)

    model_inputs = tokenizer([prompt], return_tensors="pt").to(CUDA_DEVICE)

    generated_ids = model.generate(model_inputs.input_ids,
                                   max_new_tokens=512,
                                   attention_mask=attention_mask,
                                   pad_token_id=tokenizer.eos_token_id)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    match = re.search(r'{[^a-zA-Z]*}', response, re.DOTALL)

    if match:
        content = match.group()
        # 转换为列表
        parsed_data = json.loads(content)
    else:
        raise Exception("can not find json str.")


    torch_gc()

    return parsed_data

def get_gpu_usage(device=None):
    r"""Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return "{:.2f} G/{:.2f} G".format(reserved, total)

# 读取 JSON 文件
with open(prompt_files, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 存储结果的列表
results = []
need_retry_count = 0
total_retry_count = 0
failed_user = []
# 解析 JSON 列表
for element in tqdm(data,
                    desc=set_color("GPU RAM: " + get_gpu_usage(CUDA_DEVICE), "yellow"),
                    unit="user"):
    recommendation_text = element['recommendation_text']
    retry_count = 0  # 当前重试次数
    probability_list = ''
    if use_LLM:
        while retry_count < max_retries:
            try:
                probability_list = call_llm(recommendation_text)
                break  # 成功时退出循环
            except Exception as e:
                retry_count += 1
                time.sleep(0.5)  # 等待一段时间再重试
        need_retry_count += 1 if retry_count > 0 else 0
        total_retry_count += retry_count

    # if call llm arrive MAX LIMIT nums, and it still failed, use default recommend result
    if probability_list == '':
        failed_user.append(element['user_id'])
        d = dict()
        score_upper_bound = 100
        for item_id in element['item_id_list'].split(","):
            d[str(item_id)] = score_upper_bound
            score_upper_bound = score_upper_bound - 1
        probability_list = d
    # 将结果存储到列表中
    results.append({'user_id': element['user_id'],
                    'predict_score': probability_list})

# 创建 DataFrame
df = pd.DataFrame(results)

# 写入 CSV 文件
df.to_csv(f'{prompt_files}_{model_name}_result_{"llm" if use_LLM else "ori"}.csv',  sep='\t', index=False, encoding='utf-8')
if need_retry_count != 0:
    print(f"finished! need_retry_count: {need_retry_count}, average_retry_count: {total_retry_count / need_retry_count}")

print(f"failed user list {str(failed_user)}")