# run_batch.py
import os, json, time, uuid, csv, pathlib
from openai import OpenAI
import pandas as pd

import re
from pathlib import Path
from tqdm import tqdm
import argparse  # 新增
from math import ceil

def parse_args():
    # 替换原有参数定义：
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='o1',
                        help='Model name (e.g., gpt-4o, gpt-4o1, etc.)')
    parser.add_argument('--split-id',
                            type=int,
                            help='Number of splits for distributed inference.')
    parser.add_argument('--num-chunk',
                            type=int,
                            help='Total splits for distributed inference.')
    return parser.parse_args()


### 1. 参数区 —— 按需修改 #############################################
MODEL          = "gpt-4o"        # 📌 保持与你的原设置一致
MAX_TOKENS_OUT = 5120            # 📌 调整为你的5120设置
TEMP           = 0.7            # 📌 调整为你的高温设置

REQUESTS_JSONL = "./uploads/{:}/requests_chunk{:}.jsonl"
RESULT_NDJSON  = "./results/{:}_results/output_chunk{:}.ndjson"
RESULT_JSON = "./results/{:}_results/Evaluation-Chunk{:}.json"

SYSTEM_PROMPT  = """
You are a reasoning evaluator designed to assess the alignment, coherence, and quality of reasoning steps in text responses. Your task is to evaluate reasoning steps between the * ground truth * and the * LLM response * using the following metrics:

1. ** Faithfulness (1 - 10) :**
    - Definition : Measures how well the reasoning steps in the LLM response align with the source reasoning steps .
    - Scoring Guidelines :
        - 9 - 10: All or almost all steps match or closely reflect the ground truth reasoning.
        - 7 - 8: Most steps are aligned , with minor deviations.
        - 5 - 6: Some steps align , but several are missing or significantly altered.
        - 3 - 4: Few steps align correctly ; most are off or missing .
        - 1 - 2: The majority of steps are not aligned with the source .

2. ** Informativeness ( Info - Step ) (1 - 10) :**
    - Definition : Measures how well the reasoning steps extract all relevant information from the source .
    - Scoring Guidelines :
        - 9 - 10: Almost all critical information steps are present and accurate .
        - 7 - 8: Most important points are included , with minor omissions .
        - 5 - 6: Some key information is missing or underdeveloped .
        - 3 - 4: Limited inclusion of critical content .
        - 1 - 2: Very poor extraction of relevant information .

3. ** Repetition and Redundancy (1 - 10) :**
    - Definition : Identifies repeated or unnecessarily paraphrased reasoning steps within the hypothesis or redundant reasoning steps that do not add value.
    - Scoring Guidelines :
        - 9 -10: No or minimal unnecessary repetition and redundancy.
        - 7 -8: Minor repetition or redundancy that doesn ' t impede clarity .
        - 5 -6: Noticeable repetition or redundancy that doesn ' t add value .
        - 3 -4: Frequent repetition or redundancy that disrupts coherence .
        - 1 -2: Excessive repetition or redundancy reducing the quality of reasoning .

4. ** Hallucination (1 - 10) :**
    - Definition : Detect irrelevant or invented reasoning steps not aligned with the source .
    - Scoring Guidelines :
        - 9 - 10: No hallucinations ; all reasoning is grounded in the source .
        - 7 - 8: One or two minor hallucinations .
        - 5 - 6: Several steps contain invented or irrelevant details .
        - 3 - 4: Many hallucinations , but some grounding remains .
        - 1 - 2: Mostly hallucinated reasoning .

5. ** Missing Step (1 -10) :**
    - Definition : Identify if any necessary reasoning steps are missing .
    - Scoring Guidelines :
        - 9 - 10: No critical steps missing .
        - 7 - 8: Minor missing steps that don ' t significantly affect
        the conclusion .
        - 5 - 6: Some important steps absent , affecting the outcome .
        - 3 - 4: Several crucial missing steps .
        - 1 - 2: Major gaps ; the reasoning chain is incomplete .
    
** Additional Instructions for Consistency :**
    - Always follow the above scoring guidelines strictly .
    - Before scoring , re - read both the ground truth and the LLM response carefully .
    - Compare the reasoning steps directly to determine where they align or diverge .
    - Use the provided scoring benchmarks ( anchor examples , if any ) as a reference to maintain consistency across evaluations .
    - Avoid subjective interpretation and adhere to the given thresholds .
    - Once scores for all metrics are determined , compute the Overall Score as the average of all metric scores .
    - Provide the final output as a Python dictionary with the structure only don ' t add a anything extra , beacuase your out will be used in code pipeline . So single change in you output will crash whole system . :
        # Example output : { 'Faithfulness ': 8.0 , 'Informativeness': 8.5 , 'Repetition&Redundancy': 9.0 , 'Hallucination': 9.5 , 'Missing': 8.5 , 'Overall': 8.65}
"""  # 📌 保持原SYSTEM_MESSAGE内容

#######################################################################

client = OpenAI(
    api_key="",
    base_url="")


def build_jsonl(args):

    id_mapping = {}
    df_gt = pd.read_parquet("./dataset/", engine="pyarrow")
    gt_indexer = {df_gt['question'][i]: df_gt['solution'][i] for i in range(len(df_gt))}

    if args.model == 'o1':
        model_path = "./prediction/o1"
        files = [os.path.join(model_path, path) for path in os.listdir(model_path)]
        files_list = [pd.read_json(f, lines=True) for f in files]
        df_model = pd.concat(files_list, ignore_index=True)
    elif args.model == 'gemini':
        model_path = "./prediction/gemini"
        files = [os.path.join(model_path, path) for path in os.listdir(model_path)]
        files_list = [pd.read_json(f, lines=True) for f in files]
        df_model = pd.concat(files_list, ignore_index=True)
    elif args.model == 'gpt-4o':
        model_path = "./prediction/gpt-4o"
        files = [os.path.join(model_path, path) for path in os.listdir(model_path)]
        files_list = [pd.read_json(f, lines=True) for f in files]
        df_model = pd.concat(files_list, ignore_index=True)

    elif args.model == 'Qwen2.5-VL-72B-Instruct':
        model_path = "./prediction/Qwen2.5-VL-72B-Instruct"
        files = [os.path.join(model_path, path) for path in os.listdir(model_path)]
        files_list = [pd.read_json(f, lines=True) for f in files]
        df_model = pd.concat(files_list, ignore_index=True)

    elif args.model == 'Qwen2.5-VL-7B-Instruct':
        model_path = "./prediction/Qwen2.5-VL-7B-Instruct"
        files = [os.path.join(model_path, path) for path in os.listdir(model_path)]
        files_list = [pd.read_json(f, lines=True) for f in files]
        df_model = pd.concat(files_list, ignore_index=True)

    elif args.model == 'o4-mini':
        model_path = "./prediction/o4-mini"
        files = [os.path.join(model_path, path) for path in os.listdir(model_path)]
        files_list = [pd.read_json(f, lines=True) for f in files]
        df_model = pd.concat(files_list, ignore_index=True)
    
    elif args.model == 'llava-next-7b':
        model_path = "./prediction/llava-next-7b"
        files = [os.path.join(model_path, path) for path in os.listdir(model_path)]
        files_list = [pd.read_json(f, lines=True) for f in files]
        df_model = pd.concat(files_list, ignore_index=True)

    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    data = []

    pid = df_model['question_id'].tolist()
    questions = df_model['prompt'].tolist()
    responses = df_model['response'].tolist()
    answers = df_model['answer'].tolist()

    data = []
    pattern = r'\\boxed\{\}.*?\n(.*?)(?:\nChoices:|\Z)'
    for idx, item in enumerate(questions):
        match = re.search(pattern, item, re.DOTALL)
        try:
            question = match.group(1)
            gt_reason = gt_indexer[question]
            assert gt_reason != ''
            new_item = dict(
                question_id=pid[idx],
                question=question,
                gt_reason=gt_reason,
                response=responses[idx],
            )
            data.append(new_item)
        except Exception as e:
            pass

    total_items = len(data)
    chunk_size = ceil(total_items / args.num_chunk)
    
    start = args.split_id * chunk_size
    end = min(start + chunk_size, total_items)
    split_data = data[start:end]
    
    with open(REQUESTS_JSONL.format(args.model, args.split_id), "w", encoding="utf-8") as fout:
        for idx, item in enumerate(tqdm(split_data)):
            uid = str(uuid.uuid4())
            id_mapping[uid] = idx  # 📌 记录映射关系
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

            messages.append({"role": "user", "content": f"* ground truth *: {item['gt_reason']}\n* LLM response *: {item['response']}"})
            
            body = {
                "model": MODEL,
                "temperature": TEMP,
                "max_tokens": MAX_TOKENS_OUT,
                "messages": messages
            }
            req = {
                "custom_id": uid,
                "method":   "POST",
                "url":      "/v1/chat/completions",
                "body":     body
            }
            fout.write(json.dumps(req, ensure_ascii=False) + "\n")
    
    return data, id_mapping, start, end

def process_results(original_data, id_mapping, args, start, end):
    """📌 新增：处理结果并更新原文件"""

    original_data = original_data[start:end]

    # 2. 解析批处理结果
    updated_count = 0
    hit_idx = []
    with open(RESULT_NDJSON.format(args.model, args.split_id), 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']
            response = result['response']['body']
        
            # 3. 找到对应的原始条目
            if custom_id in id_mapping:
                idx = id_mapping[custom_id]
                try:
                    # 4. 更新process字段
                    original_data[idx]['score'] = response['choices'][0]['message']['content'].strip()
                    updated_count += 1
                    hit_idx.append(idx)
                except KeyError as e:
                    print(f"处理{custom_id}时出错: {e}")
            
    original_data = [original_data[i] for i in range(len(original_data)) if i in hit_idx]
    
    with open(RESULT_JSON.format(args.model, args.split_id), 'w', encoding='utf-8') as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)


def submit_batch(args):
    print("② 上传文件 ...")
    batch_file = client.files.create(
        file=open(REQUESTS_JSONL.format(args.model, args.split_id), "rb"),
        purpose="batch"
    )
    print("   file_id =", batch_file.id)

    print("③ 创建 Batch Job ...")
    batch = client.batches.create(
        input_file_id  = batch_file.id,
        endpoint       = "/v1/chat/completions",
        completion_window = "24h"          # 24 小时内保证处理完
    )
    print("   batch_id =", batch.id)
    return batch.id

def wait_for_batch(batch_id, interval=10):
    while True:
        batch = client.batches.retrieve(batch_id)
        print(time.strftime("%H:%M:%S"), "status:", batch.status, end="\r")
        if batch.status in {"completed", "failed", "expired", "cancelled"}:
            print()       # 换行
            return batch
        time.sleep(interval)

def download_results(batch):
    if batch.status != "completed":
        raise RuntimeError(f"Batch not successful: {batch.status}")

    out_file_id = batch.output_file_id
    print("④ 下载结果文件 ...")
    content = client.files.content(out_file_id)
    with open(RESULT_NDJSON.format(args.model, args.split_id), "wb") as f:
        for chunk in content.iter_bytes():
            f.write(chunk)
    print(f"✅ 已保存到 {RESULT_NDJSON.format(args.model, args.split_id)}")


if __name__ == "__main__":

    args = parse_args()

    # 📌 修改主流程
    data, id_mapping, start, end = build_jsonl(args)  # 📌 获取映射关系
    
    # 2. 提交批处理任务
    bid = submit_batch(args)
    
    # 3. 等待任务完成
    info = wait_for_batch(bid)
    
    # 4. 下载结果
    download_results(info)
    
    # 5. 处理结果并更新原文件
    process_results(data, id_mapping, args, start, end)  # 📌 新增结果处理
    
    print("🎉 批处理流程完成")