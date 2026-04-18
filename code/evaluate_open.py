import json
import requests
import time
from openai import OpenAI
import os
import pandas as pd
from tqdm import tqdm
import argparse

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

eval_model = "Qwen2.5-72B-Instruct"
tested_model = 'InternVL3-8B-Instruct'


def geninput(example):

    judge = "You are given a response from a model and the correct answer. " + \
        "Your task is to determine if the model's response is correct. " + \
        "You should only return 'true' if the response matches the answer. " + \
        "If the answer is a floating-point number greater than 1, when it is represented in scientific notation, " + \
        "a difference of up to 0.1 is allowed. Otherwise, return 'false'.\n" + \
        f"Response: {example['response']}\n" + \
        f"Correct Answer: {example['answer']}\n" + \
        "Is the response correct? (true/false)"
    return judge


# Function to get model response from OpenAI
def get_model_response(query, max_retries=2, timeout=10):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
            ],
        }
    ]

    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=eval_model,
                messages=messages,
            )
            return response.choices[0].message.content
        except (requests.exceptions.RequestException, Exception) as e:
            if isinstance(e, requests.exceptions.Timeout):
                # Handle timeout specifically
                print(f"Request timed out: {e}. Retrying {retries + 1}/{max_retries}...")
            else:
                # Handle other types of errors
                print(f"Error occurred: {e}. Retrying {retries + 1}/{max_retries}...")
            retries += 1
            time.sleep(2)  # Wait for 2 seconds before retrying

    print(f"Failed to get model response after {max_retries} retries.")
    return None  # Return None if all retries fail


def main(data_path, output_path):
    os.makedirs(output_path+'/'+tested_model, exist_ok=True)
        
    for subject in ['math', 'physics', 'chemistry', 'biology', 'geography', 'astronomy']:
        json_file = os.path.join(data_path, subject+'_results.jsonl')
        df = pd.read_json(json_file, lines=True)

        output = os.path.join(output_path + '/' + tested_model, subject+'_results.jsonl')
        ans_file = open(output, 'w')

        for i in tqdm(range(len(df))):
            example = df.iloc[i]
            pid = example['question_id']
            query = geninput(example)

            response = get_model_response(query, max_retries=2)
            print(response)
            if "true" in response.lower() or "correct" in response.lower():
                correct = True
            elif "false" in response.lower() or "incorrect" in response.lower():
                correct = False
            else:
                print("not judged")

            ans_file.write(json.dumps({"question_id": str(pid),
                                       "prompt": example['prompt'],
                                       "response": example['response'],
                                       "choices": example['choices'],
                                       "answer": example['answer'],
                                       "model_id": tested_model,
                                       "metadata": {},
                                       "correct": correct}) + "\n")
            ans_file.flush()
        ans_file.close()
    return
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='output')
    args = parser.parse_args()

    main(args.data_path, args.output_path)
