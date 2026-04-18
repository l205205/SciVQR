from tqdm import tqdm
import json
from utils import save_jsonl, load_jsonl, find_math_answer, is_equal, is_number
import os
import argparse
from latex2sympy2 import latex2sympy
from sympy import Integer, Rational


def evaluate(answer_file, regen_answer):
    lines = load_jsonl(answer_file)
    for line in tqdm(lines, desc='gen_correct'):

        gt_answer_value = line['answer']
        if len(line["choices"]) > 0:
            sequential_characters = [chr(ord('A') + i) for i in range(len(line["choices"]))]
            try:
                gt_answer = sequential_characters[line["choices"].index(gt_answer_value)]
            except:
                gt_answer = ''
        else:
            gt_answer = ''

        if 'model_answer' not in line or regen_answer:
            model_answer = line['response'].strip()
            for c in 'ABCDE':
                if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(
                        f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n") \
                        or model_answer.endswith(f"\\{c}") or model_answer.endswith(f":{c}"):
                    model_answer = c
            if is_number(model_answer.split('is ')[-1].rstrip('.')):
                model_answer = model_answer.split('is ')[-1].rstrip('.')
            if 'oxed{' not in model_answer:
                if len(line["choices"]) > 0:
                    for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
                        raw_model_answer = model_answer
                        model_answer = model_answer.split(flag)[-1].strip()
                        if flag in raw_model_answer:
                            if ':\n\n' in model_answer :
                                model_answer = model_answer.split(':\n\n')[1].split('. ')[0]
                                for c in 'ABCDE':
                                    if (model_answer.startswith(f"{c}") or f"\\{c}" in model_answer or f"/{c}"
                                        in model_answer or f"({c})" in model_answer or f"*{c}" in model_answer
                                        or f":{c}" in model_answer or f"box{{{c}}}" in model_answer
                                        or f"ircled{c}" in model_answer or f"ircle{{{c}}}" in model_answer
                                        or f"\\u{c}" in model_answer) :
                                        model_answer = c

                            elif ':\n' in model_answer:
                                if (model_answer.startswith(f"{c}") or f"\\{c}" in model_answer or f"/{c}"
                                    in model_answer or f"({c})" in model_answer or f"*{c}" in model_answer
                                    or f":{c}" in model_answer or f"box{{{c}}}" in model_answer
                                    or f"ircled{c}" in model_answer or f"ircle{{{c}}}" in model_answer
                                    or f"\\u{c}" in model_answer) :
                                    model_answer = c

                            else:
                                model_answer = model_answer.split('\n')[0].split('. ')[0]

                        flag = flag.replace('the', 'The')
                        raw_model_answer = model_answer
                        model_answer = model_answer.split(flag)[-1].strip()
                        if flag in raw_model_answer:
                            if ':\n\n' in model_answer and len(line["choices"]) > 0:
                                model_answer = model_answer.split(':\n\n')[1].split('. ')[0]
                                for c in 'ABCDE':
                                    if (model_answer.startswith(f"{c}") or f"\\{c}" in model_answer or f"/{c}"
                                        in model_answer or f"({c})" in model_answer or f"*{c}" in model_answer
                                        or f":{c}" in model_answer or f"box{{{c}}}" in model_answer
                                        or f"ircled{c}" in model_answer or f"ircle{{{c}}}" in model_answer
                                        or f"\\u{c}" in model_answer) \
                                            and len(line["choices"]) > 0:
                                        model_answer = c

                            elif ':\n' in model_answer and len(line["choices"]) > 0:
                                if (model_answer.startswith(f"{c}") or f"\\{c}" in model_answer or f"/{c}"
                                    in model_answer or f"({c})" in model_answer or f"*{c}" in model_answer
                                    or f":{c}" in model_answer or f"box{{{c}}}" in model_answer
                                    or f"ircled{c}" in model_answer or f"ircle{{{c}}}" in model_answer
                                    or f"\\u{c}" in model_answer) \
                                        and len(line["choices"]) > 0:
                                    model_answer = c
                            else:
                                model_answer = model_answer.split('\n')[0].split('. ')[0]
                            # print(model_answer)

                else:
                    for flag in ['the final answer is', 'the answer is', 'the correct answer is',
                                 'the answer should be']:
                        raw_model_answer = model_answer
                        model_answer = model_answer.split(flag)[-1].strip()
                        if flag in raw_model_answer:
                            model_answer = model_answer.split('\n')[0].split('. ')[0]
                        flag = flag.replace('the', 'The')
                        raw_model_answer = model_answer
                        model_answer = model_answer.split(flag)[-1].strip()
                        if flag in raw_model_answer:
                            model_answer = model_answer.split('\n')[0].split('. ')[0]

            elif model_answer.count('oxed{') > 1:
                model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]

            model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)',
                                                                                                          'c').replace(
                '(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace(
                '{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
            line['model_answer'] = model_answer
        else:
            model_answer = line['model_answer']
        if len(line["choices"]) > 0:
            line['correct'] = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
            try:
                if type(latex2sympy(model_answer)) == Integer or type(latex2sympy(model_answer)) == Rational:
                    if model_answer in gt_answer or model_answer in gt_answer_value:
                        line['correct'] = True
            except:
                pass
        else:
            line['correct'] = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)

    save_jsonl(answer_file, lines, t_stamp=False)


def cal_acc(answer_file):
    lines = load_jsonl(answer_file)
    is_correct = 0
    for line in tqdm(lines, desc='math_level_subject_acc'):
        correct = line['correct']
        if correct:
            is_correct += 1
    return is_correct / len(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default=None)
    args = parser.parse_args()

    metrics = {}
    for subject in ['math', 'physics', 'chemistry', 'biology', 'geography', 'astronomy']:
        answer_file = os.path.join(args.result_dir, subject + '_results.jsonl')
        evaluate(answer_file, False)
        metrics[subject] = cal_acc(answer_file)

    with open(os.path.join(args.result_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
