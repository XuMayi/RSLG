import os
from parse import parse_args
import json
from vllm import LLM
import uuid
from llm_runner import llm_runner_chat2
import math

Convert_dict = {"hotpotqa":"HotpotQA", "musique": "MuSiQue", "2wikimqa": "2WikiMQA"}

def is_only_one_true(condition1, condition2, condition3):
    return (condition1 and not condition2 and not condition3) or (
        not condition1 and condition2 and not condition3) or (
        not condition1 and not condition2 and condition3)
def find_nth_percentile_value(lst):
    sorted_desc = sorted(lst)
    n = len(sorted_desc)
    index = int(n * 0.1)
    return sorted_desc[index]

def calculate_entropy(dictionary):
    total = sum(dictionary.values())
    entropy = 0
    for value in dictionary.values():
        probability = value / total
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy

def get_mac_address():
    mac = uuid.getnode()
    mac_hex = ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
    return mac_hex

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    inputs = []
    with open('../data/longhopqa/{}_train_docs_short.json'.format(Convert_dict[args.train_dataset]), 'r') as f:
        datas = json.load(f)


    with open('../prompt/sequential_parallel.txt','r') as f:
        lines = f.readlines()

    prompt = ''
    for line in lines:
        prompt += line

    questions = []
    inputs = []
    for i, data in enumerate(datas):
        question = data['question']
        questions.append(question)
        input = prompt + ' ' + question
        inputs.append(input)

    if args.llm == 'llama':
        llm_type = 'Meta-Llama-3.1-8B-Instruct'
    elif args.llm == 'mistral':
        llm_type = 'Mistral-7B-Instruct-v0.3'
    elif args.llm == 'qwen':
        llm_type = 'Qwen2.5-7B-Instruct'

    tensor_parallel_size = args.gpu.count(",") + 1
    llm = LLM(model=f"{llm_type}", enable_lora=False, gpu_memory_utilization=0.8, tensor_parallel_size=tensor_parallel_size)

    preds_all = llm_runner_chat2(llm, inputs, 1, 1024, 10)

    one_hop_questions = []
    parallel_multi_hop_questions = []
    sequential_multi_hop_questions = []


    question_pred_entropy = []
    entropy_dicts = []
    entropies = []
    for question, preds in zip(questions, preds_all):
        entropy_dict = {'parallel multi-hop':0, 'sequential multi-hop':0, 'one_hop':0, 'other':0}
        for pred in preds:
            if is_only_one_true('parallel multi-hop' in pred.lower(), 'sequential multi-hop' in pred.lower(), 'one-hop' in pred.lower()):
                if 'parallel multi-hop' in pred.lower():
                    entropy_dict['parallel multi-hop'] += 1
                elif 'sequential multi-hop' in pred.lower():
                    entropy_dict['sequential multi-hop'] += 1
                elif 'one-hop' in pred.lower():
                    entropy_dict['one_hop'] += 1
            else:
                entropy_dict['other'] += 1
        max_key = max(entropy_dict, key=entropy_dict.get)
        entropy = calculate_entropy(entropy_dict)
        entropies.append(entropy)
        entropy_dict['question'] = question
        entropy_dict['preds'] = preds
        entropy_dicts.append(entropy_dict)
        question_pred_entropy.append([question, entropy, max_key])
        print(entropy_dict)

    if not os.path.exists('{}/{}'.format(args.llm, args.train_dataset)):
        os.makedirs('{}/{}'.format(args.llm, args.train_dataset))

    with open('{}/{}/entropys_dcit.json'.format(args.llm, args.train_dataset), 'w', encoding='utf-8') as f:
        json.dump(entropy_dicts, f, indent=4)

    parallel_questions = []
    parallel_entropies = []
    sequential_questions = []
    sequential_entropies = []
    one_hop_questions = []
    one_hop_entropies = []
    for i, item in enumerate(question_pred_entropy):
        if item[2] == 'parallel multi-hop':
            parallel_questions.append(item[0])
            parallel_entropies.append(entropies[i])
        elif item[2] == 'sequential multi-hop':
            sequential_questions.append(item[0])
            sequential_entropies.append(entropies[i])

    parallel_questions_filter = []
    parallel_value = find_nth_percentile_value(parallel_entropies)
    sequential_questions_filter = []
    sequential_value = find_nth_percentile_value(sequential_entropies)
    for question, entropy in zip(parallel_questions, parallel_entropies):
        if entropy <= parallel_value:
            parallel_questions_filter.append(question)
    for question, entropy in zip(sequential_questions, sequential_entropies):
        if entropy <= sequential_value:
            sequential_questions_filter.append(question)

    with open('../prompt/sequential_with_evidence.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sequential_prompt_with_evidence = ''
    for line in lines:
        sequential_prompt_with_evidence += line

    inputs_all = []
    for question in sequential_questions_filter:
        inputs_all.append(sequential_prompt_with_evidence + ' ' + question)
    preds = llm_runner_chat2(llm, inputs_all, 0, 1024, 1, stop='Question:')

    sequential_dict = {}
    for question, pred in zip(sequential_questions_filter, preds):
        if 'sequential multi-hop' in pred and 'parallel multi-hop' not in pred and 'one_hop' not in pred:
            sequential_dict[question] = pred.replace('\n\n', '\n')

    with open('{}/{}/sequential_questions_dict.json'.format(args.llm, args.train_dataset), 'w', encoding='utf-8') as f:
        json.dump(sequential_dict, f, indent=4)

    with open('../prompt/parallel.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    parallel_prompt = ''
    for line in lines:
        parallel_prompt += line

    inputs_all = []
    for question in parallel_questions:
        inputs_all.append(parallel_prompt + ' ' + question)
    preds = llm_runner_chat2(llm, inputs_all, 0, 1024, 1, stop='Question:')

    parallel_dict = {}
    for question, pred in zip(parallel_questions, preds):
        if 'sequential multi-hop' not in pred and 'parallel multi-hop' in pred and 'one_hop' not in pred:
            parallel_dict[question] = pred.replace('\n\n', '\n')

    with open('{}/{}/parallel_questions_dict.json'.format(args.llm, args.train_dataset), 'w', encoding='utf-8') as f:
        json.dump(parallel_dict, f, indent=4)
