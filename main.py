from parse import parse_args
from tqdm import tqdm
from functools import partial
import json
import pickle as pkl
import os
from vllm import LLM
from vllm.lora.request import LoRARequest
from retriever import *
import uuid
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

Convert_dict = {"hotpotqa":"HotpotQA", "musique": "MuSiQue", "2wikimqa": "2WikiMQA"}

def get_mac_address():
    mac = uuid.getnode()
    mac_hex = ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
    return mac_hex

def run(i_d, retriever, test_dataset):
    i, d, G = i_d
    corpus = retriever.retrieve(d, G)

    if corpus == None:
        if test_dataset.lower() == 'fanoutqa':
            return i, d['question'], d['answer'], 'System mistake', corpus, None
        else:
            return i, d['question'], d['answer'], 'System mistake', corpus, d['supports']
    try:
        if test_dataset.lower() == 'fanoutqa':
            return i, d['question'], d['answer'], None, corpus, None,
        else:
            return i, d['question'], d['answer'], None, corpus, d['supports']
    except:
        if test_dataset.lower() == 'fanoutqa':
            return i, d['question'], d['answer'], 'System mistake', corpus, None
        else:
            return i, d['question'], d['answer'], 'System mistake', corpus, d['supports']


if __name__ == '__main__':
    args = parse_args()
    args.path = os.getcwd()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists('./result_ours_{}/{}/{}'.format(args.version, args.llm, args.test_dataset)):
        os.makedirs('./result_ours_{}/{}/{}'.format(args.version, args.llm, args.test_dataset))
    gpu_memory_utilization_main = args.gpu_memory_utilization

    if args.llm == 'llama':
        llm_type = 'Meta-Llama-3.1-8B-Instruct'
    elif args.llm == 'mistral':
        llm_type = 'Mistral-7B-Instruct-v0.3'
    elif args.llm == 'qwen':
        llm_type = 'Qwen2.5-7B-Instruct'


    lora_path = 'lora/{}_{}/checkpoint-2500'.format(args.llm, args.train_dataset)
    lora_request = LoRARequest("self_adapter_v1", 1, lora_path=lora_path)

    tensor_parallel_size = args.gpu.count(",") + 1
    llm = LLM(model=f"{llm_type}",enable_lora=True, gpu_memory_utilization=gpu_memory_utilization_main, tensor_parallel_size=tensor_parallel_size)

    if args.test_dataset.lower() == 'fanoutqa':
        with open('data/{}/test_docs.json'.format(args.test_dataset), 'r') as f:
            test_data = json.load(f)
        test_Gs = pkl.load(open('data/{}/{}_test_long.pkl'.format(args.test_dataset, args.kg), 'rb'))
    elif  args.test_dataset.lower() == 'longhopqa':
        with open('data/{}/test_docs_long_musique_3+4.json'.format(args.test_dataset), 'r') as f:
            test_data = json.load(f)
        test_Gs = pkl.load(open('data/longhopqa/{}_test_long_musique_3+4.pkl'.format(args.kg), 'rb'))
    elif args.test_dataset.lower() in ['hotpotqa', '2wikimqa', 'musique']:
        with open('data/{}/test_docs_short.json'.format(Convert_dict[args.test_dataset]), 'r') as f:
            test_data = json.load(f)
        test_Gs = pkl.load(open('data/{}/{}_test_short.pkl'.format(Convert_dict[args.test_dataset], args.kg), 'rb'))

    with open('generate_demonstrations/{}/{}/sequential_parallel_{}.txt'.format(args.llm, args.train_dataset, args.sample_num), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sequential_parallel = ''
    for line in lines:
        sequential_parallel += line

    with open('generate_demonstrations/{}/{}/sequential_with_evidence_{}.txt'.format(args.llm, args.train_dataset, args.sample_num), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sequential_prompt_with_evidence = ''
    for line in lines:
        sequential_prompt_with_evidence += line

    with open('generate_demonstrations/{}/{}/sequential_without_evidence_{}.txt'.format(args.llm, args.train_dataset, args.sample_num), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sequential_prompt_without_evidence = ''
    for line in lines:
        sequential_prompt_without_evidence += line

    with open('generate_demonstrations/{}/{}/parallel_{}.txt'.format(args.llm, args.train_dataset, args.sample_num), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    parallel_prompt = ''
    for line in lines:
        parallel_prompt += line

    retriever = llm_retriever_ours(args, args.k, args.k_nei, llm, lora_request, args.max_dep, sequential_parallel, sequential_prompt_with_evidence, sequential_prompt_without_evidence, parallel_prompt)

    func = partial(run, retriever=retriever, test_dataset=args.train_dataset)
    val_file = './result_ours_{}/{}/{}/step:{}-max_dep:{}-val.json'.format(args.version, args.llm, args.train_dataset, args.step, args.max_dep)
    if not os.path.exists(val_file) and args.do_eval:
        with open('data/{}/val_docs_short.json'.format(Convert_dict[args.train_dataset]), 'r') as f:
            val_data = json.load(f)[:100]
        val_Gs = pkl.load(
            open('data/{}/{}_val_short.pkl'.format(Convert_dict[args.train_dataset], args.kg), 'rb'))
        val_data_idx = [(i, d, val_Gs[i]) for i, d in enumerate(val_data)]
        val_res = []
        for idx in tqdm(val_data_idx):
            i, question, answer, pred, corpus, supports = func(i_d=idx)
            val_res.append({'idx': i,
                        'question': question,
                        'answer': answer,
                        'prediction': pred,
                        'grade': None,
                        'corpus': corpus,
                        'supports': supports
                        })
        if not os.path.exists('./result_ours_{}/{}/{}'.format(args.version, args.llm, args.train_dataset)):
            os.makedirs('./result_ours_{}/{}/{}'.format(args.version, args.llm, args.train_dataset))
        json.dump(val_res, open(val_file, 'w'))

    func = partial(run, retriever=retriever, test_dataset=args.test_dataset)
    test_file = './result_ours_{}/{}/{}/train:{}-step:{}-max_dep:{}-test.json'.format(args.version, args.llm, args.test_dataset, args.train_dataset, args.step, args.max_dep)
    test_data_idx = [(i, d, test_Gs[i]) for i, d in enumerate(test_data)]
    test_res = []
    for idx in tqdm(test_data_idx):
        i, question, answer, pred, corpus, supports = func(i_d=idx)
        test_res.append({'idx': i,
                        'question': question,
                        'answer': answer,
                        'prediction': pred,
                        'grade': None,
                        'corpus': corpus,
                        'supports': supports
                        })

    json.dump(test_res, open(test_file, 'w'))