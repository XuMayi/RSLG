from parse import parse_args
import os
import nltk
from vllm import LLM
from llm_runner import llm_runner_chat2
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import uuid
def get_mac_address():
    mac = uuid.getnode()
    mac_hex = ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
    return mac_hex

if __name__ == '__main__':
    args = parse_args()
    args.path = os.getcwd()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from utils import *
    if args.llm == 'llama':
        llm_type = 'Meta-Llama-3.1-8B-Instruct'
    elif args.llm == 'mistral':
        llm_type = 'Mistral-7B-Instruct-v0.3'
    elif args.llm == 'qwen':
        llm_type = 'Qwen2.5-7B-Instruct'

    gpu_memory_utilization = 0.95
    tensor_parallel_size = args.gpu.count(",") + 1
    llm = LLM(model=f"{llm_type}", gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size,
              load_format='pt')


    if args.test_dataset in ['hotpotqa_val', '2wikimqa_val', 'musique_val']:
        file = './result_ours_{}/{}/{}/step:{}-max_dep:{}-val.json'.format(args.version, args.llm, args.test_dataset.split('_val')[0], args.step, args.max_dep)
    else:
        file = './result_ours_{}/{}/{}/train:{}-step:{}-max_dep:{}-test.json'.format(args.version, args.llm, args.test_dataset, args.train_dataset, args.step, args.max_dep)

    datas = json.load(open(file, 'r'))

    res = []
    inputs_all = []

    for data in datas:
        idx, question, answer, corpus, supports, pred = data['idx'], data['question'], data['answer'], data['corpus'], data['supports'], data['prediction']

        res.append({'idx': idx,
                    'question': question,
                    'answer': answer,
                    'prediction': pred,
                    'grade': None,
                    'corpus': corpus,
                    'supports': supports,
                    })

        corpus_step = []
        for item in corpus:
            corpus_step.append('\n\n'.join(item.split('\n\n')[:args.test_step]))
        context = context_joint(corpus_step, args.llm)


        start = "Given the following question and contexts, create a final answer to the question."

        context = start + '\n=========\n' + 'QUESTION: ' + question + '\n=========\n'\
                   + 'CONTEXT:\n' + str(context) + '\n=========\n' + 'QUESTION: ' + question + '\n=========\n'

        if args.test_dataset.lower() != 'fanoutqa':
            context = context + 'ANSWER: please answer less than 6 words.'
        else:
            context = context + 'ANSWER:'
        inputs_all.append(context)

    preds = llm_runner_chat2(llm, inputs_all, 0, 1024)

    for i, pred in enumerate(preds):
        res[i]['prediction'] = pred

    if args.test_dataset in ['hotpotqa_val', '2wikimqa_val', 'musique_val']:
        file = './result_ours_{}/{}/{}/test_step:{}-max_dep:{}-val.json'.format(args.version, args.llm, args.test_dataset.split('_val')[0], args.test_step, args.max_dep)
    else:
        file = './result_ours_{}/{}/{}/train:{}-test_step:{}-max_dep:{}-test.json'.format(args.version, args.llm, args.test_dataset, args.train_dataset, args.test_step, args.max_dep)


    json.dump(res, open(file, 'w'))

