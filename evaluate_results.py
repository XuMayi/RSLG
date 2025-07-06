import os
import json
from functools import partial
from multiprocessing import Pool
from parse import parse_args
from prompt import get_eval_prompt, get_eval_prompt_fan
from tqdm import tqdm
from gpt_runner import basic_runner
from utils import evaluate_ours


def run(data, eval_prompt, llm):
    eval_prompt = eval_prompt.replace('{prediction}', data['prediction']).replace('{question}', data['question']).replace('{answer}', str(data['answer']))
    try:
        get_result, grade, error_msg = basic_runner(llm, eval_prompt, 0, n=1)
        return data['idx'], data['question'], data['answer'], data['prediction'], grade, data['corpus'], data['supports']
    except Exception as e:
        print(e)
        return data['idx'], data['question'], data['answer'], data['prediction'], 'System mistake', data['corpus'], data['supports']

if __name__ == '__main__':
    args = parse_args()
    #llm = "gpt-4o-2024-11-20"
    llm = "gpt-4o-mini-2024-07-18"
    if args.test_dataset == 'fanoutqa':
        eval_prompt = get_eval_prompt_fan()
    else:
        eval_prompt = get_eval_prompt()

    if args.test_dataset in ['hotpotqa_val', '2wikimqa_val', 'musique_val']:
        file_name = './result_ours_{}/{}/{}/test_step:{}-max_dep:{}-val.json'.format(args.version, args.llm, args.test_dataset.split('_val')[0], args.test_step, args.max_dep)
    else:
        file_name = './result_ours_{}/{}/{}/train:{}-test_step:{}-max_dep:{}-test.json'.format(args.version, args.llm, args.test_dataset, args.train_dataset, args.test_step, args.max_dep)

    with open(file_name, 'r') as f:
        res = json.load(f)

    for i, item in enumerate(res):
        res[i]['corpus'] = item['corpus']

    run_partial = partial(run, eval_prompt=eval_prompt,llm=llm)
    pool = Pool(args.n_processes)
    res_with_grade = []

    with tqdm(total=len(res)) as pbar:
        for res_ in pool.imap(run_partial, res):
            idx, question, answer, pred, grade, corpus, supports = res_
            res_with_grade.append({'idx': idx,
                        'question': question,
                        'answer': answer,
                        'prediction': pred,
                        'grade': grade,
                        'corpus': corpus,
                        'supports': supports
                        })
            pbar.update()
    pool.close()
    pool.join()
    new_file_name = file_name.split('.json')[0] + '_grade.json'
    json.dump(res_with_grade, open(new_file_name, 'w'))

    if args.test_dataset != 'fanoutqa':
        acc, f1, retrieval_recall, retrieval_precision, retrieval_em = evaluate_ours(dataset=args.test_dataset, File=new_file_name, k=args.k)
        if args.test_dataset in ['hotpotqa_val', '2wikimqa_val', 'musique_val']:
            log_data = {
                            "Train Dataset": args.test_dataset.split('_val')[0],
                            "Test Dataset": args.test_dataset.split('_val')[0],
                            "Max_dep": args.max_dep,
                            "Accuracy": acc,
                            "F1": f1,
                            "Retrieval recall": retrieval_recall,
                            "Retrieval precision": retrieval_precision,
                            "Retrieval em": retrieval_em
                        }
        else:
            log_data = {
                            "Train Dataset": args.train_dataset,
                            "Test Dataset": args.test_dataset,
                            "Max_dep": args.max_dep,
                            "Accuracy": acc,
                            "F1": f1,
                            "Retrieval recall": retrieval_recall,
                            "Retrieval precision": retrieval_precision,
                            "Retrieval em": retrieval_em
                        }

    else:
        acc, f1 = evaluate_ours(dataset=args.test_dataset, File=new_file_name, k=args.k)
        log_data = {"Train Dataset": args.train_dataset,
                    "Test Dataset": args.test_dataset,
                    "Max_dep": args.max_dep,
                    "Accuracy": acc,
                    "F1": f1
                    }
    if args.test_dataset in ['hotpotqa_val', '2wikimqa_val', 'musique_val']:
        results_file = './result_ours_{}/{}/{}/test_step:{}-max_dep:{}-val-results.json'.format(args.version, args.llm, args.test_dataset.split('_val')[0], args.test_step, args.max_dep)
    else:
        results_file = './result_ours_{}/{}/{}/train:{}-test_step:{}-max_dep:{}-test-results.json'.format(args.version, args.llm, args.test_dataset, args.train_dataset, args.test_step, args.max_dep)

    if not os.path.exists(results_file):
        log_datas = []
        log_datas.append(log_data)
        with open(results_file,'w') as f:
            json.dump(log_datas, f, indent=4)
    else:
        with open(results_file,'r') as f:
            log_datas = json.load(f)
        log_datas.append(log_data)
        with open(results_file,'w') as f:
            json.dump(log_datas, f, indent=4)


