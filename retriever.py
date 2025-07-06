#coding=utf-8

from llm_runner import llm_runner_chat, llm_runner_chat2
from utils import tf_idf
import copy
from collections import Counter
import json
from utils import unique
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import re
import ast
import os

def extract_after_last_marker(pred_questions):
    marker1 = "[\'"
    marker2 = "[\""
    idx1 = pred_questions.rfind(marker1)
    idx2 = pred_questions.rfind(marker2)

    if idx1 == -1 and idx2 == -1:
        return None
    elif idx1 > idx2:
        return pred_questions[idx1:]
    else:
        return pred_questions[idx2:]

def find_duplicate_arrays_ignore_order(array_of_arrays):
    array_tuples = [tuple(sorted(arr)) for arr in array_of_arrays]
    counts = Counter(array_tuples)
    duplicates = [(list(arr), count) for arr, count in counts.items() if count > 1]
    return duplicates

class llm_retriever_ours(object):
    def __init__(self, args, k, k_nei, llm, lora, max_dep, sequential_parallel, sequential_prompt_with_evidence, sequential_prompt_without_evidence, parallel_prompt):
        self.k = k
        self.k_nei = k_nei
        self.llm = llm
        self.lora = lora
        self.max_dep = max_dep
        self.sequential_parallel = sequential_parallel
        self.sequential_prompt_with_evidence = sequential_prompt_with_evidence
        self.sequential_prompt_without_evidence = sequential_prompt_without_evidence
        self.parallel_prompt = parallel_prompt
        self.args = args
        self.encoder = SentenceTransformer("all-mpnet-base-v2")
        self.extract_questions_prompt = ''
        with open('prompt/extract_question.txt', 'r') as f:
            lines = f.readlines()
        for line in lines:
            self.extract_questions_prompt += line

    def decompose(self, data, question, G, dep, existing_emb, ori_emb):
        if 'question' in question.lower():
            return []
        if dep == self.max_dep:
            return [question]
        print('\n---------------------------')
        print("dep:{}".format(dep))
        print('----------------------------\n')

        input = self.sequential_parallel + ' ' + question
        print('\n------------sequential parallel input---------------')
        print(input)
        print('----------------------------\n')
        flag_temperature = 0
        while True:
            if flag_temperature == 1:
                print('\n---------------------------')
                print('failed start')
                print('----------------------------\n')
            pred = llm_runner_chat2(self.llm, [input], flag_temperature, 1024, stop='Question:')[0]
            flag_temperature = 1

            if dep != 1 and 'one-hop' in pred.lower() and 'parallel' not in pred.lower() and 'sequential' not in pred.lower():
                print('\n---------------------------')
                print('on-hop question: {}'.format(question))
                print('----------------------------\n')
                return [question]

            elif 'one-hop' not in pred.lower() and 'parallel' in pred.lower() and 'sequential' not in pred.lower():
                print('\n---------------------------')
                print('parallel question: {}'.format(question))
                print('----------------------------\n')

                question_all = [question]

                input = self.parallel_prompt + ' ' + question
                print('\n------------parallel input---------------')
                print(input)
                print('----------------------------\n')

                pred = llm_runner_chat2(self.llm, [input], 0, 1024)[0]
                print('\n------------parallel pred---------------')
                print(pred)
                print('----------------------------\n')

                input = self.extract_questions_prompt + ' ' + pred.replace('\n\n','\n') + '\nOutput:'
                print('\n------------parallel extract input---------------')
                print(input)
                print('----------------------------\n')

                flag_extract = 0
                while True:
                    pred_questions = llm_runner_chat2(self.llm, [input], flag_extract, 1024)[0]
                    flag_extract = 1
                    if '[\'' in pred_questions or '[\"' in pred_questions:
                        pred_questions = extract_after_last_marker(pred_questions)
                        print('\n------------------parallel pred_questions-------------------')
                        print(pred_questions)
                        print('------------------parallel pred_questions-------------------\n')
                        try:
                            questions = re.search(r"\[.*\]", pred_questions).group()
                            questions = questions.replace('\"',"\'").replace('[\'', '[\"').replace('\', \'', '\", \"').replace('\']', '\"]').replace('\',\n', '\',')
                            questions = ast.literal_eval(questions)
                            for i, item in enumerate(questions):
                                if ":" in item:
                                    questions[i]= item.split(":")[1].strip()
                            break
                        except:
                            print('\n----------------------------')
                            print('extract error!')
                            print('----------------------------\n')
                            with open('result_ours_{}/{}/{}/error_extract_{}_{}_{}.txt'.format(self.args.version, self.args.llm,
                                                                                               self.args.test_dataset,
                                                                                               self.args.train_dataset,
                                                                                               self.args.step,
                                                                                               self.args.sample_num), 'a',
                                      encoding='=utf-8') as f:
                                f.write(pred_questions + '\n\n\n')
                            if dep == 1:
                                print('\n----------------------------')
                                print('continue when dep is 1!')
                                print('----------------------------\n')
                                continue

                print('\n------------------parallel questions-------------------')
                print(questions)
                print('------------------parallel questions-------------------\n')

                if questions == []:
                    return [question]
                if len(questions) > 10:
                    questions = questions[:10]

                embeddings = self.encoder.encode(questions)
                for i, q_emb in enumerate(embeddings):
                    flag = 0
                    temp_emb = copy.deepcopy(existing_emb)
                    if dep != 1:
                        if (1 - cosine(ori_emb, q_emb)) < 0.5:
                            flag = 1
                            print('\n-------------------------------------')
                            print('the question is far from the original question:{}'.format(questions[i]))
                            print('-------------------------------------\n')
                        elif "question" in questions[i].lower():
                            flag = 1
                            print('\n-------------------------------------')
                            print('the question is not context clear:{}'.format(questions[i]))
                            print('-------------------------------------\n')
                        else:
                            for emb in existing_emb:
                                cos_sim = 1 - cosine(q_emb, emb)
                                if cos_sim > 0.9:
                                    flag = 1
                                    print('\n-------------------------------------')
                                    print('there are similar question with:{}'.format(questions[i]))
                                    print('-------------------------------------\n')
                                    break
                    if flag == 0:
                        temp_emb.append(q_emb)
                        sub_question_all = self.decompose(data, questions[i], G, dep + 1, temp_emb, ori_emb)
                        question_all.extend(sub_question_all)
                        print('question:{}'.format(questions[i]))
                        print('subquestion of question:{}'.format(sub_question_all))

                return question_all

            elif 'one-hop' not in pred.lower() and 'parallel' not in pred.lower() and 'sequential' in pred.lower():
                print('\n---------------------------')
                print('sequential question: {}'.format(question))
                print('----------------------------\n')

                if 'Sub-question 1:' in pred:
                    question_all = [question]
                    question1 = pred.split('Sub-question 1:')[1].split('\n')[0].strip()
                    print('\n---------------------------')
                    print('sub-question 1: {}'.format(question1))
                    print('----------------------------\n')
                    question1_emb = self.encoder.encode(question1)
                    flag_1 = 0
                    if dep != 1:
                        if (1 - cosine(ori_emb, question1_emb)) < 0.5:
                            flag_1 = 1
                            print('\n-------------------------------------')
                            print('the question is far from the original question:{}'.format(question1))
                            print('-------------------------------------\n')
                        else:
                            for emb in existing_emb:
                                cos_sim = 1 - cosine(question1_emb, emb)
                                if cos_sim > 0.9:
                                    flag_1 = 1
                                    print('\n-------------------------------------')
                                    print('there are similar question with:{}'.format(question1))
                                    print('-------------------------------------\n')
                                    break
                    if flag_1 == 0:
                        existing_emb.append(question1_emb)
                        question_all.append(question1)

                        evidence = self.retrieve_once(data, G, question1, self.args.step)

                        start = "Given the following question and contexts, create a final answer to the question."
                        input = start + '\n=========\n' + 'QUESTION: ' + question1 + '\n=========\n' \
                                  + 'CONTEXT:\n' + unique(evidence) + '\n=========\n' + 'QUESTION: ' + question + '\n=========\n'


                        if self.args.test_dataset.lower() != 'fanoutqa':
                            input = input + 'ANSWER: please answer less than 6 words.'
                        else:
                            input = input + 'ANSWER:'

                        print('\n------------sequential question1 answer input---------------')
                        print(input)
                        print('----------------------------\n')

                        pred = llm_runner_chat2(self.llm, [input], 0, 1024, n=1)[0]
                        answer = pred

                        print('\n------------sequential question1 answer---------------')
                        print(answer)
                        print('----------------------------\n')

                        input = self.sequential_prompt_without_evidence + ' ' + question + '\n'\
                                + 'Let\'s generate sub-question 1 first.\n'\
                                + 'Sub-question 1: ' + question1 + '\n'\
                                + 'Answer of Sub-question 1: ' + answer + '\n'\
                                + 'Based on the Answer to Sub-question 1, we can further generate the rest Sub-questions as follows:\n'


                        print('\n------------sequential without evidence decompose---------------')
                        print(input)
                        print('----------------------------\n')

                        pred = llm_runner_chat2(self.llm, [input], 0, 1024, n=1, stop='Question:')[0]
                        print('\n------------sequential pred---------------')
                        print(pred)
                        print('----------------------------\n')

                        input = self.extract_questions_prompt + ' ' + pred.replace('\n\n', '\n') + '\nOutput:'
                        flag_extract = 0
                        while True:
                            pred_questions = llm_runner_chat2(self.llm, [input], flag_extract, 1024)[0]
                            flag_extract = 1
                            if '[\'' in pred_questions or '[\"' in pred_questions:
                                pred_questions = extract_after_last_marker(pred_questions)
                                print('\n------------------sequential pred_questions-------------------')
                                print(pred_questions)
                                print('------------------sequential pred_questions-------------------\n')
                                try:
                                    questions = re.search(r"\[.*\]", pred_questions).group()
                                    questions = questions.replace('\"', "\'").replace('[\'', '[\"').replace('\', \'','\", \"').replace('\']', '\"]').replace('\',\n', '\',')
                                    questions = ast.literal_eval(questions)
                                    for i, item in enumerate(questions):
                                        if ":" in item:
                                            questions[i] = item.split(":")[1].strip()
                                    break
                                except:
                                    print('\n-------------------------------------')
                                    print('extract error!')
                                    print('-------------------------------------\n')
                                    with open('result_ours_{}/{}/{}/error_extract_{}_{}_{}.txt'.format(self.args.version,
                                                                                                       self.args.llm,
                                                                                                       self.args.test_dataset,
                                                                                                       self.args.train_dataset,
                                                                                                       self.args.step,
                                                                                                       self.args.sample_num), 'a', encoding='=utf-8') as f:
                                        f.write(pred_questions + '\n\n\n' + "="*50)
                                    if dep == 1:
                                        print('\n----------------------------')
                                        print('continue when dep is 1!')
                                        print('----------------------------\n')
                                        continue

                        print('\n------------------sequential questions-------------------')
                        print(questions)
                        print('------------------sequential questions-------------------\n')

                        if questions == []:
                            return [question]
                        if len(questions) > 10:
                            questions = questions[:10]

                        embeddings = self.encoder.encode(questions)
                        for i, q_emb in enumerate(embeddings):
                            flag = 0
                            temp_emb = copy.deepcopy(existing_emb)
                            if dep != 1:
                                if (1 - cosine(ori_emb, q_emb)) < 0.5:
                                    flag = 1
                                    print('\n-------------------------------------')
                                    print('the question is far from the original question:{}'.format(questions[i]))
                                    print('-------------------------------------\n')
                                elif "question" in questions[i].lower():
                                    flag = 1
                                    print('\n-------------------------------------')
                                    print('the question is not context clear:{}'.format(questions[i]))
                                    print('-------------------------------------\n')
                                else:
                                    for emb in existing_emb:
                                        cos_sim = 1 - cosine(q_emb, emb)
                                        if cos_sim > 0.9:
                                            flag = 1
                                            print('\n-------------------------------------')
                                            print('there are similar question with:{}'.format(questions[i]))
                                            print('-------------------------------------\n')
                                            break
                            if flag == 0:
                                temp_emb.append(q_emb)
                                sub_question_all = self.decompose(data, questions[i], G, dep + 1, temp_emb, ori_emb)
                                question_all.extend(sub_question_all)
                                print('question:{}'.format(questions[i]))
                                print('subquestion of question:{}'.format(sub_question_all))

                    return question_all
            elif dep != 1:
                print('\n----------')
                print('exist')
                print('----------\n')
                break
        return [question]

    def retrieve(self, data, G):
        print('\n-------------------------------------')
        print('strat retrieve')
        print('-------------------------------------\n')
        existing_emb = [self.encoder.encode(data['question'])]
        questions = self.decompose(data, data['question'], G, 1, existing_emb, existing_emb[0])
        questions.insert(0, data['question'])
        unique_questions = []
        for item in questions:
            if item not in unique_questions and item != "":
                unique_questions.append(item)

        if not os.path.exists('result_ours_{}/{}/{}'.format(self.args.version, self.args.llm, self.args.test_dataset)):
            os.makedirs('result_ours_{}/{}/{}'.format(self.args.version, self.args.llm, self.args.test_dataset))
        if not os.path.exists('result_ours_{}/{}/{}/{}_question_{}.json'.format(self.args.version, self.args.llm, self.args.test_dataset, self.args.train_dataset, self.args.step)):
            with open('result_ours_{}/{}/{}/{}_question_{}.json'.format(self.args.version, self.args.llm, self.args.test_dataset, self.args.train_dataset, self.args.step), 'w') as f:
                json.dump({data['question']: unique_questions}, f,indent=4)
        else:
            with open('result_ours_{}/{}/{}/{}_question_{}.json'.format(self.args.version, self.args.llm, self.args.test_dataset, self.args.train_dataset,self.args.step), 'r') as f:
                data_dict = json.load(f)
            if data['question'] not in data_dict.keys():
                data_dict[data['question']] = unique_questions
            with open('result_ours_{}/{}/{}/{}_question_{}.json'.format(self.args.version, self.args.llm, self.args.test_dataset, self.args.train_dataset,self.args.step), 'w') as f:
                json.dump(data_dict, f, indent=4)


        print("="*50)
        print(len(unique_questions))
        print(data['question'])
        print(unique_questions)
        print("=" * 50)

        question_embs = self.encoder.encode(unique_questions)
        seed_question_emb = self.encoder.encode(data['question'])

        evidence_all = []
        for i, question in enumerate(unique_questions):
            cos_sim = 1 - cosine(question_embs[i], seed_question_emb)
            if cos_sim >= self.args.alpha:
                evidence = self.retrieve_once(data, G, question, self.args.step)
                evidence_all.extend(evidence)
        return evidence_all

    def retrieve_once(self, data, G, question, step=2):
        corpus = [c for _, c in data['title_chunks']]
        candidates_idx = list(range(len(corpus)))

        seed = question
        context_final = []

        strat_sample_num = max(self.k // (self.k_nei ** (step - 1)), self.k_nei)

        sample_num = self.k_nei

        seed_idxs = tf_idf(seed, candidates_idx, corpus, k=strat_sample_num, visited=[])
        idxs = [[_] for _ in seed_idxs]
        context_now = [seed + '\n\n' + corpus[idx[0]] for idx in idxs]
        context_next = []
        idxs_all = []
        for step_i in range(step - 1):
            idxs_temp = []
            sum = 0

            next_reason_texts = llm_runner_chat(self.llm, self.lora, context_now, 0, 1024)

            for i, idx_cur in enumerate(idxs):

                next_reason_text = next_reason_texts[i]

                if 'no more evidence is needed' in next_reason_text.lower():
                    context_final.append(context_now[i].strip(seed).strip())
                    idxs_all.append(idx_cur)
                    continue

                next_candidates_idx = list(G.neighbors(idx_cur[-1]))

                idx = idx_cur[-1]

                if strat_sample_num * (self.k_nei ** (step_i + 1)) > self.k:
                    sample_num = 1

                next_idxs = tf_idf(next_reason_text, next_candidates_idx, corpus, k=sample_num, visited=idx_cur)

                if len(next_idxs) < sample_num:
                    sum += 1

                if next_idxs != []:
                    for next_idx in next_idxs:
                        if corpus[next_idx] != corpus[idx]:
                            context_next.append(context_now[i] + '\n\n' + corpus[next_idx])
                            new_list = copy.deepcopy(idx_cur)
                            idxs_temp.append(new_list)
                            idxs_temp[-1].append(next_idx)
                else:
                    context_final.append(context_now[i].strip(seed).strip())
                    idxs_all.append(idx_cur)

            idxs = idxs_temp

            context_now = context_next
            context_next = []

        idxs_all.extend(idxs)
        contexts = [item.strip(seed).strip() for item in context_now]
        context_final.extend(contexts)

        return context_final

