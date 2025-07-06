import os
from parse import parse_args
import json
from dependency_parser_complexity import dependency_analysis
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import random
def get_sample(questions, K):
    embeddings = encoder.encode(questions)

    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    closest_questions = []
    for cluster_id in range(K):
        cluster_points_indices = np.where(labels == cluster_id)[0]
        cluster_points = embeddings[cluster_points_indices]

        cluster_center = centers[cluster_id]

        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)

        closest_point_index = cluster_points_indices[np.argmin(distances)]

        closest_questions.append(questions[closest_point_index].split('\n')[0])
    return closest_questions

if __name__ == '__main__':
    args = parse_args()
    sample_questions_prompt = []
    sample_parallel_questions_prompt = []
    sample_sequential_questions_prompt = []

    encoder = SentenceTransformer("all-mpnet-base-v2")
    one_hop_questions = []

    parallel_questions = []
    parallel_questions_str = []
    with open('{}/{}/parallel_questions_dict.json'.format(args.llm, args.train_dataset), 'r') as f:
        data = json.load(f)
    print(len(data))
    for question, pred in data.items():
        pred = pred.replace('sub-question', 'Sub-question')
        if 'Sub-question 1:' in pred and 'Sub-question 2:' in pred and 'Sub-question 3:' not in pred:
            sub_question1 = pred.split('Sub-question 1:')[1].split('\n')[0]
            sub_question2 = pred.split('Sub-question 2:')[1].split('\n')[0]
            if 'question' in sub_question1.lower() or 'question' in sub_question2.lower():
                print(sub_question1)
                print(sub_question2)
                continue
            rel_num_one, avg_dis_one, max_dep_one = dependency_analysis(sub_question1)
            rel_num_two, avg_dis_two, max_dep_two = dependency_analysis(sub_question2)
            rel_num_three, avg_dis_three, max_dep_three = dependency_analysis(question)
            if rel_num_one and rel_num_two and rel_num_three:
                if avg_dis_three >= avg_dis_two and avg_dis_three >= avg_dis_one and max_dep_three >= max_dep_two and max_dep_three >= max_dep_one:
                    parallel_questions.append([question, sub_question1, sub_question2])
                    parallel_questions_str.append(question + '\n' +sub_question1 +'\n' +sub_question2)

    if len(parallel_questions) < args.sample_num:
        parallel_questions = []
        parallel_questions_str = []
        for question, pred in data.items():
            pred = pred.replace('sub-question', 'Sub-question')
            if 'Sub-question 1:' in pred and 'Sub-question 2:' in pred and 'Sub-question 3:' not in pred:
                sub_question1 = pred.split('Sub-question 1:')[1].split('\n')[0]
                sub_question2 = pred.split('Sub-question 2:')[1].split('\n')[0]
                if 'question' in sub_question1.lower() or 'question' in sub_question2.lower():
                    print(sub_question1)
                    print(sub_question2)
                    continue
                parallel_questions.append([question, sub_question1, sub_question2])
                parallel_questions_str.append(question + '\n' + sub_question1 + '\n' + sub_question2)

    if len(parallel_questions) < args.sample_num:
        parallel_questions = []
        parallel_questions_str = []
        for question, pred in data.items():
            pred = pred.replace('sub-question', 'Sub-question')
            if 'Sub-question 1:' in pred and 'Sub-question 2:' in pred:
                sub_question1 = pred.split('Sub-question 1:')[1].split('\n')[0]
                sub_question2 = pred.split('Sub-question 2:')[1].split('\n')[0]
                if 'question' in sub_question1.lower() or 'question' in sub_question2.lower():
                    print(sub_question1)
                    print(sub_question2)
                    continue
                parallel_questions.append([question, sub_question1, sub_question2])
                parallel_questions_str.append(question + '\n' + sub_question1 + '\n' + sub_question2)

    print(len(parallel_questions))
    sample_parallel_questions = get_sample(parallel_questions_str, args.sample_num)

    for question, pred in data.items():
        if question not in sample_parallel_questions:
            if 'Sub-question 1:' in pred and 'Sub-question 2:' in pred and 'Sub-question 3:' not in pred:
                sub_question1 = pred.split('Sub-question 1:')[1].split('\n')[0]
                sub_question2 = pred.split('Sub-question 2:')[1].split('\n')[0]
                one_hop_questions.append(sub_question1)
                one_hop_questions.append(sub_question2)


    for question in sample_parallel_questions:
        sub_question1 = data[question].split('Sub-question 1:')[1].split('\n')[0].strip()
        sub_question2 = data[question].split('Sub-question 2:')[1].split('\n')[0].strip()
        parallel_q = "Question: " + question + '\n' + \
            "Sub-question 1: " + sub_question1 + '\n' +\
            "Sub-question 2: " + sub_question2 + '\n' +\
            """These Sub-questions are independent.
Furthermore, answering these Sub-questions is necessary for obtaining the final answer to the parent question.
Hence, this question is a parallel multi-hop question."""
        sample_questions_prompt.append(parallel_q)
        sample_parallel_questions_prompt.append(parallel_q)

    sequential_questions = []
    sequential_questions_str = []
    with open('{}/{}/sequential_questions_dict.json'.format(args.llm, args.train_dataset), 'r') as f:
        data = json.load(f)
    print(len(data))
    for question, pred in data.items():
        pred = pred.replace('sub-question', 'Sub-question')
        if 'Sub-question 1:' in pred and 'Sub-question 2:' in pred and 'Sub-question 3:' not in pred and ('the answer of Sub-question 1:' in pred or 'the answer to Sub-question 1:' in pred):
            sub_question1 = pred.split('Sub-question 1:')[1].split('\n')[0]
            sub_question2 = pred.split('Sub-question 2:')[1].split('\n')[0]
            if 'question' in sub_question1.lower() or 'question' in sub_question2.lower():
                print(sub_question1)
                print(sub_question2)
                continue
            rel_num_one, avg_dis_one, max_dep_one = dependency_analysis(sub_question1)
            rel_num_two, avg_dis_two, max_dep_two = dependency_analysis(sub_question2)
            rel_num_three, avg_dis_three, max_dep_three = dependency_analysis(question)
            if rel_num_one and rel_num_two and rel_num_three:
                if avg_dis_three >= avg_dis_two and avg_dis_three >= avg_dis_one and max_dep_three >= max_dep_two and max_dep_three >= max_dep_one:
                    sequential_questions.append([question, sub_question1, sub_question2])
                    sequential_questions_str.append(question + '\n' + sub_question1 + '\n' + sub_question2)

    if len(sequential_questions) < args.sample_num:
        sequential_questions = []
        sequential_questions_str = []
        with open('{}/{}/sequential_questions_dict.json'.format(args.llm, args.train_dataset), 'r') as f:
            data = json.load(f)
        print(len(data))
        for question, pred in data.items():
            pred = pred.replace('sub-question', 'Sub-question')
            if 'Sub-question 1:' in pred and 'Sub-question 2:' in pred and 'Sub-question 3:' not in pred and (
                    'the answer of Sub-question 1:' in pred or 'the answer to Sub-question 1:' in pred):
                sub_question1 = pred.split('Sub-question 1:')[1].split('\n')[0]
                sub_question2 = pred.split('Sub-question 2:')[1].split('\n')[0]
                if 'question' in sub_question1.lower() or 'question' in sub_question2.lower():
                    print(sub_question1)
                    print(sub_question2)
                    continue
                sequential_questions.append([question, sub_question1, sub_question2])
                sequential_questions_str.append(question + '\n' + sub_question1 + '\n' + sub_question2)

    print(len(sequential_questions))
    sample_sequential_questions = get_sample(sequential_questions_str, args.sample_num)

    for question, pred in data.items():
        if question not in sample_parallel_questions:
            if 'Sub-question 1:' in pred and 'Sub-question 2:' in pred and 'Sub-question 3:' not in pred:
                sub_question1 = pred.split('Sub-question 1:')[1].split('\n')[0]
                sub_question2 = pred.split('Sub-question 2:')[1].split('\n')[0]
                one_hop_questions.append(sub_question1)
                one_hop_questions.append(sub_question2)

    for question in sample_sequential_questions:
        sample_questions_prompt.append("Question: " + question + '\n' + data[question].replace('sub-question', 'Sub-question').replace("To obtain the Sub-question 2", "To obtain the rest Sub-questions").replace("we can further generate the Sub-question 2","we can further generate the rest Sub-questions").replace("Sub-question 2:","The rest Sub-questions:").replace("The Sub-questions 1 and 2 are dependent","These Sub-questions are dependent").replace("before generating the Sub-question 2","before generating the rest Sub-questions").replace("answering Sub-question 1 and 2 is necessary", "answering these Sub-questions is necessary"))
        sample_sequential_questions_prompt.append("Question: " + question + '\n' + data[question].replace('sub-question', 'Sub-question').replace("To obtain the Sub-question 2", "To obtain the rest Sub-questions").replace("we can further generate the Sub-question 2","we can further generate the rest Sub-questions").replace("Sub-question 2:","The rest Sub-questions:").replace("The Sub-questions 1 and 2 are dependent","These Sub-questions are dependent").replace("before generating the Sub-question 2","before generating the rest Sub-questions").replace("answering Sub-question 1 and 2 is necessary", "answering these Sub-questions is necessary"))


    sample_one_hop_questions = get_sample(one_hop_questions, args.sample_num)
    for question in sample_one_hop_questions:
        sample_questions_prompt.append("Question: " + question + '\nThe question is already a minimal question unit and cannot be broken down into multiple smaller Sub-questions.\nHence, this question is a one-hop question.')

    random.shuffle(sample_questions_prompt)
    prompt = """<Instruction>
Definition of Sub-question: A Sub-question is a question that must be answered in the process of finding the answer to its parent question. In other words, obtaining the answer to the Sub-question is a necessary condition for obtaining the answer to the parent question. Note that if question A and question B are highly related, but obtaining the answer to question A is not necessary for obtaining the answer to question B, then question A is not a Sub-question of question B. For example, question A "What is the city of Incheon?" and question B "What is the symbol of Incheon?" are very related, but the answer to question A is not necessary for obtaining the answer to question B, so question A is not a Sub-question of question B.
There are three types of questions, and their definitions are as follows:
One-hop Question: The question is already a minimal question unit and cannot be broken down into multiple smaller Sub-questions.
For instance, "Kahibah FC is based is what country?" is a one-hop question since it cannot be broken down into multiple smaller Sub-questions.
Parallel Multi-hop Question: The question includes multiple independent Sub-questions, so it can be directly broken down into multiple independent Sub-questions.
For instance, "Did LostAlone and Guster have the same number of members?" is a parallel multi-hop question, which can be directly broken down into Sub-question 1 "How many members are in LostAlone?" and Sub-question 2 "How many members are in Guster?".
Sequential Multi-hop Question: The Sub-questions of sequential multi-hop question are dependent. If we want to generate Sub-question 2, we must first obtain the answer to Sub-question 1.
For instance, "Girard city is located in a county with how many inhabitants?" is a sequential multi-hop question, the Sub-question 1 is "Which county is Girard city located in?". Before obtaining the Sub-questions 2 "How many inhabitants are there in Crawford County?", we must obtain the answer "Crawford County" to the Sub-question 1.
Please identify the type for the given question based on the above definitions and the following examples.
</Instruction>"""
    for i, item in enumerate(sample_questions_prompt):
        prompt += '\n<Example {}>'.format(i+1)
        prompt += '\n' + item
        prompt += '\n</Example {}>'.format(i+1)

    prompt = prompt + """\n<Instruction>
Definition of Sub-question: A Sub-question is a question that must be answered in the process of finding the answer to its parent question. In other words, obtaining the answer to the Sub-question is a necessary condition for obtaining the answer to the parent question. Note that if question A and question B are highly related, but obtaining the answer to question A is not necessary for obtaining the answer to question B, then question A is not a Sub-question of question B. For example, question A "What is the city of Incheon?" and question B "What is the symbol of Incheon?" are very related, but the answer to question A is not necessary for obtaining the answer to question B, so question A is not a Sub-question of question B.
There are three types of questions, and their definitions are as follows:
One-hop Question: The question is already a minimal question unit and cannot be broken down into multiple smaller Sub-questions.
For instance, "Kahibah FC is based is what country?" is a one-hop question since it cannot be broken down into multiple smaller Sub-questions.
Parallel Multi-hop Question: The question includes multiple independent Sub-questions, so it can be directly broken down into multiple independent Sub-questions.
For instance, "Did LostAlone and Guster have the same number of members?" is a parallel multi-hop question, which can be directly broken down into Sub-question 1 "How many members are in LostAlone?" and Sub-question 2 "How many members are in Guster?".
Sequential Multi-hop Question: The Sub-questions of sequential multi-hop question are dependent. If we want to generate Sub-question 2, we must first obtain the answer to Sub-question 1.
For instance, "Girard city is located in a county with how many inhabitants?" is a sequential multi-hop question, the Sub-question 1 is "Which county is Girard city located in?". Before obtaining the Sub-questions 2 "How many inhabitants are there in Crawford County?", we must obtain the answer "Crawford County" to the Sub-question 1.
Please identify the type for the given question based on the above definitions and examples.
</Instruction>
Given question:"""

    with open('{}/{}/sequential_parallel_{}.txt'.format(args.llm, args.train_dataset,args.sample_num), 'w') as f:
        f.write(prompt)

    random.shuffle(sample_sequential_questions_prompt)
    prompt = """<Instruction>
Please imitate the following examples to decompose the given question.
Please note that during the decomposition process, you must strictly adhere to the following three guidelines:
Guideline 1: Do not decompose into Sub-questions unrelated to the given question.
Guideline 2: Do not expand the scope of the question.
Guideline 3: Please ensure each Sub-question is self-contained and contextually clear. Avoid using ambiguous references or placeholders that rely on information outside the specific Sub-question. For example, in Sub-question \"What is the composer of the aforementioned movie?\", you should replace \"the aforementioned movie\" with the actual name of the movie it represents.
</Instruction>"""
    for i, item in enumerate(sample_sequential_questions_prompt):
        prompt += '\n<Example {}>'.format(i+1)
        prompt += '\n' + item
        prompt += '\n</Example {}>'.format(i+1)

    prompt = prompt + \
    """\n<Instruction>
Please imitate the following examples to decompose the given question.
Please note that during the decomposition process, you must strictly adhere to the following three guidelines:
Guideline 1: Do not decompose into Sub-questions unrelated to the given question.
Guideline 2: Do not expand the scope of the question.
Guideline 3: Please ensure each Sub-question is self-contained and contextually clear. Avoid using ambiguous references or placeholders that rely on information outside the specific Sub-question. For example, in Sub-question \"What is the composer of the aforementioned movie?\", you should replace \"the aforementioned movie\" with the actual name of the movie it represents.
</Instruction>
Given question:"""
    with open('{}/{}/sequential_with_evidence_{}.txt'.format(args.llm, args.train_dataset,args.sample_num), 'w') as f:
        f.write(prompt)


    prompt = """<Instruction>
Please imitate the following examples to decompose the given question.
Please note that during the decomposition process, you must strictly adhere to the following three guidelines:
Guideline 1: Do not decompose into Sub-questions unrelated to the given question.
Guideline 2: Do not expand the scope of the question.
Guideline 3: Please ensure each Sub-question is self-contained and contextually clear. Avoid using ambiguous references or placeholders that rely on information outside the specific Sub-question. For example, in Sub-question \"What is the composer of the aforementioned movie?\", you should replace \"the aforementioned movie\" with the actual name of the movie it represents.
</Instruction>"""
    for i, item in enumerate(sample_sequential_questions_prompt):
        prompt += '\n<Example {}>'.format(i+1)

        if 'the answer of Sub-question 1:' in item:
            answer = item.split('the answer of Sub-question 1:')[1].split('\n')[0].strip()
        elif 'the answer to Sub-question 1:' in item:
            answer = item.split('the answer to Sub-question 1:')[1].split('\n')[0].strip()


        sub_question1 = item.split('Sub-question 1:')[1].split('\n')[0].strip()
        sub_question2 = item.split('The rest Sub-questions:')[1].split('\n')[0].strip()
        question = item.split('\n')[0].strip()

        prompt += '\n' + question \
        + '\nSub-question 1: ' + sub_question1 + '\n'\
        + 'Answer of Sub-question 1: ' + answer + '\n'\
        + 'Based on the Answer to Sub-question 1, we can further generate the rest Sub-questions as follows:\n'\
        + 'The rest Sub-questions: ' + sub_question2
        prompt += '\n</Example {}>'.format(i+1)

    prompt = prompt + \
    """\n<Instruction>
Please imitate the following examples to decompose the given question.
Please note that during the decomposition process, you must strictly adhere to the following three guidelines:
Guideline 1: Do not decompose into Sub-questions unrelated to the given question.
Guideline 2: Do not expand the scope of the question.
Guideline 3: Please ensure each Sub-question is self-contained and contextually clear. Avoid using ambiguous references or placeholders that rely on information outside the specific Sub-question. For example, in Sub-question \"What is the composer of the aforementioned movie?\", you should replace \"the aforementioned movie\" with the actual name of the movie it represents.
</Instruction>
Given question:"""

    with open('{}/{}/sequential_without_evidence_{}.txt'.format(args.llm, args.train_dataset,args.sample_num), 'w') as f:
        f.write(prompt)

    random.shuffle(sample_parallel_questions_prompt)
    prompt = "<Instruction>\n" + \
             "Please imitate the following examples to generate the Sub-questions.\n" + \
             "Please note that during the decomposition process, you must strictly adhere to the following three guidelines:\n" + \
             "Guideline 1: Do not decompose into Sub-questions unrelated to the original question.\n" +\
             "Guideline 2: Do not expand the scope of the question.\n" +\
             "Guideline 3: Please ensure each Sub-question is self-contained and contextually clear. Avoid using ambiguous references or placeholders that rely on information outside the specific Sub-question. For example, in Sub-question \"What is the composer of the aforementioned movie?\", you should replace \"the aforementioned movie\" with the actual name of the movie it represents.\n" + \
             "</Instruction>"

    for i, item in enumerate(sample_parallel_questions_prompt):
        prompt += '\n<Example {}>'.format(i+1)
        prompt += '\n' + item
        prompt += '\n</Example {}>'.format(i+1)

    prompt = prompt + "\n<Instruction>\n" + \
             "Please imitate the above examples to generate the Sub-questions.\n" + \
             "Please note that during the decomposition process, you must strictly adhere to the following three guidelines:\n" + \
             "Guideline 1: Do not decompose into Sub-questions unrelated to the original question.\n" +\
             "Guideline 2: Do not expand the scope of the question.\n" +\
             "Guideline 3: Please ensure each Sub-question is self-contained and contextually clear. Avoid using ambiguous references or placeholders that rely on information outside the specific Sub-question. For example, in Sub-question \"What is the composer of the aforementioned movie?\", you should replace \"the aforementioned movie\" with the actual name of the movie it represents.\n" + \
             "</Instruction>\nGiven question:"


    with open('{}/{}/parallel_{}.txt'.format(args.llm, args.train_dataset, args.sample_num), 'w') as f:
        f.write(prompt)
