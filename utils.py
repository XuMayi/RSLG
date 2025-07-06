from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import requests
import json
import datetime
from evaluation.evaluation import f1_score, exact_match_score, f1, ems
import itertools
from itertools import combinations
import numpy as np
from typing import List
nlp = spacy.load('en_core_web_lg')

def evaluate_dpr_mdr(dataset, retriever, File, step=99, k=None, topks=[1, 5, 10, 20, 30]):

    res = json.load(open(File, 'rb'))

    filter_res = [r for r in res if r['prediction'] != 'System mistake']

    f1s, accs = [], []

    if retriever not in ['golden', 'no'] and dataset.lower() != 'fanoutqa':
        recall, precision, sp_em = [], [], []

    for r in filter_res:
        accs.append(('1' in r['grade']) * 1.0)
        f1s.append(f1_score(r['prediction'], r['answer']))


        if retriever not in ['golden', 'no'] and dataset.lower() != 'fanoutqa':
            r['corpus'] = list(itertools.chain(*[r['corpus'][:step*27]]))
            evi = set([_[1] for _ in r['supports']])

            tmp_recall = []
            tmp_precision = []
            tmp_sp_em = []
            for kk in topks:
                if kk <= k:
                    tmp = set(r['corpus'][:kk*step])

                    tmp_recall.append(len(evi.intersection(tmp)) / len(evi))
                    tmp_precision.append(len(evi.intersection(tmp)) / (kk*step))

                    if evi.issubset(tmp):
                        tmp_sp_em.append(1)
                    else:
                        tmp_sp_em.append(0)

            recall.append(tmp_recall)
            precision.append(tmp_precision)
            sp_em.append(tmp_sp_em)
    print('Acc:', round(np.mean(accs)*100,2))
    print('F1:', round(np.mean(f1s)*100,2))

    if retriever not in ['golden', 'no'] and dataset.lower() != 'fanoutqa':
        print('Recall:', np.mean(np.array(recall), axis=0))
        print('Precision:', np.mean(np.array(precision), axis=0))
        print('SP_EM:', np.mean(np.array(sp_em), axis=0))
        return (round(np.mean(accs)*100,2), round(np.mean(f1s)*100,2),
                [round(item*100,2) for item in np.mean(np.array(recall), axis=0)],
                [round(item*100,2) for item in np.mean(np.array(precision), axis=0)],
                [round(item*100,2) for item in np.mean(np.array(sp_em), axis=0)])
    else:
        return round(np.mean(accs)*100,2), round(np.mean(f1s)*100,2)

def evaluate(dataset, retriever, File, step=99, k=None, topks=[1, 5, 10, 20, 30]):

    res = json.load(open(File, 'rb'))

    filter_res = [r for r in res if r['prediction'] != 'System mistake']

    f1s, accs = [], []

    if retriever not in ['golden', 'no'] and dataset.lower() != 'fanoutqa':
        recall, precision, sp_em = [], [], []

    for r in filter_res:
        accs.append(('1' in r['grade']) * 1.0)
        f1s.append(f1_score(r['prediction'], r['answer']))


        if retriever not in ['golden', 'no'] and dataset.lower() != 'fanoutqa':
            r['corpus'] = list(itertools.chain(*[_.split('\n\n')[:step] for _ in r['corpus']]))
            evi = set([_[1] for _ in r['supports']])

            tmp_recall = []
            tmp_precision = []
            tmp_sp_em = []
            for kk in topks:
                if kk <= k:
                    tmp = set(r['corpus'][:kk*step])

                    tmp_recall.append(len(evi.intersection(tmp)) / len(evi))
                    tmp_precision.append(len(evi.intersection(tmp)) / (kk*step))

                    if evi.issubset(tmp):
                        tmp_sp_em.append(1)
                    else:
                        tmp_sp_em.append(0)

            recall.append(tmp_recall)
            precision.append(tmp_precision)
            sp_em.append(tmp_sp_em)
    print('Acc:', round(np.mean(accs)*100,2))
    print('F1:', round(np.mean(f1s)*100,2))

    if retriever not in ['golden', 'no'] and dataset.lower() != 'fanoutqa':
        print('Recall:', np.mean(np.array(recall), axis=0))
        print('Precision:', np.mean(np.array(precision), axis=0))
        print('SP_EM:', np.mean(np.array(sp_em), axis=0))
        return (round(np.mean(accs)*100,2), round(np.mean(f1s)*100,2),
                [round(item*100,2) for item in np.mean(np.array(recall), axis=0)],
                [round(item*100,2) for item in np.mean(np.array(precision), axis=0)],
                [round(item*100,2) for item in np.mean(np.array(sp_em), axis=0)])
    else:
        return round(np.mean(accs)*100,2), round(np.mean(f1s)*100,2)


def evaluate_ours(dataset, File, k=None, topks=[1, 5, 10, 20, 30]):

    res = json.load(open(File, 'rb'))

    filter_res = [r for r in res if r['prediction'] != 'System mistake']

    f1s, accs = [], []

    if dataset.lower() != 'fanoutqa':
        recall, precision, sp_em = [], [], []

    for r in filter_res:
        accs.append(('1' in r['grade']) * 1.0)
        f1s.append(f1_score(r['prediction'], r['answer']))


        if dataset.lower() != 'fanoutqa':

            kkk = len(r['corpus'])//27
            r['corpus'] = list(itertools.chain(*[_.split('\n\n') for _ in r['corpus']]))
            mmm = len(r['corpus'])//len(set(r['corpus']))
            ddd = mmm//kkk
            if ddd <= 2:
                ddd = 2

            evi = set([_[1] for _ in r['supports']])

            tmp_recall = []
            tmp_precision = []
            tmp_sp_em = []
            for kk in topks:
                if kk <= k:
                    tmp = set(r['corpus'][:kk*kkk])

                    tmp_recall.append(len(evi.intersection(tmp)) / len(evi))
                    tmp_precision.append(len(evi.intersection(tmp)) / (kk*ddd))

                    if evi.issubset(tmp):
                        tmp_sp_em.append(1)
                    else:
                        tmp_sp_em.append(0)

            recall.append(tmp_recall)
            precision.append(tmp_precision)
            sp_em.append(tmp_sp_em)
    print('Acc:', round(np.mean(accs)*100,2))
    print('F1:', round(np.mean(f1s)*100,2))

    if dataset.lower() != 'fanoutqa':
        print('Recall:', np.mean(np.array(recall), axis=0))
        print('Precision:', np.mean(np.array(precision), axis=0))
        print('SP_EM:', np.mean(np.array(sp_em), axis=0))
        return (round(np.mean(accs)*100,2), round(np.mean(f1s)*100,2),
                [round(item*100,2) for item in np.mean(np.array(recall), axis=0)],
                [round(item*100,2) for item in np.mean(np.array(precision), axis=0)],
                [round(item*100,2) for item in np.mean(np.array(sp_em), axis=0)])
    else:
        return round(np.mean(accs)*100,2), round(np.mean(f1s)*100,2)

def context_joint(arr, llm):
    joint_sentences = []

    for element in arr:
        sentences = element.strip().split("\n\n")
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence not in joint_sentences:
                joint_sentences.append(sentence)

    result_text = ''
    for i,s in enumerate(joint_sentences):
        if len(result_text.split()) >= 3072 and llm == 'mistral':
            break
        result_text = result_text + "{}: ".format(i+1) + s + '\n\n'
    result_text = result_text.strip('\n\n')
    return result_text

def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=8)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


def tf_idf(seed, candidates_idx, corpus, k, visited):
    vectorizer = TfidfVectorizer()
    
    try:
        tfidf_matrix = vectorizer.fit_transform([corpus[_] for _ in candidates_idx])

        query_emb = vectorizer.transform([seed])
        cosine_sim = cosine_similarity(query_emb, tfidf_matrix).flatten()
        idxs = cosine_sim.argsort()[::-1]

        tmp_idxs = []
        for idx in idxs:
            if candidates_idx[idx] not in visited:
                tmp_idxs.append(candidates_idx[idx])
            
                k -= 1

            if k == 0:
                break

        return tmp_idxs

    except Exception as e:
        return []



def get_encoder(encoder_type):
    return SentenceTransformer(encoder_type)


def strip_string(string, only_stopwords = False):
    if only_stopwords:
        return ' '.join([str(t) for t in nlp(string) if not t.is_stop])
    else:
        return ' '.join([str(t) for t in nlp(string) if t.pos_ in ['NOUN', 'PROPN']])
    

def window_encodings(sentence, window_size, overlap):
    """Compute encodings for a string by splitting it into windows of size window_size with overlap"""
    tokens = sentence.split()

    if len(tokens) <= window_size:
        return [sentence]
    
    return [' '.join(tokens[i:i + window_size]) for i in range(0, len(tokens) - window_size, overlap)]


def cal_local_llm_llama(input, port):
    # Define the url of the API
    url = "http://localhost:{}/predict".format(port)

    # Replace with actual instruction and input data
    data = {
        'prompt' :
            {
                'prompt': 'What evidence do we need to answer the question given the current evidence?',
                'input': input
            }
        }

    # print(data)
    # Convert the data to JSON format
    data_json = json.dumps(data)

    # print(data_json)

    # Make the POST request
    response = requests.post(url, data=data_json)

    # Get the json response
    # print(response.text)
    response_json = response.json()
    # print("-"*50)
    # print(response_json['output'])
    # print("-" * 50)
    return response_json['output']

# def cal_local_llm_llama(input, port):
#     # Define the url of the API
#     url = "http://localhost:{}/api/ask".format(port)
#
#     # Define the headers for the request
#     headers = {
#         'Content-Type': 'application/json',
#     }
#
#     # Define the data to be sent in the POST request
#     # Replace with actual instruction and input data
#     data = {
#         'instruction': 'What evidence do we need to answer the question given the current evidence?',
#         'input': input
#         }
#
#     # print(data)
#     # Convert the data to JSON format
#     data_json = json.dumps(data)
#
#     # Make the POST request
#     response = requests.post(url, headers=headers, data=data_json)
#
#     # Get the json response
#     response_json = response.json()
#
#     #return response_json['answer']
#     return response_json['output']


def cal_local_llm_t5(input, port):
    # Define the url of the API
    url = "http://localhost:{}/api/ask".format(port)

    # Define the headers for the request
    headers = {
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the POST request
    # Replace with actual instruction and input data
    data = {
        'source_text': input
        }

    # Convert the data to JSON format
    data_json = json.dumps(data)

    # Make the POST request
    response = requests.post(url, headers=headers, data=data_json)

    # Get the json response
    response_json = response.json()

    return response_json['answer']