import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default='hotpotqa', choices=['no','hotpotqa', '2wikimqa', 'musique', '2wikimqa_stop', 'musique_stop', 'hotpotqa_stop'])

    parser.add_argument("--test_dataset", nargs="?", default="fanoutqa",choices=['hotpotqa', '2wikimqa', 'musique', 'fanoutqa', 'longhopqa', 'hotpotqa_val', '2wikimqa_val', 'musique_val'])

    parser.add_argument("--step", type=int, default=20)

    parser.add_argument("--sample_num", type=int, default=4)

    parser.add_argument("--gpu", type=str, default='0,1,2,3,4,5,6,7')

    parser.add_argument("--llm", type=str, default='llama',choices=["llama","qwen","mistral"])

    parser.add_argument("--retriever", type=str, default='kgp', choices=['no', 'kgp', 'ircot'])

    parser.add_argument("--seed", type=int, default=1028)
    parser.add_argument("--mode", type=str, default='str', choices=['str','unique'])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--n_processes", type=int, default = 8)
    parser.add_argument("--k", type = int, default = 30)
    parser.add_argument("--k_nei", type = int, default = 3)
    parser.add_argument("--k_emb", type = int, default = 15)
    parser.add_argument("--kg", type = str, default = "KG_TAGME_0.8")
    return parser.parse_args()