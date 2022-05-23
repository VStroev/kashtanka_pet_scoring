import random
from pathlib import Path
import pandas as pd
from fire import Fire

PROB_THESHOLD = [1,3, 10]


def process_case(query, answers):
    scores = [(x.name, random.random()) for x in answers]
    scores = list(sorted(scores, key= lambda x: -x[1]))
    scores = scores[:100]
    return scores


def process_cases(queries_path, answers_path):
    queries = list(queries_path.glob('*'))
    answers = list(answers_path.glob('*'))
    ret = []
    for q in queries:
        scores = process_case(q, answers)
        for item in scores:
            ret.append([q.name, *item, None, None])
        for p in PROB_THESHOLD:
            ret.append([q.name, None, None, p, random.random()])
    return ret

def random_score(data_path):
    data_path = Path(data_path)

    header = ['query', 'answer', 'score', 'k', 'hit_prob']
    data = []

    data += process_cases(data_path/'lost'/'lost', data_path/'lost'/'synthetic_found')
    data += process_cases(data_path/'found'/'found', data_path/'found'/'synthetic_lost')
    df = pd.DataFrame(data, columns=header)
    df.to_csv('pred.csv', index=False)

if __name__ == '__main__':
    Fire(random_score)
