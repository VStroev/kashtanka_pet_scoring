import numpy as np
import random
from pathlib import Path
import pandas as pd
from fire import Fire


def process_cases(queries_path, answers_path):
    queries = list(queries_path.glob('*'))
    answers = np.array([a.name for a in answers_path.glob('*')])
    
    df = pd.DataFrame({'query': [q.name for q in queries]})

    for k in [1,3,10]:
        df[f'matched_{k}'] = np.random.rand(len(df))

    inds = np.random.randint(low=0, high=len(answers), size=(len(df),100))
    df['answer'] = [','.join(row) for row in answers[inds]]
    return df


def random_score(data_path):
    data_path = Path(data_path)
    df1 = process_cases(data_path/'lost'/'lost', data_path/'lost'/'synthetic_found')
    df2 = process_cases(data_path/'found'/'found', data_path/'found'/'synthetic_lost')
    df = pd.concat([df1,df2], ignore_index=False).to_csv('preds.tsv', index=False, sep='\t')


if __name__ == '__main__':
    Fire(random_score)
