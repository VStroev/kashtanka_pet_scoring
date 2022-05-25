import random
from pathlib import Path
import pandas as pd
from fire import Fire

def golden_scores(data_path):
    data_path = Path(data_path)
    tdf = pd.read_csv(data_path/'registry.csv')

    df = pd.DataFrame({'query':tdf.query_name})
    for k in [1,3,10]:
        df[f'matched_{k}'] = (~tdf.answer_name.isnull()).astype(float)

    allanswers = tdf.dropna(subset=['answer_name']).answer_name
    df['answer'] = tdf.answer_name.fillna('asdf').apply(lambda a: ','.join([a] + allanswers.sample(99).to_list()))
    df.to_csv('preds.tsv',sep='\t', index=False)


if __name__ == '__main__':
    Fire(golden_scores)
