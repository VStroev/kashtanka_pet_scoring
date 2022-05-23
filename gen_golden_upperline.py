import random
from pathlib import Path
import pandas as pd
from fire import Fire

def golden_scores(data_path):
    data_path = Path(data_path)
    tdf = pd.read_csv(data_path/'registry.csv')

    allanswers = tdf.dropna(subset=['answer_name']).answer_name
    gen = ( (r.query_name, answer, 1./(rank+1)) for _,r in tdf[['query_name','answer_name']].iterrows() 
           for rank, answer in enumerate( ['abc' if pd.isna(r.answer_name) else r.answer_name] + allanswers.sample(100).to_list()) if rank<100 )
    best_preds = pd.DataFrame.from_records(gen, columns=['query','answer','score'])


    best_scores = pd.DataFrame.from_records( ((r.query_name, k, 0.0 if pd.isna(r.answer_name) else 1.0) for _,r in tdf[['query_name','answer_name']].iterrows()
                                for k in [1,3,10]), columns=['query','k','hit_prob'] )

    best_df = pd.concat([best_preds,best_scores],ignore_index=True)
    best_df.to_csv('pred.csv',index=False)

if __name__ == '__main__':
    Fire(golden_scores)
