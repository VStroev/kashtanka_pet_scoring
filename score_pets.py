from pathlib import Path

import numpy as np
import pandas as pd
from fire import Fire
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
K_VALS = [5, 10, 30, 100]
SCORED_PARTS = ('dev', 'dev_small', 'test')

def get_cands_df(pdf, tdf):
    """
    Combines prediction and true answer dataframes for candidate recall calculation.
    """
    df = pdf.dropna(subset=['answer']).copy()  # leave only candidate rows
    df.score = df.score.astype(float)
    assert df.score.isnull().sum()==0  # TODO: replace with an exception! test: submit incorrect preds to the server!

    # for each query get a list of answers in appropriate order; droupby preserves the original order for each group!
    df = df.sort_values('score', ascending=False).groupby('query').agg({'answer':list})

    assert (df.answer.apply(len)==100).all()   # TODO: replace with an exception! test: submit <100 preds for a few queries to the server!

    # expand df with true answers with correponding preds
    df = tdf.merge(df, left_on='query_name', right_on='query', how='left', validate='one_to_one')
    assert df.answer.isnull().sum()==0  # TODO: replace with an exception! test: submit preds with some missing queries to the server!
    return df

def write_true_ranks(df):
    """
    Adds a true_rank column with the ranks of the true answer for matchables and NaNs for non-matchables.
    Ranks are between 0 and 100, 100 meaning the true answer is not among 100 predicted candidates.
    """
    df_matchable = df.dropna(subset=['answer_name'])  # cand_recall@k is calculated on matchable examples only

    # for each query: append the true answer to the end of the candidate list, 
    # find the rank of the true answer; appending the true answer is required to avoid exceptions from list.index()
    rdf = df_matchable.answer.apply(lambda l: l.copy()).to_frame()  # copy lists to append true answers w/o changing df_matchable
    rdf['answer_name'] = df_matchable.answer_name
    rdf.apply(lambda r: r.answer.append(r.answer_name), axis=1) 
    all_ranks = rdf.apply(lambda r: r.answer.index(r.answer_name), axis=1)  # ranks 0..99, 100 for non-found

    # assignment shall respect index; to make sure, validate that non_matchables don't have true_ranks
    df['true_rank'] = all_ranks
    assert (df.answer_name.isnull()==df.true_rank.isnull()).all() 


def add_cand_metrics(ranks, metrics):
    ranks = ranks.dropna()  # drop non-matchables; cand metrics are calculated on matchables only!
    metrics['matchable_num'] = len(ranks)
    for topk in [1,3,10,30,100]:
        metrics[f'candR@{topk}'] = (ranks < topk).mean()

    # reciprocal rank is 0.0 for non-found, otherwise 1/(rank+1) (for 0-based ranks)
    rr = (1./(ranks+1)).where(ranks<100, 0.0)  
    metrics['candMRR'] = rr.mean()


def merge_hit_probs(df, pdf):    
    match_scores = pdf.dropna(subset=['k']).pivot(index='query', columns='k', values='hit_prob')
    match_scores = match_scores.rename(columns={c: f'matched_{int(c)}' for c in match_scores.columns})
    df = df.merge(match_scores, left_on='query_name', right_on='query',  how='left', validate='one_to_one')
    return df

def add_hitpred_metrics(q, k, metrics):
    non_matchable_pct = int(q.true_rank.isnull().mean()*100)
    # for now take 10% of the queries with largest matched scores; later shall draw prec(pct) curve
    topprop = 0.1
    topn = int(len(q)*topprop)  
    metrics[f'nonm{non_matchable_pct}%_top{topprop}'] = topn
    prec = (q.head(topn).true_rank < k).mean()
    # print(k, prec, topn, int(non_matchable_prop*100)/100)
    metrics[f'hit{k}pred_nonm{non_matchable_pct}%_P@top{topprop}'] = prec


def score_part(tdf, pdf):
    from collections import defaultdict
    metrics = defaultdict(dict)
    
    df = get_cands_df(pdf, tdf)
    write_true_ranks(df)

    for case_type in ['hard','simple']:
        for type in ['lost','found']:
            ranks = df[(df.case_type==case_type)&(df.type==type)].true_rank
            add_cand_metrics(ranks, metrics[f'{case_type}_{type}'])


    df = merge_hit_probs(df,pdf)
    for k in [1,3,10]:
        field = f'matched_{k}'
        assert df[field].isnull().sum()==0, f'No matched score for some queries for k={k}' 

        for case_type in ['hard','simple']:
            for type in ['lost','found']:
                mdf = df[(df.case_type==case_type)&(df.type==type)]
                mdf_nonmatchable = mdf[mdf.true_rank.isnull()]
                # generate more non-matchables preserving the hit_prob distribution, the number is such that
                # after concatenating with mdf there will be 10*len(matchable) examples, 
                # i.e. if for an ideal model top 10% will be matchable and hitpred precision is 1.0
                nonmatchable_10x = mdf_nonmatchable.sample(9*len(mdf)-10*len(mdf_nonmatchable),replace=True) 
                for q in [mdf, pd.concat([mdf, nonmatchable_10x], ignore_index=True)]:
                    q = q.sort_values(by=field, ascending=False)
                    add_hitpred_metrics(q, k, metrics[f'{case_type}_{type}'])

    return metrics


def score_preds(preds_path, data_dir, parts='dev', case_type=None, ad_type=None, compressed=False):
    data_dir = Path(data_dir)
    compression = 'gzip' if compressed else 'infer'
    scores = {}
    if type(parts)==str: 
        parts = [parts]
    for part in parts:

        pred_df = pd.read_csv(preds_path, compression = compression, header=0)
        if compressed:
            pred_df = pred_df.rename(columns={pred_df.columns[0]: 'query'}) #  For sime reason query column name is replaced

        true_df = pd.read_csv(str(data_dir / part / 'registry.csv'))

        if ad_type is not None:
            true_df = true_df[true_df.type == ad_type]
        if case_type is not None:
            true_df = true_df[true_df.case_type == case_type]
        queries = true_df['query_name'].unique()
        assert len(queries) == len(true_df)

        metrics = score_part(true_df, pred_df)
        scores[part] = metrics

    return scores


if __name__ == '__main__':
    Fire(score_preds)
