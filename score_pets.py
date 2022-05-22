from pathlib import Path

import pandas as pd
from fire import Fire
K_VALS = [5, 10, 30, 100]
SCORED_PARTS = ('dev', 'dev_small', 'test')


def hit_k(scores, answer, k):
    scores = list(sorted(scores, key= lambda x: x[1]))
    preds = {x[0] for x in scores[:k]}
    return int(answer in preds)


def get_true_answer(true_df, query):
    df = true_df.loc[query]
    return df.answer_name


def get_preds(df, query):
    df = df.loc[query]
    ret = list(zip(df.answer.values, df.score.values))
    return ret


def get_pred_existing(pred_df_k, query, k):
    df = pred_df_k[k]
    df = df.loc[query]
    return df.exists_prob.item()
def score_part(true_df, pred_df):
    df = pred_df.dropna(subset=['answer']).copy()
    df = df.sort_values('score', ascending=False).groupby('query').agg({'answer': list})
    df = true_df.merge(df, left_on='query_name', right_on='query', how='left', validate='one_to_one')
    df_matchable = df.dropna(subset=['answer_name'])  # search_recall@k is calculated on matchable examples only

    metrics = {}
    for topk in K_VALS:
        metrics[f'search_recall@{topk}'] = df_matchable.apply(lambda r: r.answer_name in r.answer[:topk], axis=1).mean()
    return metrics

def score_preds(preds_path, data_dir, case_type=None, ad_type=None, compressed=False, parts=SCORED_PARTS):
    data_dir = Path(data_dir)
    compression = 'gzip' if compressed else 'infer'
    scores = {}
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