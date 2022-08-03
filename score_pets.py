from pathlib import Path

import numpy as np
import pandas as pd
from fire import Fire
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import datetime
import cv2 

SCORED_PARTS = ('dev', 'dev_small', 'test')


def load_preds_compact(pred_path):
    pdf = pd.read_csv(pred_path, sep='\t')
    return pdf

def load_preds(pred_path):
    return load_preds_compact(pred_path)


def merge_true_answers(df, answers_path):
    tdf = pd.read_csv(answers_path)
    # expand df with true answers with correponding preds
    df = tdf.merge(df, left_on='query_name', right_on='query', how='left', validate='one_to_one')
    assert df.answer.isnull().sum()==0  # TODO: replace with an exception! test: submit preds with some missing queries to the server!
    return df
    

def write_true_ranks(df, opt=1):
    """
    Adds a true_rank column with the ranks of the true answer for matchables and NaNs for non-matchables.
    Ranks are between 0 and 100, 100 meaning the true answer is not among 100 predicted candidates.
    opt=0 is faster but requires more memory, opt=1 is slower (not significantly compared to all other code),
    but requires less memory (100MB vs. 130MB at peak while executing this script). 
    """
    assert (df.answer.str.count(',')==99).all(), f'For some queries not exactly 100 candidates are predicted!'
    df_matchable = df.dropna(subset=['answer_name'])  # cand_recall@k is calculated on matchable examples only

    # for each query: append the true answer to the end of the candidate list, 
    # find the rank of the true answer; appending the true answer is required to avoid exceptions from list.index()
    rdf = (df_matchable.answer + ',' + df_matchable.answer_name).to_frame('answer')
    rdf['answer_name'] = df_matchable.answer_name
    if opt==0:
        all_ranks = rdf.answer.str.split(',').apply(lambda l: l.index(l[-1]))   # ranks 0..99, 100 for non-found
    else:
        all_ranks = rdf.apply(lambda r: r.answer.split(',').index(r.answer_name), axis=1)  # ranks 0..99, 100 for non-found

    # assignment shall respect index; to make sure, validate that non_matchables don't have true_ranks
    df['true_rank'] = all_ranks
    assert (df.answer_name.isnull()==df.true_rank.isnull()).all() 

def add_cand_metrics(ranks, metrics):
    """
    Populate metrics with candidate ranking metrics: 1) MRR: the mean reciprocal rank;
    2) R@k: the recall@k (0 or 1, because there is only 1 true answer for each query) averaged over 
    queries, i.e. the proportion of queries for which the true answer is among k top candidates returned).
    These metrics are calculated on matchable queries only.
    """
    ranks = ranks.dropna()  # drop non-matchables; cand metrics are calculated on matchables only!
    metrics['nmatchable'] = len(ranks)
    for topk in [1,3,10,30,100]:
        metrics[f'candR@{topk}'] = (ranks < topk).mean()

    # reciprocal rank is 0.0 for non-found, otherwise 1/(rank+1) (for 0-based ranks)
    rr = (1./(ranks+1)).where(ranks<100, 0.0)  
    metrics['candMRR'] = rr.mean()


def add_hitpred_metrics(q, k, metrics):
    """
    Populate metrics with the hit prediction metrics:
    1) hit3pred_nonm90%_P@top0.1: we take queries with the highest predicted hit probabilities 
    and calculate the precision for them, i.e. the proportion of queries having the correct answer 
    among 3 best candidates returned. The number of queries taken is 0.1*len(matchable queries).
    nonm is the percentage of non-matchable queries.
    """
    field = f'matched_{k}'
    assert q[field].isnull().sum()==0, f'No hit probs for some queries for k={k}' 
    q = q.sort_values(by=field, ascending=False)
    
    non_matchable = q.true_rank.isnull()
    non_matchable_pct = int(non_matchable.mean()*100)
    # for now take 10% of the queries with largest predicted hit scores; TODO: draw some curves!
    topprop = 0.1
    topn = int((~non_matchable).sum()*topprop)  
    # metrics[f'hitpred_top{topprop}'] = topn
    prec = (q.head(topn).true_rank < k).mean()  # ranks are from 0!
    # print(k, prec, topn, int(non_matchable_prop*100)/100)
    metrics[f'hit{k}pred_nonm{non_matchable_pct}%_P@top{topprop}'] = prec
   
def score_part(true_path, pred_path, vis_dir=None):    
    df = load_preds(pred_path)
    df = merge_true_answers(df, true_path)
    write_true_ranks(df)
    if vis_dir is not None:
        vis_preds(df, true_path.parents[0], vis_dir)
    
    metrics = defaultdict(dict)
    # Candidate ranking metrics
    for case_type in ['hard','simple']:
        for type in ['lost','found']:
            ranks = df[(df.case_type==case_type)&(df.type==type)].true_rank
            add_cand_metrics(ranks, metrics[f'{case_type}_{type}'])

    # Hit probability prediction metrics
    for k in [1,3,10]:
        for case_type in ['hard','simple']:
            for type in ['lost','found']:
                mdf = df[(df.case_type==case_type)&(df.type==type)]
                mdf_nonmatchable = mdf[mdf.true_rank.isnull()]
                # generate more non-matchables preserving the hit_prob distribution, the number is such that
                # after concatenating with mdf there will be 90% of non-matchable examples.
                nonmatchable_90 = mdf_nonmatchable.sample(9*len(mdf)-10*len(mdf_nonmatchable),replace=True)
                for q in [mdf, pd.concat([mdf, nonmatchable_90], ignore_index=True)]:
                    add_hitpred_metrics(q, k,  metrics[f'{case_type}_{type}'])
                    
    return metrics

def write_metrics(metrics, part, method_descr, outdir):
    """
    Create a new folder, save all metrics to this folder. Print the most important metrics.
    """
    for k,v in metrics.items():
        sdf = pd.DataFrame(v, index=[0])
        sdf['part'] = f'{part}_{k}'
        sdf['method'] = method_descr
        out_path = outdir/f'{part}_{k}_results.tsv'
        sdf.to_csv(out_path, sep='\t', index=False)
        print(f"{k}, full results saved to:", out_path)
        print(sdf[['part']+[f'candR@{i}' for i in (10,100)]+[c for c in sdf.columns if 'hit10pred' in c]])
        print()
    return str(outdir)


HEIGHT= 384
def rescale_to_height(img, h):
    w = int(img.shape[1]* (h / img.shape[0]))
    return cv2.resize(img, (w, h))

def form_row(imgs):
    imgs = [rescale_to_height(x, HEIGHT) for x in imgs]
    return np.concatenate(imgs, axis=1)

def pad_len(img, max_len):
    return np.pad(img, [(0, 0), (0, max_len - img.shape[1]), (0, 0)])

def form_gallery(q, top, true_ans):
    q_row = form_row(q)
    top_rows = [form_row(x) for x in top]
    if true_ans is not None:
        true_ans_row = form_row(true_ans)
        
        true_ans_row_shape = true_ans_row.shape
    else:
        true_ans_row_shape = [0, 0]
    max_len = max([q_row.shape[1]] + [x.shape[1] for x in top_rows] + [true_ans_row_shape[1]])
    q_row = pad_len(q_row, max_len)
    top_rows = [pad_len(x, max_len) for x in top_rows]
    
    line = np.zeros((10, max_len, 3), dtype=np.uint8)
    line[:, :, 0] = 255
    concat = [q_row, line, *top_rows]
    if true_ans is not None:
        true_ans_row = pad_len(true_ans_row, max_len)
        concat += [line, true_ans_row]
    return np.concatenate(concat, axis=0)

def load_ad_imgs(path):
    imgs = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    ret = []
    for im in imgs:
        ret.append(cv2.imread(str(im))[:, :, ::-1]/255)
    return ret

def vis_row(row, queries, answers):
    row_answers = row['answer'].split(',')
    true_ans = row['answer_name']
    
    top = []
    for i in range(3):
        a = row_answers[i]
        top.append(load_ad_imgs(answers[a]))
    q = queries[row['query']]
    q = load_ad_imgs(q)
    true_ans = load_ad_imgs(answers[true_ans]) if isinstance(true_ans, str) else None
    gallery = form_gallery(q, top, true_ans)
    return gallery * 255

def vis_preds_type(pred_df, queries, answers, outdir):
    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df)):
        gallery = vis_row(row, queries, answers)
        name = f"{row['matched_1']:.3f}_{row['query']}.jpg"
        cv2.imwrite(str(outdir / name), gallery[:, :, ::-1])
def vis_preds(pred_df, data_path, outdir):
    # print(data_path.parents[0])
    answers = list((data_path/'found'/'synthetic_lost').glob('*')) + list((data_path/'lost'/'synthetic_found').glob('*'))
    queries = list((data_path/'found'/'found').glob('*')) + list((data_path/'lost'/'lost').glob('*'))
    answers = {x.name: x for x in answers}
    queries = {x.name: x for x in queries}
    outdir = outdir/'vis'
    outdir.mkdir(exist_ok=False)
    types = pred_df['case_type'].unique()
    for ad_type in pred_df['type'].unique():
        for c_type in types:
            type_outdir = outdir/ad_type/c_type
            type_outdir.mkdir(exist_ok=True, parents=True)
            df = pred_df[pred_df.case_type == c_type]
            df = df[df.type == ad_type]
            df = df.sort_values('matched_1',ascending = False).head(100)
            vis_preds_type(df, queries, answers, type_outdir)

def main(preds_path, data_dir, part, method_descr, visualise=False):
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
    outdir = Path('score_pets_'+timestr)
    outdir.mkdir(exist_ok=False)
    vis = outdir if visualise else None
    metrics = score_part( Path(data_dir) / part / 'registry.csv', preds_path, vis)
    outdir = write_metrics(metrics, part, method_descr, outdir)
    return outdir
    
def score_preds(preds_path, data_dir, parts, compressed):
    assert len(parts)==1, 'Only scoring on a single subset is currently supported.'
    part = parts[0]
    true_path = Path(data_dir) / part / 'registry.csv'
    return score_part(true_path, preds_path)


if __name__ == '__main__':
    Fire(main)
