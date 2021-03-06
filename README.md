# Kashtanka pet scoring tools 

## Introduction
Tools for evaluation of search quality for data from [kashtanka pet project](https://kashtanka.pet). It includes scoring script and baselines (random and upperbound). Baselines can be useful as example of preporation of submission file. 

## Submission file format
Submission should be tsv file with following columns: 
* **query** - name of query folder
* **matched_1** - expected probability that first answer is true answer
* **matched_3** - expected probability that true answer is among top 3 answer according to score
* **matched_10** - expected probability that true answer is among top 10 answer according to score
* **answer** - string contatining top 100 answer names according to score separated by ',' (`aba67b58ed,4b701568a2,244f13a50c,...`) 

## Usage
Suppose `${DATASET}` is the path to the dataset.

To run the random baseline on the dev set:

```python baselines/gen_random_baseline.py ${DATASET}/dev```

To evaluate the predictions on the dev set:

```python score_pets.py preds.tsv ${DATASET} dev "baseline: random predictions"```

All baselines can be run and evaluated with:
```
cd baselines
bash run_baselines.sh
```


## Metrics
* **cand_recall@k:** R@k: the recall@k (0 or 1, because there is only 1 true answer for each query) averaged over 
    queries, i.e. the proportion of queries for which the true answer is among k top candidates returned).
    These metrics are calculated on matchable queries only.
* **hit3pred_nonm90%_P@top0.1:** we take queries with the highest predicted hit probabilities 
    and calculate the precision for them, i.e. the proportion of queries having the correct answer 
    among 3 best candidates returned. The number of queries taken is 0.1*len(matchable queries).
    nonm is the percentage of non-matchable queries.

# Before commiting scripts should work with
```
pandas=0.25.0
numpy=1.14.0
```
