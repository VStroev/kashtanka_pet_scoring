# Kashtanka pet scoring tools 

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

**Before commiting scripts should be tented with**
```
pandas=0.25.0
numpy=1.14.0
```