DATASET=$1


for part in dev_small dev test; do
    python gen_golden_upperline.py ${DATASET}/${part}
    python ../score_pets.py preds.tsv ${DATASET} $part "upperbound: gold answers"
    mv preds.tsv ${part}_gold_preds.tsv
done

for part in dev_small dev test; do 
    python gen_random_baseline.py ${DATASET}/${part}
    python ../score_pets.py preds.tsv ${DATASET} $part "baseline: random predictions"
    mv preds.tsv ${part}_random_preds.tsv
done
