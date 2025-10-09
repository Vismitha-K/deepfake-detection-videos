# project/compare_and_stats.py
import os, json
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def read_metrics(json_dir):
    files = [f for f in os.listdir(json_dir) if f.endswith("_metrics.json")]
    summary = {}
    for f in files:
        mname = f.replace("_metrics.json","")
        with open(os.path.join(json_dir, f), "r") as fh:
            summary[mname] = json.load(fh)
    return summary

def load_preds(csv_path):
    df = pd.read_csv(csv_path)
    return df["label"].values, df["pred"].values

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    args = parser.parse_args()

    # collect metrics
    summ = read_metrics(args.eval_dir)
    print("Model metrics:")
    for k,v in summ.items():
        print(k, v)

    # find top model by f1
    best = max(summ.items(), key=lambda kv: kv[1]["f1"])[0]
    print("Best model:", best)

    # load per-frame CSVs
    models = list(summ.keys())
    perfs = {}
    dfs = {}
    for m in models:
        csv_path = os.path.join(args.eval_dir, f"{m}_per_frame.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        dfs[m] = df
        perfs[m] = {
            "accuracy": accuracy_score(df.label, df.pred),
            "precision": precision_score(df.label, df.pred, zero_division=0),
            "recall": recall_score(df.label, df.pred, zero_division=0),
            "f1": f1_score(df.label, df.pred, zero_division=0)
        }

    # plot bar chart for f1 and accuracy
    models_sorted = sorted(perfs.keys())
    accs = [perfs[m]["accuracy"] for m in models_sorted]
    f1s = [perfs[m]["f1"] for m in models_sorted]

    x = range(len(models_sorted))
    plt.figure(figsize=(8,4))
    plt.bar(x, accs)
    plt.xticks(x, models_sorted)
    plt.ylabel("Accuracy")
    plt.title("Model comparison - Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(args.eval_dir, "accuracy_comparison.png"))

    plt.figure(figsize=(8,4))
    plt.bar(x, f1s)
    plt.xticks(x, models_sorted)
    plt.ylabel("F1 Score")
    plt.title("Model comparison - F1")
    plt.tight_layout()
    plt.savefig(os.path.join(args.eval_dir, "f1_comparison.png"))

    # McNemar test: compare best against others on per-sample predictions
    results = {}
    best_df = dfs[best]
    for m, df in dfs.items():
        if m == best:
            continue
        # ensure same ordering / same samples
        # use filepath column to align
        merged = best_df[["filepath","label","pred"]].merge(df[["filepath","pred"]], on="filepath", suffixes=("_best","_other"))
        y_true = merged["label"].values
        pred_best = merged["pred_best"].values
        pred_other = merged["pred_other"].values

        # build contingency table
        # b = count(pred_best correct, pred_other wrong)
        # c = count(pred_best wrong, pred_other correct)
        b = ((pred_best == y_true) & (pred_other != y_true)).sum()
        c = ((pred_best != y_true) & (pred_other == y_true)).sum()
        table = [[0, b],[c,0]]
        # run McNemar
        res = mcnemar(table, exact=False, correction=True)
        results[m] = {"b": int(b), "c": int(c), "statistic": float(res.statistic), "pvalue": float(res.pvalue)}

    with open(os.path.join(args.eval_dir, "mcnemar_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Saved comparison plots and McNemar test results.")