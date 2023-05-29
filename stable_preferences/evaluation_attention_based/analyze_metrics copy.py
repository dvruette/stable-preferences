import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def confidence_interval(data, groupby=None, confidence=0.95):
    if groupby is None:
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        return m-h, m+h
    else:
        lower, upper = [], []
        for i, xs in data.groupby(groupby):
            a = 1.0 * np.array(xs)
            n = len(a)
            m, se = np.mean(a), stats.sem(a)
            h = se * stats.t.ppf((1 + confidence) / 2., n-1)
            lower.append(m-h)
            upper.append(m+h)
        return np.array(lower), np.array(upper)


def plot_score_progression(groups, score_key="hps", round_key="round", confidence=0.95, std=False):
    max_hps_per_round = groups[score_key].max()
    min_hps_per_round = groups[score_key].min()
    avg_hps_per_round = groups[score_key].mean()
    mean_max_hps_per_round = max_hps_per_round.groupby(round_key).mean()
    mean_min_hps_per_round = min_hps_per_round.groupby(round_key).mean()
    mean_avg_hps_per_round = avg_hps_per_round.groupby(round_key).mean()

    ts = np.arange(len(mean_max_hps_per_round)) + 1
    
    if std==True:
        # convert max_hps_per_round to a torch tensor with shape (num_rounds, num_prompts)
        max_hps_per_round_t = max_hps_per_round.unstack()
        min_hps_per_round_t = min_hps_per_round.unstack()
        avg_hps_per_round_t = avg_hps_per_round.unstack()
        # in each row, subtract the value in the first column from all other columns
        max_hps_per_round_t2 = max_hps_per_round_t.sub(max_hps_per_round_t.iloc[:, 0], axis=0)
        min_hps_per_round_t2 = min_hps_per_round_t.sub(min_hps_per_round_t.iloc[:, 0], axis=0)
        avg_hps_per_round_t2 = avg_hps_per_round_t.sub(avg_hps_per_round_t.iloc[:, 0], axis=0)
        # convert back to a pandas dataframe
        max_hps_per_round_t3 = max_hps_per_round_t2.stack()
        min_hps_per_round_t3 = min_hps_per_round_t2.stack()
        avg_hps_per_round_t3 = avg_hps_per_round_t2.stack()
        std_max_hps_per_round = max_hps_per_round_t3.groupby(round_key).std()
        std_min_hps_per_round = min_hps_per_round_t3.groupby(round_key).std()
        std_avg_hps_per_round = avg_hps_per_round_t3.groupby(round_key).std()
        ci_max = (mean_max_hps_per_round - std_max_hps_per_round, mean_max_hps_per_round + std_max_hps_per_round)
        ci_min = (mean_min_hps_per_round - std_min_hps_per_round, mean_min_hps_per_round + std_min_hps_per_round)
        ci_mean = (mean_avg_hps_per_round - std_avg_hps_per_round, mean_avg_hps_per_round + std_avg_hps_per_round)
    else:
        ci_max = confidence_interval(max_hps_per_round, groupby=round_key, confidence=confidence)
        ci_min = confidence_interval(min_hps_per_round, groupby=round_key, confidence=confidence)
        ci_mean = confidence_interval(avg_hps_per_round, groupby=round_key, confidence=confidence)
       
    plt.errorbar(ts, mean_max_hps_per_round, yerr=(ci_max[1] - ci_max[0])/2, capsize=4, label="max")
    plt.errorbar(ts, mean_min_hps_per_round, yerr=(ci_min[1] - ci_min[0])/2, capsize=4, label="min")
    plt.errorbar(ts, mean_avg_hps_per_round, yerr=(ci_mean[1] - ci_mean[0])/2, capsize=4, label="mean")
    plt.xticks(ts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/local/home/jthomm/stable-preferences/stable_preferences/evaluation/pos_exp_14")
    parser.add_argument('--metrics', type=str, default="metrics.csv")
    return parser.parse_args()

def main(args):
    df = pd.read_csv(os.path.join(args.path, "metrics.csv"))

    if "prompt_idx" not in df.columns:
        # remove all rows that have non-unique prompts
        prompt_counts = df["prompt"].value_counts()
        normal_count = prompt_counts.min()
        duplicate_prompts = prompt_counts[prompt_counts > normal_count].index
        df = df[~df["prompt"].isin(duplicate_prompts)]
        # assign a prompt_idx to each prompt
        df["prompt_idx"] = df.groupby("prompt").ngroup()

    plot_score_progression(
        df.groupby(["prompt_idx", "round"]),
        score_key="hps",
    )
    plt.xlabel("Round")
    plt.ylabel("HPS")
    plt.legend()

    out_path = os.path.join(args.path, "hps_per_round.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

    plt.figure()
    plot_score_progression(
        df.groupby(["prompt_idx", "round"]),
        score_key="pos_sim",
    )
    plot_score_progression(
        df.groupby(["prompt_idx", "round"]),
        score_key="neg_sim",
    )
    plt.xlabel("Round")
    plt.ylabel("CLIP Similarity")
    plt.legend()

    out_path = os.path.join(args.path, "clip_sim_per_round.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
    print()

    if "target_img_sim" in df.columns:
        plt.figure()
        plot_score_progression(
            df.groupby(["prompt_idx", "round"]),
            score_key="target_img_sim",
            confidence=0.80,
            std=False,
        )
        plt.xlabel("Round")
        plt.ylabel("Target Image Similarity")
        plt.legend()

        out_path = os.path.join(args.path, "target_img_sim_per_round.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved plot to {out_path}")
        print()




if __name__ == "__main__":
    args = parse_args()
    main(args)
