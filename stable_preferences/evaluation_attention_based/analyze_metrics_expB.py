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


def plot_score_progression(groups, score_key="hps", round_key="round", confidence=0.95, add_confidence_interval=True):
    max_hps_per_round = groups[score_key].max()
    min_hps_per_round = groups[score_key].min()
    avg_hps_per_round = groups[score_key].mean()
    mean_max_hps_per_round = max_hps_per_round.groupby(round_key).mean()
    mean_min_hps_per_round = min_hps_per_round.groupby(round_key).mean()
    mean_avg_hps_per_round = avg_hps_per_round.groupby(round_key).mean()
    
    ts = np.arange(len(mean_max_hps_per_round)) + 1
    if add_confidence_interval:
        ci_max = confidence_interval(max_hps_per_round, groupby=round_key, confidence=confidence)
        ci_min = confidence_interval(min_hps_per_round, groupby=round_key, confidence=confidence)
        ci_mean = confidence_interval(avg_hps_per_round, groupby=round_key, confidence=confidence)
        plt.errorbar(ts, mean_max_hps_per_round, yerr=(ci_max[1] - ci_max[0])/2, capsize=4, label="Baseline")
        plt.errorbar(ts, mean_min_hps_per_round, yerr=(ci_min[1] - ci_min[0])/2, capsize=4, label="Baseline")
        plt.errorbar(ts, mean_avg_hps_per_round, yerr=(ci_mean[1] - ci_mean[0])/2, capsize=4, label="Baseline")
    else:
        plt.plot(ts, mean_max_hps_per_round, label="Baseline", linestyle=':', color='C0')
        plt.plot(ts, mean_min_hps_per_round, label="Baseline", linestyle=':', color='C1')
        plt.plot(ts, mean_avg_hps_per_round, label="Baseline", linestyle=':', color='C2')
    plt.xticks(ts)

def plot_double_score_progression(groups1, groups2, score_key1="", score_key2="", round_key="round", confidence=0.95, add_confidence_interval=True):
    max_hps_per_round1 = groups1[score_key1].max()
    min_hps_per_round1 = groups1[score_key1].min()
    avg_hps_per_round1 = groups1[score_key1].mean()
    mean_max_hps_per_round1 = max_hps_per_round1.groupby(round_key).mean()
    mean_min_hps_per_round1 = min_hps_per_round1.groupby(round_key).mean()
    mean_avg_hps_per_round1 = avg_hps_per_round1.groupby(round_key).mean()
    
    max_hps_per_round2 = groups2[score_key2].max()
    min_hps_per_round2 = groups2[score_key2].min()
    avg_hps_per_round2 = groups2[score_key2].mean()
    mean_max_hps_per_round2 = max_hps_per_round2.groupby(round_key).mean()
    mean_min_hps_per_round2 = min_hps_per_round2.groupby(round_key).mean()
    mean_avg_hps_per_round2 = avg_hps_per_round2.groupby(round_key).mean()
    
    ts = np.arange(len(mean_max_hps_per_round1)) + 1
    if add_confidence_interval:
        ci_max1 = confidence_interval(max_hps_per_round1, groupby=round_key, confidence=confidence)
        ci_min1 = confidence_interval(min_hps_per_round1, groupby=round_key, confidence=confidence)
        ci_mean1 = confidence_interval(avg_hps_per_round1, groupby=round_key, confidence=confidence)
        plt.errorbar(ts, mean_max_hps_per_round1, yerr=(ci_max1[1] - ci_max1[0])/2, capsize=4, label="Ours (max)")
        plt.errorbar(ts, mean_min_hps_per_round1, yerr=(ci_min1[1] - ci_min1[0])/2, capsize=4, label="Ours (mean)")
        plt.errorbar(ts, mean_avg_hps_per_round1, yerr=(ci_mean1[1] - ci_mean1[0])/2, capsize=4, label="Ours (min)")
        
        ci_max2 = confidence_interval(max_hps_per_round2, groupby=round_key, confidence=confidence)
        ci_min2 = confidence_interval(min_hps_per_round2, groupby=round_key, confidence=confidence)
        ci_mean2 = confidence_interval(avg_hps_per_round2, groupby=round_key, confidence=confidence)
        plt.errorbar(ts, mean_max_hps_per_round2, yerr=(ci_max2[1] - ci_max2[0])/2, capsize=4, label="Ours (max) pos only", linestyle="--", color="C0")
        plt.errorbar(ts, mean_min_hps_per_round2, yerr=(ci_min2[1] - ci_min2[0])/2, capsize=4, label="Ours (mean) pos only", linestyle="--", color="C1")
        plt.errorbar(ts, mean_avg_hps_per_round2, yerr=(ci_mean2[1] - ci_mean2[0])/2, capsize=4, label="Ours (min) pos only", linestyle="--", color="C2")
    else:
        plt.plot(ts, mean_max_hps_per_round1, label="Ours (max)")
        plt.plot(ts, mean_min_hps_per_round1, label="Ours (mean)")
        plt.plot(ts, mean_avg_hps_per_round1, label="Ours (min)")
        
        plt.plot(ts, mean_max_hps_per_round2, label="Ours (max) pos only", linestyle="--", color="C0")
        plt.plot(ts, mean_min_hps_per_round2, label="Ours mean pos only", linestyle="--", color="C1")
        plt.plot(ts, mean_avg_hps_per_round2, label="Ours (min) pos only", linestyle="--", color="C2")
    plt.xticks(ts)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/local/home/jthomm/stable-preferences/outputs/target_curated_dataset/outputs/rounds/2023-05-29/experiment_6")
    parser.add_argument('--path_comp', type=str, default="/local/home/jthomm/stable-preferences/outputs/target_curated_dataset/outputs/rounds/2023-05-29/experiment_5")
    parser.add_argument('--path_baseline', type=str, default="/local/home/jthomm/stable-preferences/outputs/target_curated_dataset/outputs/rounds/2023-05-29/experiment_7")
    parser.add_argument('--metrics', type=str, default="metrics.csv")
    return parser.parse_args()

def main(args):
    df = pd.read_csv(os.path.join(args.path, "metrics.csv"))
    df_comp = pd.read_csv(os.path.join(args.path_comp, "metrics.csv"))
    df_baseline = pd.read_csv(os.path.join(args.path_baseline, "metrics.csv"))

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

    if "target_img_sim" in df.columns:
        plt.figure()
        fig, ax = plt.subplots(figsize=(9, 6))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # plot_score_progression(
        #     df.groupby(["prompt_idx", "round"]),
        #     score_key="target_img_sim",
        #     add_confidence_interval=False,
        #     confidence=.95,
        # )
        # fig, ax = plt.subplots(figsize=(15, 6))
        plot_double_score_progression(
            df.groupby(["prompt_idx", "round"]),
            df_comp.groupby(["prompt_idx", "round"]),
            score_key1="target_img_sim",
            score_key2="target_img_sim",
            add_confidence_interval=False,
            confidence=.95,
        )
        plot_score_progression(
            df_baseline.groupby(["prompt_idx", "round"]),
            score_key="target_img_sim",
            add_confidence_interval=False,
            confidence=.95,
        )
        plt.xlabel("Round")
        plt.ylabel("CLIP Similarity")
        plt.legend(bbox_to_anchor=(1, 0.8))

        out_path = os.path.join(args.path, "target_sim_per_round.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved plot to {out_path}")

    plt.figure()
    max_hps_round0 = df.loc[df["round"] == 0].groupby("prompt_idx")["hps"].max()
    max_hps_round1 = df.loc[df["round"] == 1].groupby("prompt_idx")["hps"].max()
    max_hps_round2 = df.loc[df["round"] == 2].groupby("prompt_idx")["hps"].max()
    (max_hps_round1 - max_hps_round0).hist(bins=20, alpha=0.5, density=True, label="round 1 - round 0")
    (max_hps_round2 - max_hps_round1).hist(bins=20, alpha=0.5, density=True, label="round 2 - round 1")
    plt.xlabel("Change in HPS")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Change in max. HPS per round")
    out_path = os.path.join(args.path, "max_hps_change.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

    plt.figure()
    mean_hps_round0 = df.loc[df["round"] == 0].groupby("prompt_idx")["hps"].mean()
    hps_std_round0 = df.loc[df["round"] == 0].groupby("prompt_idx")["hps"].std().mean()
    mean_hps_round1 = df.loc[df["round"] == 1].groupby("prompt_idx")["hps"].mean()
    mean_hps_round2 = df.loc[df["round"] == 2].groupby("prompt_idx")["hps"].mean()
    # hps_std_round0.hist(bins=20, density=True, alpha=0.5, label="round 0")
    (mean_hps_round1 - mean_hps_round0).hist(bins=20, alpha=0.5, density=True, label="round 1 - round 0")
    (mean_hps_round2 - mean_hps_round1).hist(bins=20, alpha=0.5, density=True, label="round 2 - round 1")
    plt.xlabel("Change in HPS")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Change in mean HPS per round")
    out_path = os.path.join(args.path, "mean_hps_change.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
    print()


if __name__ == "__main__":
    args = parse_args()
    main(args)
