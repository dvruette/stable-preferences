import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_score_progression(groups, score_key="hps", round_key="round", **kwargs):
    max_hps_per_round = groups[score_key].max()
    min_hps_per_round = groups[score_key].min()
    avg_hps_per_round = groups[score_key].mean()
    mean_max_hps_per_round = max_hps_per_round.groupby(round_key).mean()
    mean_min_hps_per_round = min_hps_per_round.groupby(round_key).mean()
    mean_avg_hps_per_round = avg_hps_per_round.groupby(round_key).mean()
    
    ts = np.arange(len(mean_max_hps_per_round)) + 1
    plt.plot(ts, mean_max_hps_per_round, label="max", **kwargs)
    plt.plot(ts, mean_min_hps_per_round, label="min", **kwargs)
    plt.plot(ts, mean_avg_hps_per_round, label="mean", **kwargs)
    plt.xticks(ts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_path', type=str, default="~/Downloads/prompt_dropout3")
    parser.add_argument('--ours_path', type=str, default="~/Downloads/prompt_dropout0")
    parser.add_argument('--output_path', type=str, default="outputs/plots")
    parser.add_argument('--metrics', type=str, default="metrics.csv")
    return parser.parse_args()

def main(args):
    df_baseline = pd.read_csv(os.path.join(args.baseline_path, "metrics.csv"))
    df_ours = pd.read_csv(os.path.join(args.ours_path, "metrics.csv"))
    
    os.makedirs(args.output_path, exist_ok=True)

    plt.figure()
    plot_score_progression(
        df_ours.groupby(["prompt_idx", "round"]),
        score_key="target_img_sim",
    )
    plt.xlabel("Round")
    plt.ylabel("CLIP similarity")
    plt.title("CLIP similarity progression per round")
    plt.legend()

    out_path = os.path.join(args.output_path, "target_sim_per_round.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

    base_max_sim_0 = df_baseline.loc[df_baseline["round"] <= 0].groupby("prompt_idx")["target_img_sim"].max().mean()
    base_max_sim_1 = df_baseline.loc[df_baseline["round"] <= 1].groupby("prompt_idx")["target_img_sim"].max().mean()
    base_max_sim_2 = df_baseline.loc[df_baseline["round"] <= 2].groupby("prompt_idx")["target_img_sim"].max().mean()
    ours_max_sim_0 = df_ours.loc[df_ours["round"] <= 0].groupby("prompt_idx")["target_img_sim"].max().mean()
    ours_max_sim_1 = df_ours.loc[df_ours["round"] <= 1].groupby("prompt_idx")["target_img_sim"].max().mean()
    ours_max_sim_2 = df_ours.loc[df_ours["round"] <= 2].groupby("prompt_idx")["target_img_sim"].max().mean()

    plt.figure()
    ts = [1, 2, 3]
    plt.plot(ts, [base_max_sim_0, base_max_sim_1, base_max_sim_2], label="baseline", linestyle="--", color="blue")
    plt.plot(ts, [ours_max_sim_0, ours_max_sim_1, ours_max_sim_2], label="ours", color="blue")
    plt.xticks(ts)
    plt.xlabel("Round")
    plt.ylabel("CLIP similarity")
    plt.title("Max CLIP similarity over all rounds")
    plt.legend()

    out_path = os.path.join(args.output_path, "global_max_target_sim.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

    pos_sim_1 = df_ours.loc[df_ours["round"] == 1]["pos_sim"].mean()
    neg_sim_1 = df_ours.loc[df_ours["round"] == 1]["neg_sim"].mean()
    pos_sim_2 = df_ours.loc[df_ours["round"] == 2]["pos_sim"].mean()
    neg_sim_2 = df_ours.loc[df_ours["round"] == 2]["neg_sim"].mean()

    plt.figure()
    ts = np.array([1, 2])  # Use numpy array for arithmetic operations
    width = 0.35  # Width of the bars
    padding = 0.025

    # Calculate the x-coordinates for the bars
    ts_pos = ts - width/2 - padding
    ts_neg = ts + width/2 + padding
    plt.bar(ts_pos, [pos_sim_1, pos_sim_2], width=width, label="positive")
    plt.bar(ts_neg, [neg_sim_1, neg_sim_2], width=width, label="negative")
    plt.xticks(ts)
    plt.ylim(75, None)
    plt.xlabel("Round")
    plt.ylabel("CLIP similarity")
    plt.legend()
    plt.title("CLIP similarity to feedback images")

    out_path = os.path.join(args.output_path, "feedback_sim.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")



if __name__ == "__main__":
    args = parse_args()
    main(args)
