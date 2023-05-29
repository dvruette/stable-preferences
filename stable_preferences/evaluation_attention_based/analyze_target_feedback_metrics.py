import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_score_progression(df1, df2, label1, label2, score_key="target_img_sim"):
    groups1 = df1.groupby(["prompt_idx", "round"])
    groups2 = df2.groupby(["prompt_idx", "round"])
    max_per_round1 = groups1[score_key].max().groupby("round").mean()
    min_per_round1 = groups1[score_key].min().groupby("round").mean()
    mean_per_round1 = groups1[score_key].mean().groupby("round").mean()
    max_per_round2 = groups2[score_key].max().groupby("round").mean()
    min_per_round2 = groups2[score_key].min().groupby("round").mean()
    mean_per_round2 = groups2[score_key].mean().groupby("round").mean()
    
    ts = [1, 2, 3]
    plt.plot(ts, max_per_round1, label=f"{label1} (max)", linestyle="--")
    plt.plot(ts, min_per_round1, label=f"{label1} (min)", linestyle="--")
    plt.plot(ts, mean_per_round1, label=f"{label1} (mean)", linestyle="--")
    plt.plot(ts, max_per_round2, label=f"{label2} (max)", color="C0")
    plt.plot(ts, min_per_round2, label=f"{label2} (min)", color="C1")
    plt.plot(ts, mean_per_round2, label=f"{label2} (mean)", color="C2")
    plt.xticks(ts)
    plt.legend()


def plot_max_progression(df1, df2, label1, label2, score_key="target_img_sim"):
    max_sim1_0 = df1.loc[df1["round"] <= 0].groupby("prompt_idx")[score_key].max().mean()
    max_sim1_1 = df1.loc[df1["round"] <= 1].groupby("prompt_idx")[score_key].max().mean()
    max_sim1_2 = df1.loc[df1["round"] <= 2].groupby("prompt_idx")[score_key].max().mean()
    max_sim2_0 = df2.loc[df2["round"] <= 0].groupby("prompt_idx")[score_key].max().mean()
    max_sim2_1 = df2.loc[df2["round"] <= 1].groupby("prompt_idx")[score_key].max().mean()
    max_sim2_2 = df2.loc[df2["round"] <= 2].groupby("prompt_idx")[score_key].max().mean()

    ts = [1, 2, 3]
    plt.plot(ts, [max_sim1_0, max_sim1_1, max_sim1_2], label=label1, linestyle="--", color="C0")
    plt.plot(ts, [max_sim2_0, max_sim2_1, max_sim2_2], label=label2, color="C0")
    plt.xticks(ts)
    plt.legend()


def feedback_similarity_bar_plot(df):
    pos_sim_1 = df.loc[df["round"] == 1]["pos_sim"].mean()
    neg_sim_1 = df.loc[df["round"] == 1]["neg_sim"].mean()
    pos_sim_2 = df.loc[df["round"] == 2]["pos_sim"].mean()
    neg_sim_2 = df.loc[df["round"] == 2]["neg_sim"].mean()

    ts = np.array([1, 2])  # Use numpy array for arithmetic operations
    width = 0.35  # Width of the bars
    padding = 0.025

    # Calculate the x-coordinates for the bars
    ts_pos = ts - width/2 - padding
    ts_neg = ts + width/2 + padding
    plt.bar(ts_pos, [pos_sim_1, pos_sim_2], width=width, label="positive")
    plt.bar(ts_neg, [neg_sim_1, neg_sim_2], width=width, label="negative")
    plt.xticks(ts)
    plt.ylim(50, None)
    plt.xlabel("Round")
    plt.ylabel("CLIP similarity")
    plt.legend()
    plt.title("CLIP similarity to feedback images")


def plot_diversity_progression(df1, df2, label1, label2):
    diversity_baseline = 100 - df1.groupby(["round", "prompt_idx"])["round_diversity"].mean().groupby("round").mean()
    diversity_ours = 100 - df2.groupby(["round", "prompt_idx"])["round_diversity"].mean().groupby("round").mean()

    ts = [1, 2, 3]
    plt.plot(ts, diversity_baseline, label=label1, linestyle="--", color="C0")
    plt.plot(ts, diversity_ours, label=label2, color="C0")
    plt.xticks(ts)
    plt.xlabel("Round")
    plt.ylabel("In-batch diversity")
    plt.title("In-batch image diversity per round")
    plt.legend()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path1', type=str, default="~/Downloads/prompt_dropout3/metrics.csv")
    parser.add_argument('--input_path2', type=str, default="~/Downloads/prompt_dropout0/metrics.csv")
    parser.add_argument('--label1', type=str, default='dropout=0.3')
    parser.add_argument('--label2', type=str, default='dropout=0.0')
    parser.add_argument('--output_path', type=str, default="outputs/plots")
    return parser.parse_args()


def main(args):
    df1 = pd.read_csv(os.path.join(args.input_path1))
    df2 = pd.read_csv(os.path.join(args.input_path2))
    
    os.makedirs(args.output_path, exist_ok=True)

    plt.figure()
    plot_score_progression(df1, df2, args.label1, args.label2)
    plt.xlabel("Round")
    plt.ylabel("CLIP similarity")
    plt.title("CLIP similarity progression per round")

    out_path = os.path.join(args.output_path, f"target_sim_per_round.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")


    plt.figure()
    plot_max_progression(df1, df2, args.label1, args.label2)
    plt.xlabel("Round")
    plt.ylabel("CLIP similarity")
    plt.title("Max CLIP similarity over all rounds")

    out_path = os.path.join(args.output_path, "global_max_target_sim.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

    plt.figure()
    feedback_similarity_bar_plot(df1)
    out_path = os.path.join(args.output_path, f"feedback_sim_{args.label1}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

    plt.figure()
    feedback_similarity_bar_plot(df2)
    out_path = os.path.join(args.output_path, f"feedback_sim_{args.label2}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

    plt.figure()
    plot_diversity_progression(df1, df2, args.label1, args.label2)
    out_path = os.path.join(args.output_path, "diversity.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")



if __name__ == "__main__":
    args = parse_args()
    main(args)
