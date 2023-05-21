import argparse

from stable_preferences.evaluation.automatic_eval.clip_score import ClipScore
from stable_preferences.evaluation.automatic_eval.hps import HumanPreferenceScore
from stable_preferences.evaluation.automatic_eval.pap import PickAPicScore


def clip_score(image_path, description):
    measure = ClipScore()
    print(
        f"Calculating CLIP score for image {image_path} and description '{description}'"
    )
    return measure.compute_clip_score_from_path(image_path, description)


def hpc_score(image_path, description):
    measure = HumanPreferenceScore(weight_path="./resources/hpc.pt")
    print(
        f"Calculating HPC score for image {image_path} and description '{description}'"
    )
    return measure.compute_from_paths(description, [image_path]).item()


def pap_score(image_path, description):
    measure = PickAPicScore()
    print(
        f"Calculating PAP score for image {image_path} and description '{description}'"
    )
    return measure.compute_from_paths(description, [image_path]).item()


def analyze(args):
    results = dict()

    if args.clip:
        results["clip_score"] = clip_score(args.image_path, args.prompt)

    if args.hps:
        results["human_preference_score"] = hpc_score(args.image_path, args.prompt)

    if args.pap:
        results["pick_a_pick_score"] = pap_score(args.image_path, args.prompt)

    print(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="Path to the image.")
    parser.add_argument("--prompt", help="Description used to generate the image.")
    parser.add_argument("--clip", action="store_true", help="Calculate CLIP score.")
    parser.add_argument("--hps", action="store_true", help="Calculate HPC score.")
    parser.add_argument("--pap", action="store_true", help="Calculate PAP score.")

    args = parser.parse_args()
    analyze(args)
