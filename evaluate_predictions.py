from pathlib import Path
import argparse

from get_score import get_score_fct


def get_perfect_score(ground_truth_csv):
    return get_score_fct(
        path_to_prediction=str(ground_truth_csv),
        path_to_gt=str(ground_truth_csv),
        log=False,
        vid=False,
    )


def score_one(prediction_csv, ground_truth_csv, log=False, make_video=False, percent=False):
    score = get_score_fct(
        path_to_prediction=str(prediction_csv),
        path_to_gt=str(ground_truth_csv),
        log=log,
        vid=make_video,
    )

    perfect_score = None
    percentage = None

    if percent:
        perfect_score = get_perfect_score(ground_truth_csv)
        percentage = 100.0 * score / perfect_score if perfect_score > 0 else 0.0
        print(f"{Path(prediction_csv).name}: {score} / {perfect_score} ({percentage:.2f}%)")
    else:
        print(f"{Path(prediction_csv).name}: {score}")

    return score, perfect_score, percentage


def parse_args():
    parser = argparse.ArgumentParser(description="Score TRACO prediction CSV files.")

    parser.add_argument("--prediction", type=Path, help="Path to one prediction CSV.")
    parser.add_argument("--ground-truth", type=Path, help="Path to the matching ground-truth CSV.")
    parser.add_argument("--prediction-dir", type=Path, help="Directory of prediction CSVs.")
    parser.add_argument("--ground-truth-dir", type=Path, default=Path("training"))
    parser.add_argument("--log", action="store_true", help="Write get_score log files.")
    parser.add_argument("--make-video", action="store_true", help="Create scorer debug video.")
    parser.add_argument("--percent", action="store_true", help="Also show score as a percentage of perfect score.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.prediction is not None:
        if args.ground_truth is None:
            raise RuntimeError("--ground-truth is required with --prediction.")

        score_one(
            prediction_csv=args.prediction,
            ground_truth_csv=args.ground_truth,
            log=args.log,
            make_video=args.make_video,
            percent=args.percent,
        )
        return

    if args.prediction_dir is None:
        raise RuntimeError("Use --prediction or --prediction-dir.")

    total = 0
    perfect_total = 0
    count = 0

    for prediction_csv in sorted(args.prediction_dir.glob("*.csv")):
        ground_truth_csv = args.ground_truth_dir / prediction_csv.name

        if not ground_truth_csv.exists():
            print(f"Skipping {prediction_csv.name}: no ground truth at {ground_truth_csv}")
            continue

        score, perfect_score, _ = score_one(
            prediction_csv=prediction_csv,
            ground_truth_csv=ground_truth_csv,
            log=args.log,
            make_video=False,
            percent=args.percent,
        )
        total += score
        if perfect_score is not None:
            perfect_total += perfect_score
        count += 1

    if args.percent and perfect_total > 0:
        percentage = 100.0 * total / perfect_total
        print(f"Total score for {count} videos: {total} / {perfect_total} ({percentage:.2f}%)")
    else:
        print(f"Total score for {count} videos: {total}")


if __name__ == "__main__":
    main()
