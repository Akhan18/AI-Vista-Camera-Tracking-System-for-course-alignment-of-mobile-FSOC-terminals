"""
AI-VISTA One-Command External MP4 Evaluator

Usage:
    Put an evaluator MP4 inside data/input/
    Then run:

        python3 run_evaluator.py

The script:
- finds the MP4 in data/input/
- runs the existing YOLO MP4 evaluator
- keeps the input folder clean
- moves generated evaluation files to data/output/
"""

from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

EVALUATOR = PROJECT_ROOT / "tools" / "evaluate_mp4.py"
YOLO_MODEL = PROJECT_ROOT / "models" / "yolo" / "beacon_best.pt"


def main():

    print("=" * 50)
    print(" AI-VISTA ONE-COMMAND MP4 EVALUATOR")
    print("=" * 50)

    # Check required folders/files
    if not INPUT_DIR.exists():
        print(f"\nERROR: Input directory not found:")
        print(f"  {INPUT_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EVALUATOR.exists():
        print(f"\nERROR: Evaluator not found:")
        print(f"  {EVALUATOR}")
        return 1

    if not YOLO_MODEL.exists():
        print(f"\nERROR: YOLO model not found:")
        print(f"  {YOLO_MODEL}")
        return 1

    # Find MP4 files
    videos = sorted(INPUT_DIR.glob("*.mp4"))

    if not videos:
        print("\nNo MP4 file found.")
        print("\nPut the evaluator video here:")
        print(f"  {INPUT_DIR}")
        return 1

    # Select video
    if len(videos) > 1:

        print("\nMultiple MP4 files found:\n")

        for index, video in enumerate(videos, start=1):
            print(f"  {index}. {video.name}")

        print()

        while True:

            choice = input(
                "Enter the number of the video to evaluate: "
            ).strip()

            try:
                selected = int(choice)

                if 1 <= selected <= len(videos):
                    video_path = videos[selected - 1]
                    break

            except ValueError:
                pass

            print("Invalid selection. Try again.")

    else:
        video_path = videos[0]

    print("\nSelected video:")
    print(f"  {video_path}")

    print("\nYOLO model:")
    print(f"  {YOLO_MODEL}")

    print("\nStarting evaluator...\n")

    # Run existing evaluator
    command = [
        sys.executable,
        str(EVALUATOR),
        "--video",
        str(video_path),
        "--detector",
        "yolo",
        "--yolo-model",
        str(YOLO_MODEL),
        "--save-video",
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:

        print("\n" + "=" * 50)
        print(" EVALUATION FAILED")
        print("=" * 50)

        return result.returncode

    # --------------------------------------------------
    # Move generated evaluation files from input
    # to output.
    #
    # The evaluator names them using the video stem:
    #   video_yolo_evaluation.csv
    #   video_yolo_evaluated.mp4
    # --------------------------------------------------

    generated_files = [
        video_path.with_name(
            video_path.stem + "_yolo_evaluation.csv"
        ),
        video_path.with_name(
            video_path.stem + "_yolo_evaluated.mp4"
        ),
    ]

    print("\nMoving evaluation results to data/output/...")

    for source_file in generated_files:

        if source_file.exists():

            destination = OUTPUT_DIR / source_file.name

            shutil.move(
                str(source_file),
                str(destination),
            )

            print(f"  Moved: {destination.name}")

    print("\n" + "=" * 50)
    print(" EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 50)

    print("\nInput:")
    print(f"  {video_path}")

    print("\nResults:")
    print(f"  {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())