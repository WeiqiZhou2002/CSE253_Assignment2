from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, load_config


VIDEO_PLACEHOLDER = "REPLACE_WITH_PUBLIC_GOOGLE_DRIVE_MP4_OR_YOUTUBE_LINK"


def make_submission(config: dict[str, Any]) -> Path:
    """Copy final artifacts into the autograder-facing submission folder."""
    submission_dir = ensure_dir(ROOT / "submission")
    midi_dir = ROOT / config["paths"]["midi_dir"]
    workbook_html = ROOT / "notebooks" / "workbook.html"

    if workbook_html.exists():
        shutil.copy2(workbook_html, submission_dir / "workbook.html")
    for filename in ["symbolic_unconditioned.mid", "symbolic_conditioned.mid"]:
        src = midi_dir / filename
        if src.exists():
            shutil.copy2(src, submission_dir / filename)

    video_url = submission_dir / "video_url.txt"
    if not video_url.exists():
        video_url.write_text(VIDEO_PLACEHOLDER + "\n", encoding="utf-8")
    return submission_dir


def validate_submission(submission_dir: str | Path = ROOT / "submission") -> list[str]:
    """Return a list of local packaging issues."""
    submission_dir = Path(submission_dir)
    issues: list[str] = []
    required = ["workbook.html", "video_url.txt", "symbolic_unconditioned.mid", "symbolic_conditioned.mid"]
    for filename in required:
        path = submission_dir / filename
        if not path.exists():
            issues.append(f"missing {filename}")
        elif path.stat().st_size == 0:
            issues.append(f"empty {filename}")
    workbook = submission_dir / "workbook.html"
    if workbook.exists() and not workbook.read_text(encoding="utf-8", errors="ignore").startswith("<!DOCTYPE html>"):
        issues.append("workbook.html does not start with <!DOCTYPE html>")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    submission_dir = make_submission(config)
    issues = validate_submission(submission_dir)
    if issues:
        print("Submission issues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("Submission folder looks structurally complete.")


if __name__ == "__main__":
    main()

