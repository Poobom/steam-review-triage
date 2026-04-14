"""Stage 1: Load Steam reviews CSV and repair broken UTF-8 encoding."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["review_id", "text", "lang", "recommended", "playtime_hours", "posted_at"]

MOJIBAKE_MARKERS = re.compile(r"[ÂÃâãÅÄÆÇèéêëÌÍÎÏðñòóôõö÷øùúûü°ª¬«»¿¡§©®™]")
HANGUL_SYLLABLE = re.compile(r"[가-힣]")
ASCII_LETTER = re.compile(r"[A-Za-z]")
PRINTABLE_NOISE = re.compile(r"[^\w\s가-힣.,!?'\"()\-:;/&%$#@*+=]")


@dataclass
class CleanStats:
    input_count: int
    output_count: int
    encoding_repaired: int
    encoding_failed: int
    lang_distribution: dict
    repaired_examples: list  # before/after pairs (max 5)


def _score_text(text: str) -> float:
    """Higher score = more likely to be intact human text.

    Penalizes mojibake markers, rewards Hangul + ASCII letter density.
    """
    if not text:
        return -1.0
    n = len(text)
    mojibake_count = len(MOJIBAKE_MARKERS.findall(text))
    hangul_count = len(HANGUL_SYLLABLE.findall(text))
    ascii_count = len(ASCII_LETTER.findall(text))
    noise_count = len(PRINTABLE_NOISE.findall(text))

    signal = (hangul_count + ascii_count) / n
    noise = (mojibake_count * 3 + noise_count) / n
    return signal - noise


def repair_text(text: str) -> tuple[str, bool]:
    """Attempt latin-1 -> utf-8 repair. Pick whichever scores higher.

    Returns (best_text, was_repaired_and_better).
    """
    if not isinstance(text, str) or not text:
        return text or "", False

    try:
        repaired = text.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text, False

    if repaired == text:
        return text, False

    if _score_text(repaired) > _score_text(text):
        return repaired, True
    return text, False


def load_and_clean(input_path: Path) -> tuple[pd.DataFrame, CleanStats]:
    df = pd.read_csv(input_path, encoding="utf-8")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["text"] = df["text"].fillna("").astype(str)

    repaired_count = 0
    repaired_examples: list = []
    new_texts = []
    for original in df["text"]:
        repaired, did_repair = repair_text(original)
        new_texts.append(repaired)
        if did_repair:
            repaired_count += 1
            if len(repaired_examples) < 5:
                repaired_examples.append({"before": original[:120], "after": repaired[:120]})

    df["text"] = new_texts
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    stats = CleanStats(
        input_count=len(new_texts),
        output_count=len(df),
        encoding_repaired=repaired_count,
        encoding_failed=0,
        lang_distribution=df["lang"].value_counts().to_dict(),
        repaired_examples=repaired_examples,
    )
    return df, stats


def save_clean_artifact(df: pd.DataFrame, stats: CleanStats, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "01_clean.json"
    payload = {
        "stats": asdict(stats),
        "rows": df.to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_clean_artifact(out_dir: Path) -> tuple[pd.DataFrame, CleanStats]:
    payload = json.loads((out_dir / "01_clean.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(payload["rows"])
    stats = CleanStats(**payload["stats"])
    return df, stats
