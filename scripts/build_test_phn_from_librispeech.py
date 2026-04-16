#!/usr/bin/env python3
"""
Build test.phn (and optionally valid.phn) aligned with clustering manifests.

Each line = space-separated ARPAbet phones, using the same lexicon as
prepare_text.sh (lexicon_filtered.lst). Normalization matches
normalize_and_filter_text.py (punctuation stripped; words lowercased;
only words present in the lexicon are kept), then word → phones as in
phonemize_with_sil with sil_prob=0 and no surround (deterministic).

Usage (from wav2vec_unsupervised root, with venv active if needed):
  python scripts/build_test_phn_from_librispeech.py \\
    --test-tsv data/clustering/librispeech/precompute_pca512_cls128_mean_pooled/test.tsv \\
    --lexicon data/text/lexicon_filtered.lst \\
    --librispeech-root data/corpora/LibriSpeech \\
    --split test-clean \\
    --out data/clustering/librispeech/precompute_pca512_cls128_mean_pooled/test.phn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import regex as regex_mod  # same as normalize_and_filter_text.py (\p{L} etc.)


def load_lexicon(path: Path) -> dict[str, list[str]]:
    wrd_to_phn: dict[str, list[str]] = {}
    with path.open() as lf:
        for line in lf:
            items = line.rstrip().split()
            if len(items) < 2:
                continue
            word, phones = items[0], items[1:]
            if word not in wrd_to_phn:
                wrd_to_phn[word] = phones
    return wrd_to_phn


def normalize_text(line: str) -> str:
    # Match normalize_and_filter_text.py
    filter_r = regex_mod.compile(r"[^\p{L}\p{N}\p{M}\' \-]")
    line = filter_r.sub(" ", line)
    return " ".join(line.split())


def transcript_for_utt(librispeech_root: Path, split: str, utt_id: str) -> str | None:
    """utt_id like 1089-134691-0025."""
    parts = utt_id.split("-")
    if len(parts) != 3:
        return None
    book, chapter, _ = parts
    trans_path = librispeech_root / split / book / chapter / f"{book}-{chapter}.trans.txt"
    if not trans_path.is_file():
        return None
    prefix = utt_id + " "
    with trans_path.open() as f:
        for line in f:
            if line.startswith(prefix):
                return line[len(prefix) :].strip()
            if line.startswith(utt_id + "\t"):  # tab-separated variant
                return line.split("\t", 1)[1].strip()
    return None


def words_to_phones(words: list[str], wrd_to_phn: dict[str, list[str]]) -> list[str]:
    phones: list[str] = []
    for w in words:
        if w in wrd_to_phn:
            phones.extend(wrd_to_phn[w])
    return phones


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-tsv", type=Path, required=True)
    p.add_argument("--lexicon", type=Path, required=True)
    p.add_argument("--librispeech-root", type=Path, required=True)
    p.add_argument("--split", default="test-clean", help="LibriSpeech subfolder (test-clean / dev-clean)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    wrd_to_phn = load_lexicon(args.lexicon)
    lines_out: list[str] = []

    with args.test_tsv.open() as f:
        header = f.readline()
        if not header.strip():
            print("empty tsv", file=sys.stderr)
            sys.exit(1)
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname = line.split("\t", 1)[0].strip()
            utt_id = Path(fname).stem
            trans = transcript_for_utt(args.librispeech_root, args.split, utt_id)
            if trans is None:
                print(f"warning: no transcript for {utt_id}", file=sys.stderr)
                phones: list[str] = []
            else:
                norm = normalize_text(trans).lower()
                words = norm.split()
                words = [w for w in words if w in wrd_to_phn]
                phones = words_to_phones(words, wrd_to_phn)
            if not phones:
                lines_out.append("<SIL>")
            else:
                lines_out.append(" ".join(phones))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for ln in lines_out:
            out.write(ln + "\n")

    print(f"wrote {len(lines_out)} lines to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
