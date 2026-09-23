"""
sequence.py — group a split's frames back into the videos they came from.

WHY THIS EXISTS
---------------
A plain detector sees one image and answers. A motion-based or tracking model
(GLAD, any DETR-with-tracking, anything with a Kalman filter) needs frames in
the order they were filmed, and needs its state cleared when one video ends and
the next begins. Feed such a model a shuffled split and it will look far worse
than it is; feed it one continuous stream across a video boundary and it will
carry a stale target across the cut.

So evaluation of those models needs two things a per-image harness doesn't:
frames in temporal order, and a known boundary between videos. This module
recovers both from the filenames, which is all a YOLO-format split gives us.

DEFAULT PATTERN
---------------
Matches the common "<video>_frame_<n>.<ext>" convention, e.g.

    DJI_20260420174320_0001_V_frame_000123.jpg
      -> sequence "DJI_20260420174320_0001_V", frame 123

and falls back to a trailing "_<n>" ("clip07_000123.png"). Override with
--sequence-regex when your naming differs; the regex needs one capture group for
the frame number, and everything before the match becomes the sequence name.

If nothing matches, every image becomes its own single-frame sequence — which
degrades to exactly the per-image behaviour, so a stateless model is unaffected.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Tried in order. One capture group = the frame index.
DEFAULT_PATTERNS = [
    r"_frame_(\d+)$",
    r"_f(\d+)$",
    r"_(\d+)$",
]


def parse_frame(stem: str, patterns: list[str]) -> tuple[str, int] | None:
    """'clip_frame_007' -> ('clip', 7). None when nothing matches."""
    for pat in patterns:
        m = re.search(pat, stem)
        if m:
            return stem[: m.start()], int(m.group(1))
    return None


def group_sequences(
    image_paths: list[Path],
    image_ids: list[int],
    pattern: str | None = None,
) -> list[tuple[str, list[int]]]:
    """
    Returns [(sequence_name, [image_id, ...]), ...].

    Sequences are ordered by name, frames within a sequence by their parsed
    frame number — NOT by filename, because zero-padding is not guaranteed and
    'frame_10' must not sort before 'frame_9'.

    Images whose names carry no frame number each become their own sequence, so
    a still-image dataset still works and stateless models see no change.
    """
    patterns = [pattern] if pattern else DEFAULT_PATTERNS

    grouped: dict[str, list[tuple[int, int]]] = {}
    singles: list[tuple[str, list[int]]] = []

    for path, img_id in zip(image_paths, image_ids):
        parsed = parse_frame(path.stem, patterns)
        if parsed is None:
            singles.append((path.stem, [img_id]))
            continue
        seq, frame_no = parsed
        grouped.setdefault(seq, []).append((frame_no, img_id))

    out: list[tuple[str, list[int]]] = [
        (seq, [img_id for _, img_id in sorted(frames)])
        for seq, frames in sorted(grouped.items())
    ]
    out.extend(sorted(singles))
    return out


def summarise(sequences: list[tuple[str, list[int]]]) -> dict:
    lengths = [len(ids) for _, ids in sequences]
    return {
        "sequences": len(sequences),
        "frames": sum(lengths),
        "shortest": min(lengths) if lengths else 0,
        "longest": max(lengths) if lengths else 0,
        "names": [name for name, _ in sequences],
    }
