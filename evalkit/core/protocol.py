"""
protocol.py — the frozen scoring settings, in one place.

WHY THIS EXISTS
---------------
Two people can run the same model on the same dataset and get different mAP if
one of them changed the confidence floor or the NMS IoU. Then the leaderboard is
measuring settings, not models. So every scoring knob lives here, and
run_eval.py takes its defaults from this file. Nobody has to remember them.

You may still override any of these on the command line — it's occasionally the
right thing when debugging — but the run is then stamped `protocol: MODIFIED`
in metrics.json and summary.txt, and compare.py flags it, because that number
is no longer comparable with everyone else's.

WHAT IS NOT FROZEN
------------------
Preprocessing: image size, letterbox vs stretch, colour space. Those belong to
the model, not the protocol. A net trained at 960 grayscale must be allowed to
run at 960 grayscale or its numbers are meaningless. The adapter reads them from
the checkpoint.

BUMPING THE VERSION
-------------------
Changing any value below invalidates comparison with older results, so bump
VERSION in the same commit. Old results keep their old stamp, and the mismatch
is then visible instead of silent.
"""

from __future__ import annotations

VERSION = "1.0"

#: Scoring knobs. Frozen — identical for every model, every run, every person.
PROTOCOL: dict = {
    # Detections below this are discarded before scoring. Deliberately tiny:
    # the PR curve and the recall-at-high-precision numbers need the low-score
    # tail. This is NOT a deployment threshold.
    "conf": 0.001,
    # NMS IoU used during evaluation.
    "nms_iou": 0.65,
    # Max detections kept per image.
    "max_det": 300,
    # The threshold a fielded system would actually run at. Drives the
    # confusion matrix, precision/recall/F1, and FP-per-empty-frame.
    "operating_conf": 0.25,
    # Split every model is scored on.
    "split": "val",
}

#: Keys a user is allowed to override without breaking comparability.
#: Everything else in PROTOCOL, if overridden, marks the run MODIFIED.
FREE_KEYS = frozenset({"split"})


def deviations(args_ns) -> dict:
    """Which frozen values did this run actually change? {} means comparable."""
    out = {}
    for key, frozen in PROTOCOL.items():
        if key in FREE_KEYS:
            continue
        actual = getattr(args_ns, key, frozen)
        if actual != frozen:
            out[key] = {"protocol": frozen, "used": actual}
    return out


def stamp(args_ns) -> dict:
    """The protocol block recorded in every metrics.json."""
    devs = deviations(args_ns)
    return {
        "version": VERSION,
        "status": "MODIFIED" if devs else "OK",
        "comparable": not devs,
        "frozen": dict(PROTOCOL),
        "deviations": devs,
    }
