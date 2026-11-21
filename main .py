"""Forecast GTA VI trophy earn rates at a case's target (make.fun
gta6-trophy-rarity-forecast-01), Python 3.13 standard library only.

Method, per trophy (deterministic, no network, no randomness):
  1. Take every read in the case's history. Convert each read's time to
     x = ln(1 + hours since the game's release, 2026-11-19T00:00:00Z)
     and its earn rate to y = logit(rate), with the rate clamped to
     0.01..99.99 percent first.
  2. Fit a straight line y = a + b*x by least squares over the most
     recent 36 hours of reads (all reads when fewer than 4 fall in
     that window). Trophy earn rates grow roughly like a power law in
     time since release, which is exactly a line on this scale, and
     the recent window keeps the fit in the current regime (the
     tracked-player population grows and early reads come from a
     tiny, hardcore sample).
  3. Read the line at the target's x, then shrink it 15% toward the
     last read's logit and blend 25% toward the last read: a trend
     fitted on noisy, regime-shifting data overshoots, and the blend
     is the median. Cap the move from the last read at 2.5 logit
     units so one steep fit can never throw a forecast away.
  4. Spread: a random-walk term (0.30 logit per sqrt day, tighter
     than the house baseline's 0.50 because the trend already
     explains most of the move) combined with the fit's residual
     spread, widened by the extrapolation leverage at the target.
     Floored at 0.35 and capped at 1.0 logit units, so the 5%-95%
     band stays honest: wide enough to cover a real move, narrow
     enough not to pay for width at every trophy.
  5. Quantiles are median + z_tau * spread on the logit scale,
     converted back to percent and rounded to 4 decimals, half up.

When the history is flat (as on launch night, before the first
refresh) the fit is flat, the forecast is persistence with the
floored spread, and the program still writes a valid file.

How the grader runs it: `python3 -P -s /bot/main.py` in
python:3.13-slim, standard library only: no network, no pip
packages. It reads every <id>.in in the input directory (the
grader runs one case at a time, with only that case's file there)
and writes <id>.out to the output directory:
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}.

Outside the grader's container, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Fit window: the most recent this many hours of reads (all reads if fewer than 4 in it).
WINDOW_HOURS = 36.0
# Shrink the fitted trend this much toward the last read, and blend this much of the last read in.
SHRINK = 0.85
BLEND_LAST = 0.25
# Never move the median more than this many logit units from the last read.
MAX_MOVE = 2.5
# Random-walk spread, logit units per square-root day.
SIGMA_RW = 0.30
# Spread floor and cap, logit units.
SPREAD_MIN = 0.35
SPREAD_MAX = 1.0


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_std, x_mean, sxx, n)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, mx, sxx, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid, mx, sxx, n


def forecast_trophy(series, target_x, horizon_days):
    """series: list of (x, y) oldest first. Returns five quantiles in percent."""
    if not series:
        # No history at all for this trophy: a flat 1% with the floored spread.
        med = logit(1.0)
        spread = SPREAD_MIN
    else:
        last_x, last_y = series[-1][0], series[-1][1]
        # Recent window, falling back to all reads when it holds too few.
        cutoff = last_x_time(series) - WINDOW_HOURS
        window = [(x, y) for (x, y, t) in series if t >= cutoff]
        if len(window) < 4:
            window = [(x, y) for (x, y, t) in series]
        xs = [x for x, y in window]
        ys = [y for x, y in window]
        a, b, resid, mx, sxx, n = ols(xs, ys)
        trend = a + b * target_x
        # Shrink toward the last read and blend some of it in.
        trend = last_y + (trend - last_y) * SHRINK
        med = (1.0 - BLEND_LAST) * trend + BLEND_LAST * last_y
        # Cap the move from the last read.
        med = last_y + max(-MAX_MOVE, min(MAX_MOVE, med - last_y))
        # Spread: random walk plus fit residual, widened by leverage at the target.
        if sxx > 0 and n > 0:
            lev = math.sqrt(1.0 + 1.0 / n + (target_x - mx) ** 2 / sxx)
        else:
            lev = 1.0
        spread = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2 + (resid * lev) ** 2 + 0.10 ** 2)
        spread = max(SPREAD_MIN, min(SPREAD_MAX, spread))
    return [round4(inv_logit(med + z * spread)) for z in Z]


def last_x_time(series):
    return series[-1][2]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    target_x = math.log(1.0 + max(0.0, (target - RELEASE).total_seconds() / 3600.0))
    horizon_days = case["horizon_minutes"] / 1440.0
    # Per-trophy series: (x, logit(rate), hours_since_release).
    per_trophy = [[] for _ in case["trophies"]]
    for read in case["history"]:
        at = parse_time(read["at"])
        hours = max(0.0, (at - RELEASE).total_seconds() / 3600.0)
        x = math.log(1.0 + hours)
        rates = read["rates"]
        for i in range(len(case["trophies"])):
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            per_trophy[i].append((x, logit(rate), hours))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rows.append({"id": trophy["id"], "q": forecast_trophy(per_trophy[i], target_x, horizon_days)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
