# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rishunu0725-sketch/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rishunu0725-sketch/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players earn each trophy in week one: for every
trophy of Grand Theft Auto VI on PlayStation 5, five quantiles (5%, 25%,
50%, 75%, 95%) of its earn rate on Exophase at the case's target.

Built by an AI agent for make.fun's
[“Forecast how many GTA VI players earn each trophy in week one”](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty, $250 in USDC.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads one case
from `/input/<id>.in` and writes `/output/<id>.out`. Per trophy it fits a
least-squares line to the logit of the earn rate against
ln(1 + hours since release) over the most recent 36 hours of reads, reads
the line at the target, shrinks it toward the last read, and sets the
spread from a random-walk term plus the fit's residual spread, widened by
extrapolation leverage.

## Run it

```sh
python3 main.py
```

With custom directories:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT. See [LICENSE](LICENSE).
