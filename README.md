# RS-RL: Reinforcement-Learning Stress Testing with a Portfolio-Aware Scenario Recommender

Classic stress tests apply a few hand-written shocks ("what if the market falls 20%?"). This project instead uses
reinforcement learning to **search** for 5-day market-shock paths that

1. breach a portfolio's risk limits, and
2. stay **plausible**: no more extreme than the most extreme 1% of real 5-day market windows (2015–2021).

The discovered paths become a validated scenario library. A recommender then picks a short, diverse, explained
shortlist of scenarios for any given portfolio.

**Authors:** [Yuval Shah](https://github.com/yuvalshah) · Vedant Pandey · Riddhima Shah · Tushar Varshney

## Results

| Method (validation, q = 0.01) | Valid plausible breaches |
|---|---|
| Random actions | 0% (47% breach, none plausible) |
| Continuous Monte Carlo, 6,000 paths | 0 valid of 2,027 breaches |
| PPO (stable-baselines3, continuous actions) | 14.8% ± 12.1% |
| Scripted decline (−0.15 on every factor, every day) | 88.5% |
| **Tabular Q-learning, β = 5 (plausibility-margin bonus)** | **99.9%**, median plausibility rank 0.042 |
| Tabular Q-learning, β = 0 | 100%, median plausibility rank 0.021 |

- At q = 0.01, PPO gets stuck on implausible crashes. Q-learning on a 9-action, 120-state discretisation matches a
  dynamic-programming benchmark.
- The adaptive Q-policy beats open-loop replay of its own paths (about 83% valid).
- 8,961 candidate paths reduce to 566 unique valid breaches, 16 clusters and 8 validated scenarios. These feed a
  diversity-aware top-5 recommender.
- The final test on held-out 2024–2025 data was run **once**, with a frozen pipeline and checksums of every artifact.
  The frozen PPO reached 22.8% valid breaches, against 25.5% on validation.

## Pipeline

| Stage | Notebook | What it does |
|---|---|---|
| 1 | `stage_1.ipynb` | Configuration, portfolio (8 US stocks, equal weight, USD 100k) |
| 2 | `stage_2.ipynb` | Data download (Yahoo Finance, FRED), cleaning, features, chronological splits |
| 3 | `stage_3.ipynb` | Risk engine: VaR / CVaR, drawdown, Euler risk contributions, limit checks |
| 4 | `stage_4.ipynb` | Factor + Student-t shock model, plausibility scoring, Gymnasium `StressTestEnv` |
| 5 | `stage_5.ipynb` | Scenario discovery: Monte Carlo, Q-learning, Double Q, n-step, DP, exhaustive enumeration |
| 6 | `stage_6.ipynb` | Validated scenario library: families, clustering, robustness |
| 7 | `stage_7.ipynb` | Portfolio-aware recommender: eligibility, multi-criteria score, MMR diversity, explanations |
| 7.2 | `stage_7_2_unseen_evaluation.ipynb` | Larger catalog, real-data evaluation, 2000–2014 backcast |
| 8 | `stage_8_final_test.ipynb` | One-shot out-of-sample test on 2024–2025 |

The notebooks are committed with their recorded outputs and charts, so the results can be read without running them.

## Repository layout

```
src/rlstress/        library code (layered: data → portfolio/risk/stress → shocks/env → rl/search → library/recommender)
  data/              download, cleaning, alignment, splits
  portfolio/ risk/ stress/   valuation, VaR/CVaR, drawdown, contributions, limits, historical stress
  shocks/ env.py     factor model, multivariate Student-t, plausibility, Gymnasium environment
  rl/ search.py      PPO training/evaluation, tabular Q-learning, Monte Carlo, DP, enumeration
  scenario_library.py, scenario_recommender.py, unseen_evaluation.py, final_evaluation.py
tests/               offline test suite, including a layered-architecture check
config/default.toml  every parameter: portfolio, limits, simulation, training, splits, seeds
examples/            command-line demos and PPO train/evaluate scripts
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,rl]"
pytest                                   # offline, no data or API key needed

cp .env.example .env                     # add a free FRED API key, needed only for Stage 2 macro data
jupyter notebook stage_2.ipynb           # downloads and processes the data into data/
```

Data and result files are not committed. Stage 2 downloads the data, and each later notebook writes its outputs to
`results/`. The protocol is train 2015–2021 for fitting and training, validation 2022–2023 for selection, and test
2024–2025 used once. Episode seed ranges are disjoint between selection, reporting and testing.

*This is a research prototype for a university course. It is not investment advice or a production risk system.*
