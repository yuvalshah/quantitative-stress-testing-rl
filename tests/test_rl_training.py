"""Stage 5 PPO training, checkpoints, reproducibility, experiment pipeline and data separation.

Tiny budgets on synthetic data (SYN tickers) so the tests stay fast and offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from rlstress.config import config_from_dict  # noqa: E402
from rlstress.data.storage import write_frame  # noqa: E402
from rlstress.rl.evaluation import evaluate_policy, summarize  # noqa: E402
from rlstress.rl.experiment import final_test, init_run, load_run, train_all, validate_run  # noqa: E402
from rlstress.rl.training import build_env, load_policy, train_ppo, vecnormalize_path  # noqa: E402
from rlstress.shocks import fit_shock_model  # noqa: E402

from conftest import stage4_raw_config  # noqa: E402

TINY = {"total_timesteps": 256, "n_steps": 64, "batch_size": 32, "n_epochs": 2, "eval_freq": 128,
        "checkpoint_freq": 128, "selection_episodes": 6, "selection_seed": 10_000, "validation_episodes": 5,
        "eval_seed_sets": [20_000, 30_000], "test_episodes": 5, "test_seed": 90_000, "seeds": [0],
        "plausibility_thresholds": [0.01, 0.02], "n_representative_scenarios": 3}


def _toml(d: dict, prefix: str = "") -> str:
    """Minimal TOML writer for the project config dict (tables, arrays of tables, scalars, lists)."""
    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return repr(v)
    lines, sub = [], []
    for k, v in d.items():
        if isinstance(v, dict):
            sub.append((k, v, False))
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            sub.append((k, v, True))
        else:
            lines.append(f"{k} = {val(v)}")
    out = "\n".join(lines) + ("\n" if lines else "")
    for k, v, is_array in sub:
        path = f"{prefix}{k}"
        if is_array:
            for item in v:
                out += f"\n[[{path}]]\n" + _toml(item, path + ".")
        else:
            out += f"\n[{path}]\n" + _toml(v, path + ".")
    return out


def _config_dict(**training) -> dict:
    raw = stage4_raw_config()
    raw["training"].update({**TINY, **training})
    return raw


@pytest.fixture()
def tiny(tmp_path, s4_source):
    raw = _config_dict()
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(_toml(raw))
    data_path = tmp_path / "consolidated_daily.csv"
    write_frame(s4_source, data_path)
    return config_from_dict(raw), cfg_path, data_path


def test_toml_writer_roundtrip(tiny) -> None:
    config, cfg_path, _ = tiny
    from rlstress.config import load_config
    assert load_config(cfg_path) == config


def _train(config, source, tmp_path: Path, seed: int = 0, name: str = "run"):
    s, e = config.data.splits.train
    model = fit_shock_model(source, config.portfolio.tickers, config.simulation, s, e)
    return model, train_ppo(config, source, model, 0.01, seed, tmp_path / name)


def test_short_training_saves_artifacts(tmp_path, s4_source, tiny) -> None:
    config, _, _ = tiny
    _, res = _train(config, s4_source, tmp_path)
    d = res.run_dir
    for f in ("best_model.zip", "best_vecnormalize.pkl", "final_model.zip", "final_vecnormalize.pkl",
              "progress.csv", "training_episodes.csv", "selection_log.json", "train_meta.json"):
        assert (d / f).exists(), f
    assert res.timesteps >= 256 and any((d / "checkpoints").iterdir())
    eps = pd.read_csv(d / "training_episodes.csv")
    comp = [c for c in eps.columns if c.startswith("reward_")]
    np.testing.assert_allclose(eps[comp].sum(axis=1), eps["total_reward"], atol=1e-9)
    assert set(eps["termination"]) <= {"valid_breach", "implausible_breach", "horizon", "invalid"}


def test_fixed_seed_reproduces_training_and_evaluation(tmp_path, s4_source, tiny) -> None:
    config, _, _ = tiny
    model, a = _train(config, s4_source, tmp_path, name="a")
    _, b = _train(config, s4_source, tmp_path, name="b")
    pa = pd.read_csv(a.run_dir / "training_episodes.csv")
    pb = pd.read_csv(b.run_dir / "training_episodes.csv")
    pd.testing.assert_frame_equal(pa, pb)
    env = build_env(config, s4_source, model, 0.01, "validation")
    ra = summarize(evaluate_policy(env, load_policy(a.run_dir, env), range(500, 510)))
    rb = summarize(evaluate_policy(env, load_policy(b.run_dir, env), range(500, 510)))
    assert ra == rb


def test_checkpoint_load_and_evaluation_does_not_modify_model(tmp_path, s4_source, tiny) -> None:
    config, _, _ = tiny
    model, res = _train(config, s4_source, tmp_path)
    env = build_env(config, s4_source, model, 0.01, "validation")
    pol = load_policy(res.run_dir, env, "final")
    before = {k: v.clone() for k, v in pol.model.policy.state_dict().items()}
    mean_before = pol.normalizer.mean.copy()
    evaluate_policy(env, pol, range(600, 620))
    after = pol.model.policy.state_dict()
    assert all(np.array_equal(before[k].numpy(), after[k].numpy()) for k in before)
    np.testing.assert_array_equal(mean_before, pol.normalizer.mean)
    obs, _ = env.reset(seed=1)
    a = pol.act(obs)
    assert a.shape == (4,) and np.all(np.abs(a) <= 1)


def test_resume_from_checkpoint(tmp_path, s4_source, tiny) -> None:
    config, _, _ = tiny
    model, res = _train(config, s4_source, tmp_path)
    ckpt = sorted((res.run_dir / "checkpoints").glob("model_*_steps.zip"))[0]
    assert vecnormalize_path(ckpt).exists()
    resumed = train_ppo(config, s4_source, model, 0.01, 0, tmp_path / "resumed", 128, resume_from=ckpt)
    assert resumed.timesteps > int(ckpt.stem.split("_")[1])


def test_experiment_pipeline_and_final_test_guard(tmp_path, tiny) -> None:
    _, cfg_path, data_path = tiny
    run = init_run(cfg_path, data_path, tmp_path / "results")
    assert run.name.startswith("run_")
    train_all(run)
    with pytest.raises(FileExistsError):
        train_all(run)  # a second training of the same (threshold, seed) is refused
    summary = validate_run(run)
    for f in ("validation/metrics.json", "validation/episodes.csv", "validation/threshold_comparison.csv",
              "validation/reward_components.csv", "validation/representative_scenarios.json",
              "selection.json", "summary.json", "stage5_report.md"):
        assert (run / f).exists(), f
    assert summary["selection"]["uses_test_data"] is False
    assert not (run / "test").exists()  # validation never touches the test split
    eps = pd.read_csv(run / "validation" / "episodes.csv")
    assert eps["anchor_date"].between("2019-01-01", "2019-12-31").all()
    counts = eps.groupby(["threshold", "policy"]).size()
    assert counts.nunique() == 1  # every policy gets the same number of episodes
    with pytest.raises(FileExistsError):
        validate_run(run)
    with pytest.raises(PermissionError):
        final_test(run)
    out = final_test(run, confirm=True)
    assert out["results"] and (run / "test" / "test_evaluation.json").exists()
    with pytest.raises(FileExistsError, match="already run"):
        final_test(run, confirm=True)


def test_run_shock_model_ignores_validation_and_test_rows(tmp_path, s4_source, tiny) -> None:
    _, cfg_path, data_path = tiny
    corrupted = s4_source.copy()
    later = corrupted.index > pd.Timestamp("2018-12-31")
    corrupted.loc[later, ["SPY_return", "QQQ_return", "SYNA_return"]] = 0.3
    bad_path = tmp_path / "corrupted.csv"
    write_frame(corrupted, bad_path)
    clean = init_run(cfg_path, data_path, tmp_path / "a")
    dirty = init_run(cfg_path, bad_path, tmp_path / "b")
    a = json.loads((clean / "shock_model.json").read_text())
    b = json.loads((dirty / "shock_model.json").read_text())
    assert a == b and a["fit_end"] <= "2018-12-31"
    config, source, model, _ = load_run(clean)
    train_env = build_env(config, source, model, 0.01, config.simulation.anchor_window)
    assert train_env.market_states.index.max() <= pd.Timestamp("2018-12-31")


def test_overrides_recorded_and_applied(tmp_path, tiny) -> None:
    from rlstress.rl.experiment import apply_overrides

    config, cfg_path, data_path = tiny
    run = init_run(cfg_path, data_path, tmp_path / "r", overrides={"training.ent_coef": 0.01,
                                                                    "training.plausibility_thresholds": [0.02]})
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["overrides"]["training.ent_coef"] == 0.01 and meta["thresholds"] == [0.02]
    loaded, _, _, _ = load_run(run)
    assert loaded.training.ent_coef == 0.01 and loaded.training.plausibility_thresholds == (0.02,)
    with pytest.raises(ValueError, match="unknown"):
        apply_overrides(config, {"training.not_a_field": 1})
    from rlstress.config import ConfigError
    with pytest.raises(ConfigError):
        apply_overrides(config, {"training.gamma": 2.0})
