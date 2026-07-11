from pathlib import Path

import pytest

from astrogwb.config.batches import parse_mcmc_batch


def _batch() -> dict:
    return {
        "catalog": {"id": "bns-n16384-df1", "path": "/data/catalog.h5"},
        "chains_dir": "/data/chains",
        "runs": [
            {"campaign": "paper-h0", "config": "configs/run-a.json"},
            {"campaign": "paper-h0", "config": "configs/run-b.toml"},
        ],
    }


def test_parse_mcmc_batch_resolves_explicit_runs() -> None:
    batch = parse_mcmc_batch(_batch())

    assert batch.catalog_id == "bns-n16384-df1"
    assert batch.catalog_path == Path("/data/catalog.h5")
    assert batch.jax_platforms == "cuda"
    assert [(run.campaign, run.run) for run in batch.runs] == [
        ("paper-h0", "run-a"),
        ("paper-h0", "run-b"),
    ]


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"runs": []}, "non-empty list"),
        ({"catalog": {"id": "bad/id", "path": "catalog.h5"}}, "invalid catalog id"),
        (
            {
                "runs": [
                    {"campaign": "paper-h0", "config": "a/run.json"},
                    {"campaign": "paper-h0", "config": "b/run.json"},
                ]
            },
            "duplicate MCMC run",
        ),
        (
            {"runs": [{"campaign": "bad/campaign", "config": "run.json"}]},
            "invalid campaign",
        ),
        (
            {"runs": [{"campaign": "paper-h0", "config": "run.yaml"}]},
            "must have a .json or .toml extension",
        ),
    ],
)
def test_parse_mcmc_batch_rejects_invalid_batches(update: dict, match: str) -> None:
    raw = _batch()
    raw.update(update)

    with pytest.raises(ValueError, match=match):
        parse_mcmc_batch(raw)
