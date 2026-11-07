import pandas as pd
import pytest

from src.common import contracts
from src.common.contracts import AREAS, CONTRACTS, load, path_for, validate


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_sample_file_matches_contract(name):
    df = pd.read_csv(path_for(name, sample=True))
    validate(name, df)
    assert 1 <= len(df) <= 10, "sample files should have 5 to 10 made-up rows"


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_real_file_matches_contract_if_present(name):
    real = path_for(name)
    if not real.exists():
        pytest.skip(f"{real.name} not produced yet")
    validate(name, pd.read_csv(real))


def test_area_list_is_the_agreed_eight():
    assert len(AREAS) == 8 and len(set(AREAS)) == 8


def test_validate_rejects_wrong_columns():
    with pytest.raises(ValueError):
        validate("network_edges", pd.DataFrame(columns=["source", "target"]))


def test_validate_rejects_unknown_area():
    df = pd.DataFrame([["2026-11-02", "LLM in finance", 1, 1.0]],
                      columns=CONTRACTS["crowding"][1])
    with pytest.raises(ValueError, match="areas not in the shared list"):
        validate("crowding", df)


def test_load_falls_back_to_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, "DATA_DIR", tmp_path)
    df = load("network_edges")
    assert list(df.columns) == CONTRACTS["network_edges"][1]
