import pandas as pd
import pytest

from src.common import contracts
from src.common.contracts import CONTRACTS, area_names, load, path_for, save, validate


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_sample_file_matches_contract(name):
    df = pd.read_csv(path_for(name, sample=True))
    validate(name, df, sample=True)
    assert 1 <= len(df) <= 10, "sample files should have 5 to 10 made-up rows"


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_real_file_matches_contract_if_present(name):
    real = path_for(name)
    if not real.exists():
        pytest.skip(f"{real.name} not produced yet")
    validate(name, pd.read_csv(real))


def test_area_list_comes_from_groups_and_is_unique():
    names = area_names(sample=True)
    assert names == pd.read_csv(path_for("groups", sample=True))["name"].tolist()
    assert len(names) == len(set(names)) > 0


def test_areas_attribute_matches_area_names():
    from src.common.contracts import AREAS

    assert AREAS == area_names()


def test_real_groups_take_priority(tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, "DATA_DIR", tmp_path)
    assert area_names() == area_names(sample=True)
    save("groups", pd.DataFrame({"group_id": [0, 1], "name": ["Alpha", "Beta"],
                                 "keywords": ["a", "b"]}))
    assert area_names() == ["Alpha", "Beta"]
    validate("crowding", pd.DataFrame([["2026-11-02", "Beta", 1, 1.0]],
                                      columns=CONTRACTS["crowding"][1]))


def test_validate_rejects_wrong_columns():
    with pytest.raises(ValueError):
        validate("network_edges", pd.DataFrame(columns=["source", "target"]))


def test_validate_rejects_unknown_area():
    df = pd.DataFrame([["2026-11-02", "LLM in finance", 1, 1.0]],
                      columns=CONTRACTS["crowding"][1])
    with pytest.raises(ValueError, match="areas not in the shared list"):
        validate("crowding", df)


@pytest.mark.parametrize("names", [["A", "A"], ["A", "a"], ["A", " "]])
def test_validate_rejects_bad_group_names(names):
    df = pd.DataFrame({"group_id": [0, 1], "name": names, "keywords": ["", ""]})
    with pytest.raises(ValueError):
        validate("groups", df)


def test_load_falls_back_to_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, "DATA_DIR", tmp_path)
    df = load("network_edges")
    assert list(df.columns) == CONTRACTS["network_edges"][1]
    assert set(load("network_nodes")["area"]) <= set(area_names(sample=True))
