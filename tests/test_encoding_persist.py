import numpy as np
import pandas as pd
import pytest

from src.common import contracts
from src.common.contracts import CONTRACTS, load, validate
from src.research_map.encoding import persist
from src.research_map.encoding.persist import (
    build_encodings_frame,
    load_encodings,
    parse_ids,
    parse_labels,
    parse_scores,
    save_encodings,
)
from src.research_map.encoding.store import save_vectors


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, "DATA_DIR", tmp_path)
    return tmp_path


def make_frame(ids=("W1", "W2", "W3")):
    return build_encodings_frame(
        list(ids),
        top_topic=[0, 1, 1][: len(ids)],
        assigned=[[(0, 0.9)], [(1, 0.6), (0, 0.3)], []][: len(ids)],
        labels={0: "alpha, beta", 1: "gamma"},
        model="m",
        text_quality=["full"] * len(ids),
    )


def test_frame_matches_contract_and_encodes_lists():
    frame = make_frame()
    validate("paper_encodings", frame)
    row = frame.iloc[1]
    assert row["topic_ids"] == "1;0" and row["topic_scores"] == "0.600;0.300"
    assert row["topic_labels"] == "gamma | alpha, beta"
    assert frame.iloc[2]["topic_ids"] == ""  # ambiguous paper: no topic above threshold


def test_parsers_round_trip_including_empty_and_nan():
    frame = make_frame()
    row = frame.iloc[1]
    assert parse_ids(row["topic_ids"]) == [1, 0]
    assert parse_scores(row["topic_scores"]) == [0.6, 0.3]
    assert parse_labels(row["topic_labels"]) == ["gamma", "alpha, beta"]
    assert parse_ids(np.nan) == [] and parse_ids("") == []


def test_save_then_load_round_trip(data_dir):
    frame, vectors = make_frame(), np.eye(3, 4, dtype=np.float32)
    csv, parquet = save_encodings(frame, vectors)
    assert csv.exists() and parquet.exists() and csv.parent == data_dir

    loaded = load_encodings()
    assert loaded.frame["paper_id"].tolist() == ["W1", "W2", "W3"]
    assert np.array_equal(loaded.vectors, vectors)
    assert loaded.frame.fillna("").iloc[1]["topic_ids"] == "1;0"


def test_load_reorders_vectors_to_csv_order(data_dir):
    frame = make_frame()
    save_encodings(frame, np.eye(3, 4, dtype=np.float32))
    save_vectors(["W3", "W1", "W2"], np.array([[3, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
                                              dtype=np.float32), "m", persist.vectors_path())
    assert load_encodings().vectors[:, 0].tolist() == [1, 2, 3]


def test_save_rejects_row_count_mismatch_and_writes_nothing(data_dir):
    with pytest.raises(ValueError, match="rows"):
        save_encodings(make_frame(), np.zeros((2, 4)))
    assert list(data_dir.iterdir()) == []


def test_save_rejects_duplicate_ids_and_mixed_models(data_dir):
    with pytest.raises(ValueError, match="duplicate"):
        save_encodings(make_frame(("W1", "W1", "W3")), np.zeros((3, 4)))
    mixed = make_frame()
    mixed.loc[0, "model"] = "other"
    with pytest.raises(ValueError, match="mixes"):
        save_encodings(mixed, np.zeros((3, 4)))


def test_save_rejects_wrong_columns(data_dir):
    with pytest.raises(ValueError, match="columns"):
        save_encodings(make_frame().drop(columns=["model"]), np.zeros((3, 4)))


def test_load_detects_id_mismatch_between_files(data_dir):
    save_encodings(make_frame(), np.eye(3, 4, dtype=np.float32))
    save_vectors(["W1", "W2", "WX"], np.eye(3, 4, dtype=np.float32), "m", persist.vectors_path())
    with pytest.raises(ValueError, match="disagree"):
        load_encodings()


def test_load_detects_model_mismatch(data_dir):
    save_encodings(make_frame(), np.eye(3, 4, dtype=np.float32))
    save_vectors(["W1", "W2", "W3"], np.eye(3, 4, dtype=np.float32), "different",
                 persist.vectors_path())
    with pytest.raises(ValueError, match="mismatch"):
        load_encodings()


def test_sample_loads_via_contract_and_ids_match_network_nodes(data_dir):
    df = load("paper_encodings")
    assert list(df.columns) == CONTRACTS["paper_encodings"][1]
    nodes = load("network_nodes")
    assert set(df["paper_id"]) == set(nodes["paper_id"])
    for _, row in df.iterrows():
        n = len(parse_ids(row["topic_ids"]))
        assert n == len(parse_scores(row["topic_scores"])) == len(parse_labels(row["topic_labels"]))
    assert load_encodings().vectors is None  # sample has no vectors
    assert (df["topic_ids"].str.count(";") >= 1).any()  # sample shows multi-topic papers
