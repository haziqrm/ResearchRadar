import numpy as np
import pandas as pd
import pytest

from src.common import contracts
from src.research_map.build_encodings import run_pipeline
from src.research_map.encoding.embed import FakeEmbedder
from src.research_map.encoding.groups import (
    GroupModel,
    assign_groups,
    build_groups_frame,
    group_model_path,
    groups_path,
    load_group_names,
    load_groups,
    save_groups,
    unique_names,
)
from src.research_map.encoding.topics import fit_topics
from tests.test_build_encodings import make_papers


def blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(3, 16) * 6
    return np.vstack([c + rng.normal(size=(40, 16)) for c in centers])


def test_frozen_model_matches_fitted_model():
    vectors = blobs()
    fitted = fit_topics(vectors, k_range=(2, 5))
    frozen = GroupModel.from_topic_model(fitted.model, "m")
    assert np.allclose(frozen.predict_proba(vectors), fitted.memberships)
    assert frozen.assign(fitted.memberships) == fitted.assigned


def test_save_load_round_trip_without_pickle(tmp_path):
    vectors = blobs()
    frozen = GroupModel.from_topic_model(fit_topics(vectors).model, "my-model")
    path = frozen.save(tmp_path / "g.npz")
    with np.load(path, allow_pickle=False) as data:  # plain arrays only
        assert "pca_components" in data.files
    loaded = GroupModel.load(path)
    assert loaded.embedding_model == "my-model"
    assert (loaded.temperature, loaded.threshold, loaded.max_topics) == (
        frozen.temperature, frozen.threshold, frozen.max_topics)
    assert np.array_equal(loaded.predict_proba(vectors), frozen.predict_proba(vectors))


def test_predict_rejects_wrong_dimension():
    frozen = GroupModel.from_topic_model(fit_topics(blobs()).model, "m")
    with pytest.raises(ValueError, match="dims"):
        frozen.predict_proba(np.zeros((2, 5)))


def test_unique_names_and_groups_frame():
    assert unique_names({0: "a", 1: "A", 2: " "}) == {0: "a", 1: "A (1)", 2: "group 2"}
    frame = build_groups_frame({0: "x, y", 1: "z"}, names={0: "First", 1: "Second"})
    assert frame.to_dict("records") == [
        {"group_id": 0, "name": "First", "keywords": "x, y"},
        {"group_id": 1, "name": "Second", "keywords": "z"},
    ]


def test_groups_csv_round_trip_and_mismatch_detection(tmp_path):
    frozen = GroupModel.from_topic_model(fit_topics(blobs(), k_range=(3, 3)).model, "m")
    frozen.save(group_model_path(tmp_path))
    save_groups(build_groups_frame({0: "a", 1: "b", 2: "c"}), tmp_path)
    model, names = load_groups(tmp_path)
    assert names == {0: "a", 1: "b", 2: "c"} and model.n_groups == 3

    save_groups(build_groups_frame({0: "a", 1: "b"}), tmp_path)
    with pytest.raises(ValueError, match="ids"):
        load_groups(tmp_path)


def test_missing_files_give_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_groups(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_group_names(tmp_path)


def test_pipeline_first_run_freezes_groups(tmp_path):
    result = run_pipeline(make_papers(), FakeEmbedder(dim=128), data_dir=tmp_path,
                          k_range=(2, 6))
    assert result.refitted
    assert group_model_path(tmp_path).exists() and groups_path(tmp_path).exists()
    names = load_group_names(tmp_path)
    # topic_labels in the encodings use the group names
    first = result.frame.iloc[0]
    assert first["topic_labels"].split(" | ")[0] == names[int(first["top_topic"])]


def test_ids_and_hand_edited_names_survive_new_papers(tmp_path):
    papers = make_papers()
    first = run_pipeline(papers.iloc[:30], FakeEmbedder(dim=128), data_dir=tmp_path,
                         k_range=(2, 6))
    # hand-edit names, as B would after a refit
    groups = pd.read_csv(groups_path(tmp_path))
    groups["name"] = [f"Named {g}" for g in groups["group_id"]]
    groups.to_csv(groups_path(tmp_path), index=False)

    second = run_pipeline(papers, FakeEmbedder(dim=128), data_dir=tmp_path)
    assert not second.refitted
    before = first.frame.set_index("paper_id")["top_topic"]
    after = second.frame.set_index("paper_id")["top_topic"].loc[before.index]
    assert before.equals(after)  # same papers keep the same group ids
    assert pd.read_csv(groups_path(tmp_path))["name"].str.startswith("Named").all()
    assert second.frame["topic_labels"].str.startswith("Named").all()


def test_refit_resets_groups(tmp_path):
    run_pipeline(make_papers(), FakeEmbedder(dim=128), data_dir=tmp_path)
    groups = pd.read_csv(groups_path(tmp_path))
    groups["name"] = [f"Named {g}" for g in groups["group_id"]]
    groups.to_csv(groups_path(tmp_path), index=False)
    again = run_pipeline(make_papers(), FakeEmbedder(dim=128), data_dir=tmp_path,
                         refit_groups=True)
    assert again.refitted
    assert not pd.read_csv(groups_path(tmp_path))["name"].str.startswith("Named").any()


def test_frozen_groups_reject_a_different_embedder(tmp_path):
    run_pipeline(make_papers(), FakeEmbedder(dim=64), data_dir=tmp_path)
    other = FakeEmbedder(dim=64)
    other.name = "different"
    with pytest.raises(ValueError, match="frozen groups"):
        run_pipeline(make_papers(), other, data_dir=tmp_path)


def test_pipeline_writes_real_groups_through_the_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, "DATA_DIR", tmp_path)
    run_pipeline(make_papers(), FakeEmbedder(dim=64))
    assert (tmp_path / "groups.csv").exists() and (tmp_path / "group_model.npz").exists()
    assert contracts.area_names() == list(load_group_names().values())


def test_assign_groups_for_new_papers(tmp_path):
    run_pipeline(make_papers(), FakeEmbedder(dim=128), data_dir=tmp_path, k_range=(2, 6))
    names = load_group_names(tmp_path)
    out = assign_groups(
        ["option hedging volatility", "bitcoin crypto exchange"],
        ["derivatives pricing smile.", "defi stablecoin liquidity."],
        embedder=FakeEmbedder(dim=128), data_dir=tmp_path,
    )
    assert list(out.columns) == ["area", "area_score", "group_ids", "group_scores"]
    assert set(out["area"]) <= set(names.values())
    assert out.loc[0, "area"] != out.loc[1, "area"]
    assert assign_groups([], embedder=FakeEmbedder(dim=128), data_dir=tmp_path).empty


def test_assign_groups_validation(tmp_path):
    run_pipeline(make_papers(), FakeEmbedder(dim=64), data_dir=tmp_path)
    with pytest.raises(ValueError, match="same length"):
        assign_groups(["a"], ["b", "c"], embedder=FakeEmbedder(dim=64), data_dir=tmp_path)
    other = FakeEmbedder(dim=64)
    other.name = "different"
    with pytest.raises(ValueError, match="fitted on"):
        assign_groups(["a"], embedder=other, data_dir=tmp_path)
