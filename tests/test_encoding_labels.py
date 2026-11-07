import pytest

from src.research_map.encoding.labels import topic_labels

TEXTS = [
    "Deep hedging of derivatives with neural networks [SEP] We hedge option portfolios.",
    "Option pricing and implied volatility surfaces [SEP] Volatility smile modelling.",
    "Hedging options under volatility uncertainty",
    "Decentralized exchange liquidity pools [SEP] Automated market makers on Ethereum.",
    "Bitcoin and crypto token price dynamics [SEP] Crypto exchange liquidity.",
    "DeFi lending protocols and stablecoin risk",
]
TOPICS = [0, 0, 0, 1, 1, 1]


def test_labels_reflect_each_theme():
    labels = topic_labels(TEXTS, TOPICS)
    assert set(labels) == {0, 1}
    assert "volatility" in labels[0] or "hedging" in labels[0] or "option" in labels[0]
    assert "crypto" in labels[1] or "liquidity" in labels[1] or "defi" in labels[1]
    # labels of different topics do not share terms
    assert not set(labels[0].split(", ")) & set(labels[1].split(", "))


def test_template_and_boilerplate_words_are_not_labels():
    labels = topic_labels(
        ["Keywords: alpha. Topics: beta [SEP] this paper shows results"] * 2 + TEXTS,
        [0, 0] + TOPICS,
    )
    for label in labels.values():
        for banned in ("sep", "keywords", "topics", "paper", "results"):
            assert banned not in label.split()


def test_n_terms_cap_and_no_redundant_unigram():
    labels = topic_labels(TEXTS, TOPICS, n_terms=2)
    assert all(len(v.split(", ")) <= 2 for v in labels.values())
    for label in labels.values():
        terms = label.split(", ")
        for a in terms:
            for b in terms:
                if a != b:
                    assert not set(a.split()) <= set(b.split())


def test_missing_topics_and_empty_vocabulary_get_placeholders():
    labels = topic_labels(TEXTS, TOPICS, n_topics=4)
    assert labels[2] == "topic 2" and labels[3] == "topic 3"
    assert topic_labels(["the and of"], [0]) == {0: "topic 0"}
    assert topic_labels([], [], n_topics=2) == {0: "topic 0", 1: "topic 1"}


def test_invalid_arguments():
    with pytest.raises(ValueError):
        topic_labels(["a"], [0, 1])
    with pytest.raises(ValueError):
        topic_labels(["a"], [0], n_terms=0)
