from hypoforge.vocabulary import (
    equivalent_aliases,
    load_vocabulary,
    normalize_entity,
    related_terms,
)


def test_science125_questions_are_all_covered():
    covered = {
        number
        for entry in load_vocabulary()["entries"]
        for number in entry.get("question_numbers", [])
    }
    assert covered == set(range(1, 126))


def test_safe_cross_lingual_aliases_merge():
    assert normalize_entity("CO₂") == "carbon dioxide"
    assert normalize_entity("二氧化碳") == "carbon dioxide"
    assert normalize_entity("脑机接口") == "brain-computer interface"
    assert normalize_entity("BCI") == "brain-computer interface"


def test_related_concepts_do_not_merge():
    assert normalize_entity("cellular senescence") == "cellular senescence"
    assert normalize_entity("dark energy") == "dark energy"
    assert normalize_entity("quantum machine learning") == "quantum machine learning"
    assert "quantum machine learning" in related_terms("quantum artificial intelligence")


def test_equivalent_aliases_exclude_related_terms():
    aliases = equivalent_aliases("aging")
    assert "ageing" in aliases
    assert "cellular senescence" not in aliases
