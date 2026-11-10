import random

from src.preprocess import english_stopwords
from src.spelling import (ERROR_TYPES, SpellCorrector, apply_corrections, build_dictionary, edit_distance, jaccard,
                          kgrams, make_typo, pick_target_token, replace_word)

TEXTS = ["probiotics prevent the common cold", "statin drugs and cholesterol", "coconut milk and saturated fat",
         "chicken nuggets contain antibiotics", "organic foods and pesticides residues"]
VOCAB = build_dictionary(TEXTS)


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("probiotics", "probitoics") == 1           # troca de ordem custa 1 (Damerau)
    assert edit_distance("cold", "colds") == 1 and edit_distance("", "abc") == 3 and edit_distance("abc", "abc") == 0


def test_kgrams_and_jaccard():
    assert kgrams("bord", 2) == {"$b", "bo", "or", "rd", "d$"}
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_corrector_fixes_each_error_type():
    cor = SpellCorrector(VOCAB, k=2, jaccard_threshold=0.3, max_edit=2)
    for typo, right in [("probiotcs", "probiotics"), ("probiiotics", "probiotics"), ("probxotics", "probiotics"),
                        ("probitoics", "probiotics"), ("cholestrol", "cholesterol")]:
        assert typo not in VOCAB and cor.correct_tokens([typo])[0] == [right]
    assert cor.correct_tokens(["cold"])[0] == ["cold"]                # termo conhecido não é alterado


def test_typo_generator_invariants():
    rng = random.Random(1)
    for kind in ERROR_TYPES:
        for term in ["probiotics", "cholesterol", "antibiotics"]:
            typo = make_typo(term, kind, rng, VOCAB)
            assert typo is not None and typo != term and typo not in VOCAB and typo[0] == term[0]
            d = edit_distance(term, typo)
            assert d == 1


def test_target_selection_and_text_replacement():
    stop = english_stopwords()
    toks = ["the", "common", "probiotics", "cold"]
    idf = {"common": 1.0, "probiotics": 5.0, "cold": 2.0}
    assert pick_target_token(toks, lambda t: idf.get(t, 0), VOCAB, stop) == "probiotics"
    assert replace_word("Do Probiotics help?", "probiotics", "probitoics") == "Do Probitoics help?"
    assert replace_word("Do Probiotics help?", "vitamin", "x") is None
    assert apply_corrections("do probitoics help", {"probitoics": "probiotics"}) == "do probiotics help"
