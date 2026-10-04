from flap.eval.metrics import normalize_answer, exact_match, f1_score, evidence_hit, aggregate


def test_normalize():
    assert normalize_answer("The Blue Album!") == "blue album"
    assert normalize_answer(None) == ""


def test_em_and_f1():
    assert exact_match("the blue album", ["The Blue Album"]) == 1.0
    assert exact_match("Weezer", ["The Blue Album"]) == 0.0
    assert exact_match(None, ["x"]) == 0.0
    assert f1_score("blue album weezer", ["The Blue Album"]) > 0.7
    assert f1_score("nothing", ["The Blue Album"]) == 0.0


def test_evidence_hit():
    docs = ["Weezer, also known as the Blue Album, is ...", "unrelated"]
    assert evidence_hit(docs, ["The Blue Album"]) == 1.0
    assert evidence_hit(docs, ["Pinkerton"]) == 0.0
    assert evidence_hit(docs, ["The Blue Album"], k=0) == 0.0


def test_aggregate():
    s = aggregate([{"em": 1, "f1": 1, "num_search_calls": 2}, {"em": 0, "f1": 0.5, "num_search_calls": 1}])
    assert s["em"] == 50.0 and s["f1"] == 75.0 and s["avg_search_calls"] == 1.5
    assert aggregate([])["n"] == 0
