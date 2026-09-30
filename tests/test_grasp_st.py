import numpy as np

from surgscene.grasp_st import (
    ACTIONS,
    INSTRUMENTS,
    N_ACT,
    answer,
    decide,
    mean_ap,
    multilabel_macro_f1,
    parse,
    single_label_macro_f1,
)


def test_answer_parse_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ins = int(rng.integers(len(INSTRUMENTS)))
        act = rng.random(N_ACT) < 0.2
        act[rng.integers(N_ACT)] = True
        i, a = parse(answer(ins, act))
        assert i == ins and (a == act).all()


def test_parse_overlapping_action_names():
    _, a = parse("Prograsp Forceps; Open Something")
    assert a[ACTIONS.index("Open Something")] and not a[ACTIONS.index("Open")]
    _, a = parse("Prograsp Forceps; Open, Open Something.")
    assert a[ACTIONS.index("Open Something")] and a[ACTIONS.index("Open")]
    assert parse("I cannot tell")[0] == -1 and not parse("I cannot tell")[1].any()


def test_decide_forces_one_action():
    p = np.array([[0.1, 0.2, 0.05], [0.9, 0.6, 0.1]])
    d = decide(p, 0.5)
    assert d.tolist() == [[False, True, False], [True, True, False]]


def test_metrics():
    Y = np.array([[1, 0], [0, 1], [1, 1]], bool)
    assert multilabel_macro_f1(Y, Y) == 1.0
    assert mean_ap(Y, Y.astype(float)) == 1.0
    # an unparsed answer (-1) is wrong: class 0 F1 = 2/3, class 1 F1 = 1
    assert np.isclose(single_label_macro_f1(np.array([0, 0, 1]), np.array([0, -1, 1]), 2), (2 / 3 + 1) / 2)
