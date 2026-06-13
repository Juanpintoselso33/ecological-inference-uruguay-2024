# tests/test_reconcile.py
import pandas as pd
from src.data.reconcile import parse_old_key, compare_party_votes


def test_parse_old_key_simple():
    depto, num = parse_old_key("Artigas_100_IAG")
    assert depto == "Artigas"
    assert num == "100"


def test_parse_old_key_multiword_depto():
    assert parse_old_key("Cerro Largo_523_ABC") == ("Cerro Largo", "523")
    assert parse_old_key("Treinta y Tres_12_XX") == ("Treinta y Tres", "12")
    assert parse_old_key("Río Negro_7_RN") == ("Río Negro", "7")


def test_compare_party_votes_perfect_match():
    old = pd.DataFrame({
        "circuito_id": ["Artigas_1_IAG"],
        "departamento": ["Artigas"],
        "fa_primera": [40], "pn_primera": [80],
    })
    new = pd.DataFrame({
        "circuito_id": ["Artigas_1"],
        "departamento": ["Artigas"], "geoId": ["1"],
        "fa_primera": [40], "pn_primera": [80],
    })
    report = compare_party_votes(old, new, ["fa_primera", "pn_primera"])
    assert report["n_matched"] == 1
    assert report["n_only_old"] == 0
    assert report["n_only_new"] == 0
    assert report["max_abs_diff"] == 0
