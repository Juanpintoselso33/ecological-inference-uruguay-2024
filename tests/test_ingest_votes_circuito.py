# tests/test_ingest_votes_circuito.py
import json
import pandas as pd
from src.data.ingest_votes_circuito import load_election


def _write_fixture(snap_dir):
    """Crea un fixture mínimo: 1 elección, 1 departamento, 2 circuitos."""
    doc = {
        "eleccionId": "balotaje-2024", "departamento": "artigas",
        "nivel": "circuito", "escrutinio": "definitivo", "tipo": "balotaje",
        "zonas": [
            {"geoId": "1", "local": "artigas-L001", "ganadorOpcionId": "nacional",
             "validos": 186,
             "porOpcion": [{"opcionId": "nacional", "votos": 152},
                           {"opcionId": "frente-amplio", "votos": 34}],
             "noPartidarios": {"enBlanco": 4, "anulados": 4, "observados": 27}},
            {"geoId": "2", "local": "artigas-L002", "ganadorOpcionId": "nacional",
             "validos": 207,
             "porOpcion": [{"opcionId": "nacional", "votos": 132},
                           {"opcionId": "frente-amplio", "votos": 75}],
             "noPartidarios": {"enBlanco": 5, "anulados": 1, "observados": 40}},
        ],
    }
    p = snap_dir / "balotaje-2024" / "artigas" / "votes-circuito.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")


def test_load_election_long_shape(tmp_path):
    _write_fixture(tmp_path)
    df = load_election("balotaje-2024", tmp_path, departamentos=["artigas"])
    # 2 circuitos x 2 opciones = 4 filas partidarias
    assert len(df) == 4
    assert set(df.columns) == {
        "departamento", "geoId", "local", "escrutinio",
        "opcionId", "votos", "validos", "enBlanco", "anulados", "observados",
    }


def test_load_election_nopartidarios_carried(tmp_path):
    _write_fixture(tmp_path)
    df = load_election("balotaje-2024", tmp_path, departamentos=["artigas"])
    c1 = df[df["geoId"] == "1"]
    assert (c1["enBlanco"] == 4).all()
    assert (c1["anulados"] == 4).all()
    assert (c1["observados"] == 27).all()
    assert (c1["validos"] == 186).all()
    assert set(c1["opcionId"]) == {"nacional", "frente-amplio"}


def test_load_election_geoid_is_string(tmp_path):
    _write_fixture(tmp_path)
    df = load_election("balotaje-2024", tmp_path, departamentos=["artigas"])
    assert df["geoId"].dtype == object
    assert "1" in set(df["geoId"])
