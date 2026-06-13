# tests/test_build_ei_dataset.py
import json
import pandas as pd
from src.data.build_ei_dataset import build_ei_dataset


def _write(snap_dir, eleccion, tipo, zonas, departamento="artigas",
           escrutinio="definitivo"):
    doc = {"eleccionId": eleccion, "departamento": departamento, "nivel": "circuito",
           "escrutinio": escrutinio, "tipo": tipo, "zonas": zonas}
    p = snap_dir / eleccion / departamento / "votes-circuito.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")


def _setup(snap_dir):
    # 1ª vuelta: 1 circuito con 6 lemas (2 menores -> otros)
    _write(snap_dir, "nacionales-2024", "nacional", [
        {"geoId": "1", "local": "artigas-L001", "validos": 200,
         "porOpcion": [
             {"opcionId": "nacional", "votos": 80},
             {"opcionId": "frente-amplio", "votos": 40},
             {"opcionId": "colorado", "votos": 30},
             {"opcionId": "cabildo-abierto", "votos": 20},
             {"opcionId": "independiente", "votos": 10},
             {"opcionId": "identidad-soberana", "votos": 12},
             {"opcionId": "asamblea-popular", "votos": 8},
         ],
         "noPartidarios": {"enBlanco": 0, "anulados": 0, "observados": 0}},
    ])
    # balotaje: mismo circuito
    _write(snap_dir, "balotaje-2024", "balotaje", [
        {"geoId": "1", "local": "artigas-L001", "validos": 186,
         "porOpcion": [{"opcionId": "nacional", "votos": 152},
                       {"opcionId": "frente-amplio", "votos": 34}],
         "noPartidarios": {"enBlanco": 4, "anulados": 4, "observados": 27}},
    ])


def test_build_columns_and_merge(tmp_path):
    _setup(tmp_path)
    df = build_ei_dataset(tmp_path, "nacionales-2024", "balotaje-2024",
                          departamentos=["artigas"], snapshot_hash="abc123")
    assert len(df) == 1
    row = df.iloc[0]
    # partidarios 1ª vuelta
    assert row["pn_primera"] == 80
    assert row["fa_primera"] == 40
    assert row["pc_primera"] == 30
    assert row["ca_primera"] == 20
    assert row["pi_primera"] == 10
    assert row["otros_primera"] == 20  # 12 + 8
    assert row["total_primera"] == 200
    # balotaje
    assert row["fa_ballotage"] == 34
    assert row["pn_ballotage"] == 152
    assert row["blancos_ballotage"] == 4   # enBlanco only
    assert row["anulados_ballotage"] == 4
    assert row["observados_ballotage"] == 27
    assert row["total_ballotage"] == 186 + 4 + 4 + 27
    # identidad / procedencia
    assert row["departamento"] == "Artigas"
    assert row["circuito_id"] == "Artigas_1"
    assert row["source_snapshot_hash"] == "abc123"


def test_build_shares(tmp_path):
    _setup(tmp_path)
    df = build_ei_dataset(tmp_path, "nacionales-2024", "balotaje-2024",
                          departamentos=["artigas"], snapshot_hash="abc123")
    row = df.iloc[0]
    assert abs(row["pn_share_primera"] - 80 / 200) < 1e-9
    assert abs(row["fa_share_ballotage"] - 34 / row["total_ballotage"]) < 1e-9


def _circuito(geoId, validos, por_opcion, npart=None):
    return {"geoId": geoId, "local": f"L{geoId}", "validos": validos,
            "porOpcion": por_opcion,
            "noPartidarios": npart or {"enBlanco": 0, "anulados": 0, "observados": 0}}


def test_build_multi_department_escrutinio(tmp_path):
    # dos departamentos, cada uno con 1 circuito en ambas elecciones,
    # con valores de escrutinio distintos para validar la alineación por clave.
    _write(tmp_path, "nacionales-2024", "nacional",
           [_circuito("1", 100, [{"opcionId": "nacional", "votos": 100}])],
           departamento="artigas", escrutinio="definitivo")
    _write(tmp_path, "nacionales-2024", "nacional",
           [_circuito("5", 100, [{"opcionId": "frente-amplio", "votos": 100}])],
           departamento="canelones", escrutinio="provisorio")
    _write(tmp_path, "balotaje-2024", "balotaje",
           [_circuito("1", 90, [{"opcionId": "nacional", "votos": 90}])],
           departamento="artigas", escrutinio="definitivo")
    _write(tmp_path, "balotaje-2024", "balotaje",
           [_circuito("5", 90, [{"opcionId": "frente-amplio", "votos": 90}])],
           departamento="canelones", escrutinio="provisorio")

    df = build_ei_dataset(tmp_path, "nacionales-2024", "balotaje-2024",
                          departamentos=["artigas", "canelones"],
                          snapshot_hash="abc123")
    assert len(df) == 2
    by_id = df.set_index("circuito_id")
    assert by_id.loc["Artigas_1", "escrutinio"] == "definitivo"
    assert by_id.loc["Canelones_5", "escrutinio"] == "provisorio"
    assert by_id.loc["Artigas_1", "pn_primera"] == 100
    assert by_id.loc["Canelones_5", "fa_primera"] == 100


def test_build_inner_join_drops_unmatched(tmp_path):
    # primera tiene geoId "1" y "2"; balotaje sólo "1" -> sobrevive sólo "1".
    _write(tmp_path, "nacionales-2024", "nacional", [
        _circuito("1", 100, [{"opcionId": "nacional", "votos": 100}]),
        _circuito("2", 100, [{"opcionId": "nacional", "votos": 100}]),
    ], departamento="artigas")
    _write(tmp_path, "balotaje-2024", "balotaje", [
        _circuito("1", 90, [{"opcionId": "nacional", "votos": 90}]),
    ], departamento="artigas")

    df = build_ei_dataset(tmp_path, "nacionales-2024", "balotaje-2024",
                          departamentos=["artigas"], snapshot_hash="abc123")
    assert len(df) == 1
    assert df.iloc[0]["geoId"] == "1"
    assert df.iloc[0]["circuito_id"] == "Artigas_1"


def test_build_ballotage_shares_sum_to_one(tmp_path):
    _setup(tmp_path)
    df = build_ei_dataset(tmp_path, "nacionales-2024", "balotaje-2024",
                          departamentos=["artigas"], snapshot_hash="abc123")
    row = df.iloc[0]
    total = (row["fa_share_ballotage"] + row["pn_share_ballotage"]
             + row["blancos_share_ballotage"] + row["anulados_share_ballotage"]
             + row["observados_share_ballotage"])
    assert abs(total - 1.0) < 1e-9
