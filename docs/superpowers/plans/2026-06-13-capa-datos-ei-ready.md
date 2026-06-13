# Capa de datos EI-ready — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir `circuitos_merged_v2.parquet` a partir de los `votes-circuito.json` del proyecto *Uruguay Electoral Map*, en paralelo al parquet actual, con un gate de reconciliación que explique toda diferencia.

**Architecture:** Snapshot fijado por hash de los datos del repo vecino → ingest a DataFrame largo → pivot + merge 1ª vuelta/balotaje por `(departamento, geoId)` → parquet v2 con columnas de procedencia → reconciliación viejo-vs-nuevo y vs totales oficiales.

**Tech Stack:** Python 3.12, pandas, pyarrow, pytest. Entorno conda `ei2024`. Todo se corre con `conda run -n ei2024 ...`.

**Spec:** `docs/superpowers/specs/2026-06-13-capa-datos-ei-ready-design.md`

**Rama:** `feature/ei-data-layer-v2` (ya creada).

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `config.yaml` (modificar) | Sección `data.external`: ruta al repo vecino, elecciones objetivo, dir de snapshot |
| `src/data/mappings.py` (crear) | `PARTY_MAP` (opcionId→código EI) y `DEPARTAMENTO_DISPLAY` (slug→nombre con acento) |
| `scripts/snapshot_external_data.py` (crear) | Copia `votes-circuito.json` + escribe `snapshot_manifest.json` (sha256) + `reference_dept_totals.csv` |
| `src/data/ingest_votes_circuito.py` (crear) | `load_election()` → DataFrame largo por circuito con partidarios + no-partidarios |
| `src/data/build_ei_dataset.py` (crear) | Pivot + merge → `circuitos_merged_v2.parquet` |
| `src/data/reconcile.py` (crear) | `outputs/reconciliation_report.{md,csv}` |
| `tests/test_mappings.py` (crear) | Tests de mapeos |
| `tests/test_ingest_votes_circuito.py` (crear) | Tests de ingest sobre fixture |
| `tests/test_build_ei_dataset.py` (crear) | Tests de build sobre fixture |
| `tests/test_reconcile.py` (crear) | Tests de reconciliación |

Elecciones objetivo de este sub-proyecto: `nacionales-2024` (1ª vuelta) y `balotaje-2024`.

---

## Task 1: Config + módulo de mapeos

**Files:**
- Modify: `config.yaml` (después de la sección `data:`, antes de `parties:`)
- Create: `src/data/mappings.py`
- Test: `tests/test_mappings.py`

- [ ] **Step 1: Agregar sección `external` a `config.yaml`**

Dentro del bloque `data:`, agregar (manteniendo la indentación de 2 espacios del bloque):

```yaml
  # Fuente externa robusta (proyecto Uruguay Electoral Map)
  external:
    map_repo_data_dir: "E:/Proyectos VS CODE/Uruguay Electoral Map/uruguay-electoral-map/public/data"
    map_repo_dumps_dir: "E:/Proyectos VS CODE/Uruguay Electoral Map/uruguay-electoral-map/public/api/v1/dumps"
    snapshot_dir: "data/raw/external"
    elections:
      primera: "nacionales-2024"
      ballotage: "balotaje-2024"
```

- [ ] **Step 2: Escribir el test que falla**

```python
# tests/test_mappings.py
import pytest
from src.data.mappings import PARTY_MAP, DEPARTAMENTO_DISPLAY, map_opcion, departamento_display


def test_party_map_known_lemas():
    assert map_opcion("cabildo-abierto") == "ca"
    assert map_opcion("frente-amplio") == "fa"
    assert map_opcion("colorado") == "pc"
    assert map_opcion("nacional") == "pn"
    assert map_opcion("independiente") == "pi"


def test_party_map_minor_lemas_become_otros():
    for lema in ["identidad-soberana", "constitucional-ambientalista",
                 "ecologista-radical-intransigente", "asamblea-popular",
                 "por-los-cambios-necesarios-pcn", "avanzar-republicano"]:
        assert map_opcion(lema) == "otros"


def test_party_map_unknown_raises():
    with pytest.raises(KeyError):
        map_opcion("partido-inventado")


def test_departamento_display_has_accents():
    assert departamento_display("paysandu") == "Paysandú"
    assert departamento_display("rio_negro") == "Río Negro"
    assert departamento_display("san_jose") == "San José"
    assert departamento_display("cerro_largo") == "Cerro Largo"
    assert departamento_display("treinta_y_tres") == "Treinta y Tres"


def test_departamento_display_count():
    assert len(DEPARTAMENTO_DISPLAY) == 19
```

- [ ] **Step 3: Correr el test y verificar que falla**

Run: `conda run -n ei2024 pytest tests/test_mappings.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.mappings'`

- [ ] **Step 4: Implementar `src/data/mappings.py`**

```python
"""
Mapeos canónicos entre los identificadores del proyecto Uruguay Electoral Map
y los códigos usados por el pipeline de inferencia ecológica.
"""

# opcionId (lema) -> código EI. Los lemas no listados se agregan en 'otros'.
_EXPLICIT_PARTY_MAP = {
    "cabildo-abierto": "ca",
    "frente-amplio": "fa",
    "colorado": "pc",
    "nacional": "pn",
    "independiente": "pi",
}

# Lemas menores conocidos de 2024 (validan que 'otros' es deliberado, no un typo).
_KNOWN_OTROS = {
    "identidad-soberana",
    "constitucional-ambientalista",
    "ecologista-radical-intransigente",
    "asamblea-popular",
    "por-los-cambios-necesarios-pcn",
    "avanzar-republicano",
}

PARTY_MAP = dict(_EXPLICIT_PARTY_MAP)
for _lema in _KNOWN_OTROS:
    PARTY_MAP[_lema] = "otros"


def map_opcion(opcion_id: str) -> str:
    """Devuelve el código EI para un opcionId. Levanta KeyError si es desconocido."""
    try:
        return PARTY_MAP[opcion_id]
    except KeyError as exc:
        raise KeyError(
            f"opcionId desconocido: {opcion_id!r}. Agregalo a mappings.PARTY_MAP "
            f"como código de partido o como 'otros'."
        ) from exc


# slug del repo vecino -> nombre con acento usado en el parquet histórico.
DEPARTAMENTO_DISPLAY = {
    "artigas": "Artigas",
    "canelones": "Canelones",
    "cerro_largo": "Cerro Largo",
    "colonia": "Colonia",
    "durazno": "Durazno",
    "flores": "Flores",
    "florida": "Florida",
    "lavalleja": "Lavalleja",
    "maldonado": "Maldonado",
    "montevideo": "Montevideo",
    "paysandu": "Paysandú",
    "rio_negro": "Río Negro",
    "rivera": "Rivera",
    "rocha": "Rocha",
    "salto": "Salto",
    "san_jose": "San José",
    "soriano": "Soriano",
    "tacuarembo": "Tacuarembó",
    "treinta_y_tres": "Treinta y Tres",
}


def departamento_display(slug: str) -> str:
    """slug lowercase del repo vecino -> nombre con acento. KeyError si desconocido."""
    try:
        return DEPARTAMENTO_DISPLAY[slug]
    except KeyError as exc:
        raise KeyError(f"departamento desconocido: {slug!r}") from exc
```

- [ ] **Step 5: Correr el test y verificar que pasa**

Run: `conda run -n ei2024 pytest tests/test_mappings.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add config.yaml src/data/mappings.py tests/test_mappings.py
git commit -m "feat: party + departamento mappings for external data layer"
```

---

## Task 2: Script de snapshot

**Files:**
- Create: `scripts/snapshot_external_data.py`

Este script no lleva test unitario (es I/O de copia); se valida ejecutándolo en Task 6.

- [ ] **Step 1: Implementar `scripts/snapshot_external_data.py`**

```python
"""
Copia un snapshot fijado por hash de los datos del proyecto Uruguay Electoral Map
necesarios para el pipeline EI, y deja una referencia auditable.

Salida bajo data/raw/external/:
  <eleccion>/<depto>/votes-circuito.json   (copias)
  reference_dept_totals.csv                (totales partidarios por depto, del dump)
  snapshot_manifest.json                   (sha256 + origen + fecha)

Uso: conda run -n ei2024 python scripts/snapshot_external_data.py
"""
import sys
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import get_config

DEPARTAMENTOS = [
    "artigas", "canelones", "cerro_largo", "colonia", "durazno", "flores",
    "florida", "lavalleja", "maldonado", "montevideo", "paysandu", "rio_negro",
    "rivera", "rocha", "salto", "san_jose", "soriano", "tacuarembo", "treinta_y_tres",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def snapshot_votes_circuito(src_data_dir: Path, snap_dir: Path, eleccion: str) -> list[dict]:
    entries = []
    for depto in DEPARTAMENTOS:
        src = src_data_dir / eleccion / depto / "votes-circuito.json"
        if not src.exists():
            raise FileNotFoundError(f"No existe: {src}")
        dst = snap_dir / eleccion / depto / "votes-circuito.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        entries.append({
            "eleccion": eleccion,
            "departamento": depto,
            "origen": str(src),
            "sha256": sha256_of(dst),
        })
    return entries


def build_reference_dept_totals(dumps_dir: Path, snap_dir: Path, elecciones: list[str]):
    """Extrae los registros nivel=departamento del dump masivo (autoritativo para
    totales partidarios) hacia un CSV de referencia para reconciliación."""
    import csv
    rows = []
    for eleccion in elecciones:
        dump = dumps_dir / f"{eleccion}.ndjson"
        if not dump.exists():
            raise FileNotFoundError(f"No existe dump: {dump}")
        with dump.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("nivel") == "departamento" and rec.get("departamento") == "_nacional":
                    rows.append({
                        "eleccion": eleccion,
                        "departamento": rec["geoId"],
                        "opcionId": rec["opcionId"],
                        "votos": rec["votos"],
                        "validos": rec["validos"],
                    })
    out = snap_dir / "reference_dept_totals.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eleccion", "departamento", "opcionId", "votos", "validos"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    config = get_config()
    ext = config["data"]["external"]
    src_data_dir = Path(ext["map_repo_data_dir"])
    dumps_dir = Path(ext["map_repo_dumps_dir"])
    snap_dir = Path(ext["snapshot_dir"])
    elecciones = [ext["elections"]["primera"], ext["elections"]["ballotage"]]

    snap_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo_data_dir": str(src_data_dir),
        "elecciones": elecciones,
        "files": [],
    }
    for eleccion in elecciones:
        manifest["files"].extend(snapshot_votes_circuito(src_data_dir, snap_dir, eleccion))

    n_ref = build_reference_dept_totals(dumps_dir, snap_dir, elecciones)

    # hash global del snapshot (sobre los sha256 ordenados de los archivos)
    digest = hashlib.sha256(
        "".join(sorted(e["sha256"] for e in manifest["files"])).encode()
    ).hexdigest()
    manifest["snapshot_sha256"] = digest

    (snap_dir / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Snapshot OK: {len(manifest['files'])} archivos, "
          f"{n_ref} filas de referencia. snapshot_sha256={digest[:12]}…")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/snapshot_external_data.py
git commit -m "feat: snapshot script for external electoral data (hash-pinned)"
```

---

## Task 3: Ingest de votes-circuito.json

**Files:**
- Create: `src/data/ingest_votes_circuito.py`
- Test: `tests/test_ingest_votes_circuito.py`

- [ ] **Step 1: Escribir el test que falla**

```python
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
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `conda run -n ei2024 pytest tests/test_ingest_votes_circuito.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.ingest_votes_circuito'`

- [ ] **Step 3: Implementar `src/data/ingest_votes_circuito.py`**

```python
"""
Lee los archivos votes-circuito.json (uno por departamento) de una elección del
snapshot externo y devuelve un DataFrame largo: una fila por (circuito, opcionId)
partidario, con los conteos no partidarios del circuito repetidos en cada fila.
"""
import json
from pathlib import Path
import pandas as pd

DEPARTAMENTOS_DEFAULT = [
    "artigas", "canelones", "cerro_largo", "colonia", "durazno", "flores",
    "florida", "lavalleja", "maldonado", "montevideo", "paysandu", "rio_negro",
    "rivera", "rocha", "salto", "san_jose", "soriano", "tacuarembo", "treinta_y_tres",
]


def load_election(eleccion: str, snapshot_dir, departamentos=None) -> pd.DataFrame:
    """
    Args:
        eleccion: p.ej. 'balotaje-2024'
        snapshot_dir: raíz del snapshot (Path o str)
        departamentos: lista de slugs; por defecto los 19.
    """
    snapshot_dir = Path(snapshot_dir)
    departamentos = departamentos or DEPARTAMENTOS_DEFAULT
    rows = []
    for depto in departamentos:
        path = snapshot_dir / eleccion / depto / "votes-circuito.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        escrutinio = doc.get("escrutinio")
        for zona in doc["zonas"]:
            npart = zona.get("noPartidarios", {}) or {}
            base = {
                "departamento": depto,
                "geoId": str(zona["geoId"]),
                "local": zona.get("local"),
                "escrutinio": escrutinio,
                "validos": zona.get("validos", 0),
                "enBlanco": npart.get("enBlanco", 0),
                "anulados": npart.get("anulados", 0),
                "observados": npart.get("observados", 0),
            }
            for op in zona["porOpcion"]:
                rows.append({**base,
                             "opcionId": op["opcionId"],
                             "votos": op["votos"]})
    cols = ["departamento", "geoId", "local", "escrutinio",
            "opcionId", "votos", "validos", "enBlanco", "anulados", "observados"]
    return pd.DataFrame(rows, columns=cols)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `conda run -n ei2024 pytest tests/test_ingest_votes_circuito.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data/ingest_votes_circuito.py tests/test_ingest_votes_circuito.py
git commit -m "feat: ingest votes-circuito.json into long DataFrame"
```

---

## Task 4: Construcción del dataset EI v2

**Files:**
- Create: `src/data/build_ei_dataset.py`
- Test: `tests/test_build_ei_dataset.py`

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_build_ei_dataset.py
import json
import pandas as pd
from src.data.build_ei_dataset import build_ei_dataset


def _write(snap_dir, eleccion, tipo, zonas):
    doc = {"eleccionId": eleccion, "departamento": "artigas", "nivel": "circuito",
           "escrutinio": "definitivo", "tipo": tipo, "zonas": zonas}
    p = snap_dir / eleccion / "artigas" / "votes-circuito.json"
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
    assert row["blancos_ballotage"] == 8   # enBlanco 4 + anulados 4
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
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `conda run -n ei2024 pytest tests/test_build_ei_dataset.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.build_ei_dataset'`

- [ ] **Step 3: Implementar `src/data/build_ei_dataset.py`**

```python
"""
Construye el dataset EI-ready v2 desde el snapshot externo:
pivotea opcionId->columnas de partido, mergea 1ª vuelta y balotaje por
(departamento, geoId), calcula shares y agrega columnas de procedencia.
"""
from pathlib import Path
import pandas as pd

from src.data.ingest_votes_circuito import load_election
from src.data.mappings import map_opcion, departamento_display

ORIGIN_PARTIES = ["ca", "fa", "pc", "pn", "pi", "otros"]


def _pivot_partido(df_long: pd.DataFrame) -> pd.DataFrame:
    """De largo (una fila por opcionId) a ancho por (departamento, geoId),
    sumando opciones que mapean al mismo código EI."""
    df = df_long.copy()
    df["party"] = df["opcionId"].map(map_opcion)
    wide = (df.groupby(["departamento", "geoId", "party"])["votos"]
              .sum().unstack("party", fill_value=0).reset_index())
    # asegurar todas las columnas de partido aunque falte alguna en los datos
    for p in ORIGIN_PARTIES:
        if p not in wide.columns:
            wide[p] = 0
    # conteos no partidarios + validos, uno por circuito
    npart = (df_long.groupby(["departamento", "geoId"])
             [["validos", "enBlanco", "anulados", "observados"]].first().reset_index())
    return wide.merge(npart, on=["departamento", "geoId"], how="left")


def build_ei_dataset(snapshot_dir, primera_eleccion, ballotage_eleccion,
                     departamentos=None, snapshot_hash="unknown") -> pd.DataFrame:
    snapshot_dir = Path(snapshot_dir)

    prim_long = load_election(primera_eleccion, snapshot_dir, departamentos)
    bal_long = load_election(ballotage_eleccion, snapshot_dir, departamentos)

    prim = _pivot_partido(prim_long)
    bal = _pivot_partido(bal_long)

    # --- 1ª vuelta ---
    prim_out = pd.DataFrame({
        "departamento": prim["departamento"],
        "geoId": prim["geoId"],
        "ca_primera": prim["ca"], "fa_primera": prim["fa"], "pc_primera": prim["pc"],
        "pn_primera": prim["pn"], "pi_primera": prim["pi"], "otros_primera": prim["otros"],
        "escrutinio": prim_long.groupby(["departamento", "geoId"])["escrutinio"]
                      .first().reset_index(drop=True),
    })
    prim_out["total_primera"] = prim_out[[f"{p}_primera" for p in ORIGIN_PARTIES]].sum(axis=1)

    # --- balotaje ---
    bal_out = pd.DataFrame({
        "departamento": bal["departamento"],
        "geoId": bal["geoId"],
        "fa_ballotage": bal["fa"], "pn_ballotage": bal["pn"],
        "blancos_ballotage": bal["enBlanco"] + bal["anulados"],
        "observados_ballotage": bal["observados"],
    })
    bal_out["total_ballotage"] = (bal["fa"] + bal["pn"] + bal["enBlanco"]
                                  + bal["anulados"] + bal["observados"])

    # --- merge por clave de circuito ---
    df = prim_out.merge(bal_out, on=["departamento", "geoId"], how="inner")

    # identidad + procedencia
    df["departamento"] = df["departamento"].map(departamento_display)
    df["circuito_id"] = df["departamento"] + "_" + df["geoId"].astype(str)
    df["source_snapshot_hash"] = snapshot_hash

    # shares
    for p in ORIGIN_PARTIES:
        df[f"{p}_share_primera"] = df[f"{p}_primera"] / df["total_primera"]
    for d in ["fa", "pn"]:
        df[f"{d}_share_ballotage"] = df[f"{d}_ballotage"] / df["total_ballotage"]
    df["blancos_share_ballotage"] = df["blancos_ballotage"] / df["total_ballotage"]

    return df


def main():
    import sys, json
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.config import get_config
    config = get_config()
    ext = config["data"]["external"]
    snap_dir = Path(ext["snapshot_dir"])
    manifest = json.loads((snap_dir / "snapshot_manifest.json").read_text(encoding="utf-8"))
    df = build_ei_dataset(snap_dir, ext["elections"]["primera"],
                          ext["elections"]["ballotage"],
                          snapshot_hash=manifest["snapshot_sha256"])
    out = Path(config.get_data_dirs()["processed"]) / "circuitos_merged_v2.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"v2 dataset: {len(df)} circuitos -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `conda run -n ei2024 pytest tests/test_build_ei_dataset.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data/build_ei_dataset.py tests/test_build_ei_dataset.py
git commit -m "feat: build circuitos_merged_v2 from external snapshot"
```

---

## Task 5: Reconciliación

**Files:**
- Create: `src/data/reconcile.py`
- Test: `tests/test_reconcile.py`

- [ ] **Step 1: Escribir el test que falla**

```python
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
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `conda run -n ei2024 pytest tests/test_reconcile.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.reconcile'`

- [ ] **Step 3: Implementar `src/data/reconcile.py`**

```python
"""
Reconcilia circuitos_merged_v2 (nuevo) contra circuitos_merged (viejo) y contra
los totales departamentales oficiales del snapshot. Emite un reporte md + csv.
"""
from pathlib import Path
import pandas as pd

PARTY_COLS = ["ca_primera", "fa_primera", "pc_primera", "pn_primera",
              "pi_primera", "otros_primera", "fa_ballotage", "pn_ballotage"]


def parse_old_key(circuito_id: str) -> tuple[str, str]:
    """'Artigas_100_IAG' -> ('Artigas', '100'). El departamento puede tener espacios;
    el número es el penúltimo token separado por '_'."""
    head, num, _serie = circuito_id.rsplit("_", 2)
    return head, num


def compare_party_votes(old: pd.DataFrame, new: pd.DataFrame, cols: list[str]) -> dict:
    """Compara votos por partido por circuito entre viejo y nuevo."""
    o = old.copy()
    o[["_depto", "_num"]] = o["circuito_id"].apply(
        lambda cid: pd.Series(parse_old_key(cid)))
    o["_key"] = o["_depto"] + "|" + o["_num"]
    n = new.copy()
    n["_key"] = n["departamento"] + "|" + n["geoId"].astype(str)

    keys_old, keys_new = set(o["_key"]), set(n["_key"])
    merged = o.merge(n, on="_key", suffixes=("_old", "_new"))

    max_abs = 0
    per_col = {}
    for c in cols:
        if f"{c}_old" in merged and f"{c}_new" in merged:
            diff = (merged[f"{c}_old"] - merged[f"{c}_new"]).abs()
            per_col[c] = {"max": int(diff.max()), "n_nonzero": int((diff > 0).sum())}
            max_abs = max(max_abs, int(diff.max()))

    return {
        "n_matched": len(merged),
        "n_only_old": len(keys_old - keys_new),
        "n_only_new": len(keys_new - keys_old),
        "keys_only_old": sorted(keys_old - keys_new)[:50],
        "keys_only_new": sorted(keys_new - keys_old)[:50],
        "max_abs_diff": max_abs,
        "per_col": per_col,
    }


def compare_dept_totals(new: pd.DataFrame, reference_csv: Path,
                        primera_eleccion: str) -> pd.DataFrame:
    """Suma votos partidarios de v2 por departamento y los compara con el CSV de
    referencia (nivel departamento del dump oficial) para la 1ª vuelta."""
    from src.data.mappings import map_opcion, departamento_display
    ref = pd.read_csv(reference_csv)
    ref = ref[ref["eleccion"] == primera_eleccion].copy()
    ref["party"] = ref["opcionId"].map(map_opcion)
    ref["depto_disp"] = ref["departamento"].map(
        lambda d: departamento_display(d.lower().replace(" ", "_")
                                       .replace("í", "i").replace("ú", "u")
                                       .replace("é", "e").replace("ó", "o"))
        if d.lower() not in ("_nacional",) else d)
    # referencia ya viene con geoId=nombre con acento capitalizado del dump _nacional
    ref_tot = ref.groupby(["departamento", "party"])["votos"].sum().unstack("party", fill_value=0)

    new_long = new.melt(id_vars=["departamento"],
                        value_vars=["ca_primera", "fa_primera", "pc_primera",
                                    "pn_primera", "pi_primera", "otros_primera"],
                        var_name="col", value_name="votos")
    new_long["party"] = new_long["col"].str.replace("_primera", "", regex=False)
    new_tot = new_long.groupby(["departamento", "party"])["votos"].sum().unstack("party", fill_value=0)

    comparison = new_tot.subtract(ref_tot, fill_value=0)
    return comparison


def write_report(party_report: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reporte de reconciliación — circuitos_merged_v2 vs viejo",
        "",
        "## Cobertura de circuitos",
        f"- Circuitos apareados: **{party_report['n_matched']}**",
        f"- Solo en viejo: **{party_report['n_only_old']}**",
        f"- Solo en nuevo: **{party_report['n_only_new']}**",
        "",
        "## Votos por partido por circuito",
        f"- Diferencia absoluta máxima: **{party_report['max_abs_diff']}**",
        "",
        "| Columna | max |Δ| | circuitos con Δ≠0 |",
        "|---|---|---|",
    ]
    for c, st in party_report["per_col"].items():
        lines.append(f"| {c} | {st['max']} | {st['n_nonzero']} |")
    if party_report["keys_only_old"]:
        lines += ["", "### Circuitos solo en viejo (hasta 50)", ""]
        lines += [f"- {k}" for k in party_report["keys_only_old"]]
    if party_report["keys_only_new"]:
        lines += ["", "### Circuitos solo en nuevo (hasta 50)", ""]
        lines += [f"- {k}" for k in party_report["keys_only_new"]]

    (out_dir / "reconciliation_report.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(party_report["per_col"]).T.to_csv(out_dir / "reconciliation_report.csv")


def main():
    import sys, json
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.config import get_config
    config = get_config()
    processed = Path(config.get_data_dirs()["processed"])
    old = pd.read_parquet(processed / "circuitos_merged.parquet")
    new = pd.read_parquet(processed / "circuitos_merged_v2.parquet")
    report = compare_party_votes(old, new, PARTY_COLS)
    out_dir = Path("outputs")
    write_report(report, out_dir)
    print(f"Reconciliación: {report['n_matched']} apareados, "
          f"max|Δ|={report['max_abs_diff']}, "
          f"solo_viejo={report['n_only_old']}, solo_nuevo={report['n_only_new']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `conda run -n ei2024 pytest tests/test_reconcile.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data/reconcile.py tests/test_reconcile.py
git commit -m "feat: reconciliation of v2 dataset vs old parquet"
```

---

## Task 6: Corrida end-to-end + impacto en EI + decisión

**Files:**
- Create: `outputs/reconciliation_report.md` (generado)
- Create: `data/processed/circuitos_merged_v2.parquet` (generado)
- Modify: `docs/superpowers/specs/2026-06-13-capa-datos-ei-ready-design.md` (agregar sección "Resultado de reconciliación")

- [ ] **Step 1: Correr la suite completa de tests nuevos**

Run: `conda run -n ei2024 pytest tests/test_mappings.py tests/test_ingest_votes_circuito.py tests/test_build_ei_dataset.py tests/test_reconcile.py -v`
Expected: PASS (todos)

- [ ] **Step 2: Generar el snapshot**

Run: `conda run -n ei2024 python scripts/snapshot_external_data.py`
Expected: `Snapshot OK: 38 archivos, ... snapshot_sha256=...`
(38 = 19 deptos × 2 elecciones)

- [ ] **Step 3: Construir el dataset v2**

Run: `conda run -n ei2024 python -m src.data.build_ei_dataset`
Expected: `v2 dataset: ~7221 circuitos -> data/processed/circuitos_merged_v2.parquet`

- [ ] **Step 4: Correr la reconciliación**

Run: `conda run -n ei2024 python -m src.data.reconcile`
Expected: imprime conteo de apareados y `max|Δ|`. Leer `outputs/reconciliation_report.md`.
Criterio: ≥99% de circuitos apareados con `max|Δ|` chico (idealmente 0). Si no, investigar la semántica de `observados` y los circuitos del set-diff antes de seguir.

- [ ] **Step 5: Re-correr el EI nacional sobre v2 (test rápido) y comparar**

Run:
```bash
conda run -n ei2024 python scripts/national_analysis_with_pi.py \
  --data data/processed/circuitos_merged_v2.parquet \
  --samples 500 --chains 2 \
  --output outputs/results/national_transfers_with_pi_v2.pkl
```
Expected: corre sin error y produce el pkl v2. Comparar las tasas de transferencia
(media + IC) contra `outputs/results/national_transfers_with_pi.pkl` (viejo).
Diferencias esperadas: pequeñas; cualquier diferencia material se documenta.

- [ ] **Step 6: Documentar el resultado en el spec**

Agregar al final de `docs/superpowers/specs/2026-06-13-capa-datos-ei-ready-design.md` una sección
`## 10. Resultado de reconciliación` con: nº de circuitos apareados / set-diff, `max|Δ|`
por partido, totales departamentales OK/diff, y la tabla EI viejo-vs-v2. Cerrar con la
**decisión registrada**: migrar el análisis a v2, o no, con justificación.

- [ ] **Step 7: Commit**

```bash
git add data/processed/circuitos_merged_v2.parquet outputs/reconciliation_report.md \
        outputs/reconciliation_report.csv data/raw/external/snapshot_manifest.json \
        docs/superpowers/specs/2026-06-13-capa-datos-ei-ready-design.md
git commit -m "feat: build v2 dataset, reconcile vs old, document EI impact"
```

---

## Self-review (autor)

- **Cobertura del spec:** §2 estrategia → Tasks 2/4; §3 fuente → Task 3; §4 flujo → Tasks 2-5;
  §5 módulos → Tasks 1-5 (mapeos, snapshot, ingest, build, reconcile); §6 gate de aceptación
  → Task 6 (steps 4-6); §7 pruebas → tests en Tasks 1,3,4,5; §9 riesgos → Task 6 step 4 (semántica
  de observados / set-diff). Cubierto.
- **Consistencia de tipos:** `geoId` es `str` en ingest, build y reconcile (clave de merge);
  `departamento` pasa de slug (ingest/build interno) a display (salida de build, entrada de
  reconcile). `load_election(eleccion, snapshot_dir, departamentos)`, `build_ei_dataset(snapshot_dir,
  primera, ballotage, departamentos, snapshot_hash)`, `compare_party_votes(old, new, cols)`,
  `parse_old_key(circuito_id)` — firmas usadas igual en tests e implementación.
- **Sin placeholders:** todos los steps con código traen el código completo.
- **Riesgo conocido (Task 5, `compare_dept_totals`):** la normalización inversa de acentos del
  CSV de referencia es frágil; por eso el gate primario de Task 6 usa la comparación
  viejo-vs-nuevo (`compare_party_votes`, robusta) y `compare_dept_totals` queda como chequeo
  secundario. Si da ruido, simplificar comparando v2 contra sumas departamentales del parquet viejo.
