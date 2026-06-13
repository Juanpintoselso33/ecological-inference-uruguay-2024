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
