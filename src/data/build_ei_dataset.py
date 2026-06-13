"""
Construye el dataset EI-ready v2 desde el snapshot externo:
pivotea opcionId->columnas de partido, mergea 1ª vuelta y balotaje por
(departamento, geoId), calcula shares y agrega columnas de procedencia.
"""
from pathlib import Path
import pandas as pd

from src.data.ingest_votes_circuito import load_election
from src.data.mappings import map_opcion, departamento_display
from src.utils import get_logger

logger = get_logger(__name__)

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
    })
    prim_out["total_primera"] = prim_out[[f"{p}_primera" for p in ORIGIN_PARTIES]].sum(axis=1)

    # adjuntar escrutinio por clave explícita (no por alineación posicional)
    esc = prim_long.groupby(["departamento", "geoId"])["escrutinio"].first().reset_index()
    prim_out = prim_out.merge(esc, on=["departamento", "geoId"], how="left")

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

    # el inner join descarta circuitos cuyos universos no coinciden entre vueltas
    n_drop_prim = len(prim_out) - len(df)
    n_drop_bal = len(bal_out) - len(df)
    if n_drop_prim or n_drop_bal:
        logger.warning("Inner merge dropped %d circuitos sin balotaje y %d sin primera",
                       n_drop_prim, n_drop_bal)

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
    df["observados_share_ballotage"] = df["observados_ballotage"] / df["total_ballotage"]

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
