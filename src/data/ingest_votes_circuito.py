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
