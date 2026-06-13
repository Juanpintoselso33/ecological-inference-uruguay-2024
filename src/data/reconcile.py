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
