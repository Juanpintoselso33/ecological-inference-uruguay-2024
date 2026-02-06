"""
Generate LaTeX tables for temporal analysis using available CSV data.
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path("E:/Proyectos VS CODE/Eco inference 2024")
TABLES_DIR = BASE_DIR / "outputs" / "tables"
LATEX_DIR = TABLES_DIR / "latex"

# Ensure output directory exists
LATEX_DIR.mkdir(exist_ok=True, parents=True)


def format_votes(value):
    """Format vote counts with thousands separator."""
    # Remove existing separators if present
    if isinstance(value, str):
        value = value.replace(",", "").replace(".", "")
    return f"{int(value):,}".replace(",", ".")


def generate_pi_national_results_2024():
    """Generate table with PI results from department data."""
    print("Generating PI national results 2024 table...")

    # Load data
    df = pd.read_csv(TABLES_DIR / "transfers_by_department_with_pi.csv")

    # Calculate weighted national averages for each party
    parties_data = []

    # CA
    ca_total = df['ca_votes'].sum()
    ca_to_fa_avg = (df['ca_to_fa'] * df['ca_votes']).sum() / ca_total * 100
    ca_to_pn_avg = (df['ca_to_pn'] * df['ca_votes']).sum() / ca_total * 100
    parties_data.append(('CA', ca_to_fa_avg, ca_to_pn_avg, 100 - ca_to_fa_avg - ca_to_pn_avg))

    # PC
    pc_total = df['pc_votes'].sum()
    pc_to_fa_avg = (df['pc_to_fa'] * df['pc_votes']).sum() / pc_total * 100
    pc_to_pn_avg = (df['pc_to_pn'] * df['pc_votes']).sum() / pc_total * 100
    parties_data.append(('PC', pc_to_fa_avg, pc_to_pn_avg, 100 - pc_to_fa_avg - pc_to_pn_avg))

    # PN
    pn_total = df['pn_votes'].sum()
    pn_to_fa_avg = (df['pn_to_fa'] * df['pn_votes']).sum() / pn_total * 100
    pn_to_pn_avg = (df['pn_to_pn'] * df['pn_votes']).sum() / pn_total * 100
    parties_data.append(('PN', pn_to_fa_avg, pn_to_pn_avg, 100 - pn_to_fa_avg - pn_to_pn_avg))

    # PI
    pi_total = df['pi_votes'].sum()
    pi_to_fa_avg = (df['pi_to_fa'] * df['pi_votes']).sum() / pi_total * 100
    pi_to_pn_avg = (df['pi_to_pn'] * df['pi_votes']).sum() / pi_total * 100
    parties_data.append(('PI', pi_to_fa_avg, pi_to_pn_avg, 100 - pi_to_fa_avg - pi_to_pn_avg))

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Resultados nacionales de transferencias de votos 2024 con Partido Independiente como categoría separada. Valores son medias ponderadas por cantidad de votos.}
\label{tab:pi_national_2024}
\begin{tabular}{lrrr}
\toprule
\textbf{Partido} & \textbf{Defección a FA (\%)} & \textbf{Retención PN (\%)} & \textbf{Blancos (\%)} \\
\midrule
"""

    for party, to_fa, to_pn, blancos in parties_data:
        latex += f"{party} & {to_fa:.1f}\\% & {to_pn:.1f}\\% & {blancos:.1f}\\% \\\\\n"

    latex += r"""\bottomrule
\multicolumn{4}{l}{\footnotesize PI muestra la mayor lealtad coalicional (2.8\% defección), seguido de PN y PC.} \\
\multicolumn{4}{l}{\footnotesize CA exhibe división casi equitativa entre FA y PN.} \\
\end{tabular}
\end{table}
"""

    output_path = LATEX_DIR / "pi_national_results_2024.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"  OK - Saved to {output_path.name}")


def generate_national_transition_matrix_2019():
    """Generate national transition matrix for 2019."""
    print("Generating national transition matrix 2019...")

    # Load data
    df = pd.read_csv(TABLES_DIR / "transfers_by_department_2019.csv")

    # Calculate weighted national averages
    parties_data = []

    for party_code, party_name in [('ca', 'CA'), ('pc', 'PC'), ('pn', 'PN'), ('pi', 'PI')]:
        votes_col = f'{party_code}_votes'
        to_fa_col = f'{party_code}_to_fa'
        to_pn_col = f'{party_code}_to_pn'
        to_blancos_col = f'{party_code}_to_blancos'

        total = df[votes_col].sum()
        to_fa = (df[to_fa_col] * df[votes_col]).sum() / total * 100
        to_pn = (df[to_pn_col] * df[votes_col]).sum() / total * 100
        to_blancos = (df[to_blancos_col] * df[votes_col]).sum() / total * 100

        parties_data.append((party_name, to_fa, to_pn, to_blancos))

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Matriz de transición nacional 2019: Transferencias de votos entre primera vuelta y balotaje. Valores son medias ponderadas por cantidad de votos.}
\label{tab:national_transitions_2019}
\begin{tabular}{lrrr}
\toprule
\textbf{Origen} & \textbf{FA (2da) (\%)} & \textbf{PN (2da) (\%)} & \textbf{Blancos (\%)} \\
\midrule
"""

    for party, to_fa, to_pn, blancos in parties_data:
        latex += f"{party} & {to_fa:.1f}\\% & {to_pn:.1f}\\% & {blancos:.1f}\\% \\\\\n"

    latex += r"""\bottomrule
\multicolumn{4}{l}{\footnotesize En 2019, todos los partidos coalicionales tuvieron mayor defección que en 2024.} \\
\multicolumn{4}{l}{\footnotesize CA: 69.5\% defección (vs 27.6\% en 2024), PC: 33.8\% (vs 7.2\%), PN: 36.0\% (vs 3.6\%).} \\
\end{tabular}
\end{table}
"""

    output_path = LATEX_DIR / "national_transition_matrix_2019.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"  OK - Saved to {output_path.name}")


def generate_comparison_table_2019_2024():
    """Generate comparison table between 2019 and 2024."""
    print("Generating comparison table 2019-2024...")

    # Load comparison data
    df = pd.read_csv(TABLES_DIR / "comparison_2019_2024.csv")

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Evolución de defecciones coalicionales 2019-2024. Mejora dramática en cohesión: todos los partidos redujeron defección entre 26.6 y 92.1 puntos porcentuales.}
\label{tab:comparison_2019_2024}
\small
\begin{tabular}{lrrrrr}
\toprule
\textbf{Partido} & \textbf{Def. 2019} & \textbf{Def. 2024} & \textbf{Cambio} & \textbf{Votos def.} & \textbf{Votos def.} \\
 & \textbf{(\%)} & \textbf{(\%)} & \textbf{(pp)} & \textbf{2019} & \textbf{2024} \\
\midrule
"""

    for _, row in df.iterrows():
        partido = row['Partido']
        def_2019 = row['Defeccion_2019_%']
        def_2024 = row['Defeccion_2024_%']
        cambio = row['Cambio_pp']
        votos_2019 = format_votes(row['Votos_defectos_2019'])
        votos_2024 = format_votes(row['Votos_defectos_2024'])

        latex += f"{partido} & {def_2019:.1f}\\% & {def_2024:.1f}\\% & {cambio:+.1f} & {votos_2019} & {votos_2024} \\\\\n"

    latex += r"""\bottomrule
\multicolumn{6}{l}{\footnotesize Defección = \% de votos transferidos de partido coalicional a FA en balotaje.} \\
\multicolumn{6}{l}{\footnotesize PI tuvo el cambio más dramático: -92.1 pp (de 94.8\% a 2.8\%).} \\
\end{tabular}
\end{table}
"""

    output_path = LATEX_DIR / "comparison_table_2019_2024.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"  OK - Saved to {output_path.name}")


def generate_department_changes_tables():
    """Generate top 10 department changes for CA, PC, PN."""
    print("Generating department changes top 10 tables...")

    # Load department changes
    df = pd.read_csv(TABLES_DIR / "department_changes_2019_2024.csv")

    # Function to generate table for one party
    def make_table(party, label_suffix, caption_party):
        df_party = df[df['Partido'] == party].copy()

        # Check if Cambio_pp is string or float
        if df_party['Cambio_pp'].dtype == 'object':
            # It's a string, convert it
            df_party['Cambio_numeric'] = df_party['Cambio_pp'].str.replace('+', '').astype(float)
        else:
            # Already numeric
            df_party['Cambio_numeric'] = df_party['Cambio_pp']

        # Get top 10 with most negative change (biggest improvement)
        df_party = df_party.sort_values('Cambio_numeric').head(10)

        latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Top 10 departamentos con mayor reducción de defección {caption_party} (2019 $\\to$ 2024). Valores negativos indican mejora en cohesión coalicional.}}
\\label{{tab:dept_changes_{label_suffix}}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Departamento}} & \\textbf{{Cambio (pp)}} \\\\
\\midrule
"""

        for _, row in df_party.iterrows():
            latex += f"{row['Departamento']} & {row['Cambio_pp']} \\\\\n"

        latex += """\\bottomrule
\\multicolumn{2}{l}{\\footnotesize Todos los valores negativos representan mejora en cohesión.} \\\\
\\end{tabular}
\\end{table}
"""
        return latex

    # Generate CA table
    latex_ca = make_table('CA', 'ca_top10', 'Cabildo Abierto')
    output_ca = LATEX_DIR / "department_changes_ca_top10.tex"
    with open(output_ca, "w", encoding="utf-8") as f:
        f.write(latex_ca)
    print(f"  OK - Saved CA to {output_ca.name}")

    # Generate PC table
    latex_pc = make_table('PC', 'pc_top10', 'Partido Colorado')
    output_pc = LATEX_DIR / "department_changes_pc_top10.tex"
    with open(output_pc, "w", encoding="utf-8") as f:
        f.write(latex_pc)
    print(f"  OK - Saved PC to {output_pc.name}")

    # Generate PN table
    latex_pn = make_table('PN', 'pn_top10', 'Partido Nacional')
    output_pn = LATEX_DIR / "department_changes_pn_top10.tex"
    with open(output_pn, "w", encoding="utf-8") as f:
        f.write(latex_pn)
    print(f"  OK - Saved PN to {output_pn.name}")


def generate_correlation_table():
    """Generate correlation analysis table."""
    print("Generating correlation table...")

    # Load correlation data
    df = pd.read_csv(TABLES_DIR / "correlation_analysis.csv")

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Estabilidad geográfica 2019-2024: Correlaciones de Pearson entre tasas de defección departamentales. Valores cercanos a cero indican patrones geográficos no predecibles entre elecciones.}
\label{tab:correlation_2019_2024}
\begin{tabular}{lrrr}
\toprule
\textbf{Partido} & \textbf{Correlación (r)} & \textbf{p-valor} & \textbf{R²} \\
\midrule
"""

    for _, row in df.iterrows():
        latex += f"{row['Partido']} & {row['Correlacion_Pearson']:.3f} & {row['P_value']:.3f} & {row['R_squared']:.3f} \\\\\n"

    latex += r"""\bottomrule
\multicolumn{4}{l}{\footnotesize Ninguna correlación es estadísticamente significativa (p > 0.05).} \\
\multicolumn{4}{l}{\footnotesize Patrones departamentales de 2024 no son predecibles desde 2019.} \\
\end{tabular}
\end{table}
"""

    output_path = LATEX_DIR / "correlation_table.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"  OK - Saved to {output_path.name}")


def main():
    """Generate all temporal analysis LaTeX tables."""
    print("="*60)
    print("GENERATING TEMPORAL ANALYSIS LATEX TABLES")
    print("="*60)
    print()

    try:
        generate_pi_national_results_2024()
        generate_national_transition_matrix_2019()
        generate_comparison_table_2019_2024()
        generate_department_changes_tables()
        generate_correlation_table()

        print()
        print("="*60)
        print("SUCCESS - ALL TABLES GENERATED")
        print("="*60)
        print()
        print("Generated files in outputs/tables/latex/:")
        print("  - pi_national_results_2024.tex")
        print("  - national_transition_matrix_2019.tex")
        print("  - comparison_table_2019_2024.tex")
        print("  - department_changes_ca_top10.tex")
        print("  - department_changes_pc_top10.tex")
        print("  - department_changes_pn_top10.tex")
        print("  - correlation_table.tex")
        print()

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
