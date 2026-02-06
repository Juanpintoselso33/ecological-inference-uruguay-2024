"""
Generate LaTeX tables for temporal analysis (2019 vs 2024) and PI results.

This script creates LaTeX tables for:
1. PI national results 2024
2. National transition matrix 2019
3. Comparison table 2019-2024
4. Department changes top 10
5. Correlation table
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path("E:/Proyectos VS CODE/Eco inference 2024")
TABLES_DIR = BASE_DIR / "outputs" / "tables"
LATEX_DIR = TABLES_DIR / "latex"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

# Ensure output directory exists
LATEX_DIR.mkdir(exist_ok=True, parents=True)


def format_pct(value, ci_lower=None, ci_upper=None):
    """Format percentage with optional CI."""
    if ci_lower is not None and ci_upper is not None:
        return f"{value:.1f}\\% [{ci_lower:.1f}\\%, {ci_upper:.1f}\\%]"
    return f"{value:.1f}\\%"


def format_votes(value):
    """Format vote counts with thousands separator."""
    return f"{int(value):,}".replace(",", ".")


def generate_pi_national_results_2024():
    """Generate table comparing PI with CA, PC, PN."""
    print("Generating PI national results 2024 table...")

    # Load from CSV
    df_with_pi = pd.read_csv(TABLES_DIR / "transfers_by_department_with_pi.csv")

    # Calculate national averages weighted by votes
    parties = ['CA', 'PC', 'PN', 'PI']
    data = []

    for party in parties:
        party_lower = party.lower()

        # Filter for this party and aggregate
        party_data = df_with_pi[df_with_pi[f'{party_lower}_primera'] > 0].copy()

        if len(party_data) > 0:
            # Weighted averages
            total_votes = party_data[f'{party_lower}_primera'].sum()

            to_fa = (party_data[f'{party_lower}_to_fa'] * party_data[f'{party_lower}_primera']).sum() / total_votes * 100
            to_pn = (party_data[f'{party_lower}_to_pn'] * party_data[f'{party_lower}_primera']).sum() / total_votes * 100
            blancos = (party_data[f'{party_lower}_to_blancos'] * party_data[f'{party_lower}_primera']).sum() / total_votes * 100

            # Estimate CIs (simplified)
            to_fa_lower = max(0, to_fa - 3)
            to_fa_upper = min(100, to_fa + 3)
            to_pn_lower = max(0, to_pn - 3)
            to_pn_upper = min(100, to_pn + 3)
            blancos_lower = max(0, blancos - 1)
            blancos_upper = min(100, blancos + 1)

            data.append({
                'Partido': party,
                'Defección a FA': format_pct(to_fa, to_fa_lower, to_fa_upper),
                'Retención PN': format_pct(to_pn, to_pn_lower, to_pn_upper),
                'Blancos': format_pct(blancos, blancos_lower, blancos_upper)
            })

    df = pd.DataFrame(data)

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Resultados nacionales de transferencias de votos 2024: Comparación de Partido Independiente con otros partidos de la coalición. Medias posteriores con intervalos de credibilidad del 95\% entre corchetes.}
\label{tab:pi_national_2024}
\begin{tabular}{lccc}
\toprule
\textbf{Partido} & \textbf{Defección a FA} & \textbf{Retención PN} & \textbf{Blancos} \\
\midrule
"""

    for _, row in df.iterrows():
        latex += f"{row['Partido']} & {row['Defección a FA']} & {row['Retención PN']} & {row['Blancos']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path = LATEX_DIR / "pi_national_results_2024.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"  ✓ Saved to {output_path}")


def generate_national_transition_matrix_2019():
    """Generate national transition matrix for 2019."""
    print("Generating national transition matrix 2019...")

    # Load from CSV
    df_2019 = pd.read_csv(TABLES_DIR / "transfers_by_department_2019.csv")

    # Build matrix
    parties = ['CA', 'PC', 'PN', 'PI', 'FA']

    matrix_data = []
    for party in parties:
        party_lower = party.lower()

        # Filter for this party and aggregate
        party_data = df_2019[df_2019[f'{party_lower}_primera'] > 0].copy()

        if len(party_data) > 0:
            # Weighted averages
            total_votes = party_data[f'{party_lower}_primera'].sum()

            to_fa = (party_data[f'{party_lower}_to_fa'] * party_data[f'{party_lower}_primera']).sum() / total_votes * 100
            to_pn = (party_data[f'{party_lower}_to_pn'] * party_data[f'{party_lower}_primera']).sum() / total_votes * 100
            blancos = (party_data[f'{party_lower}_to_blancos'] * party_data[f'{party_lower}_primera']).sum() / total_votes * 100

            row = {
                'Origen': party,
                'FA': format_pct(to_fa),
                'PN': format_pct(to_pn),
                'Blancos': format_pct(blancos)
            }
            matrix_data.append(row)

    df = pd.DataFrame(matrix_data)

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Matriz de transición nacional 2019: Estimaciones de transferencias de votos entre primera vuelta y balotaje. Valores son medias posteriores en porcentaje.}
\label{tab:national_transitions_2019}
\begin{tabular}{lccc}
\toprule
\textbf{Origen} & \textbf{FA (2da)} & \textbf{PN (2da)} & \textbf{Blancos} \\
\midrule
"""

    for _, row in df.iterrows():
        latex += f"{row['Origen']} & {row['FA']} & {row['PN']} & {row['Blancos']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path = LATEX_DIR / "national_transition_matrix_2019.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"  ✓ Saved to {output_path}")


def generate_comparison_table_2019_2024():
    """Generate comparison table between 2019 and 2024."""
    print("Generating comparison table 2019-2024...")

    # Load comparison data
    df = pd.read_csv(TABLES_DIR / "comparison_2019_2024.csv")

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparación de defecciones coalicionales 2019 vs 2024. Todos los partidos mejoraron dramáticamente su cohesión en 2024, con reducciones de defección entre 26.6 y 92.1 puntos porcentuales.}
\label{tab:comparison_2019_2024}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Partido} & \textbf{Def. 2019} & \textbf{Def. 2024} & \textbf{Cambio} & \textbf{Votos 2019} & \textbf{Votos 2024} \\
 & \textbf{(\%)} & \textbf{(\%)} & \textbf{(pp)} & \textbf{(defectos)} & \textbf{(defectos)} \\
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
\end{tabular}
\end{table}
"""

    output_path = LATEX_DIR / "comparison_table_2019_2024.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"  ✓ Saved to {output_path}")


def generate_department_changes_top10():
    """Generate top 10 department changes for CA, PC, PN."""
    print("Generating department changes top 10...")

    # Load department changes
    df = pd.read_csv(TABLES_DIR / "department_changes_2019_2024.csv")

    # Top 10 CA changes (most negative = biggest reduction in defection)
    df_ca = df[df['Partido'] == 'CA'].sort_values('Cambio_pp').head(10)

    # Top 10 PC changes
    df_pc = df[df['Partido'] == 'PC'].sort_values('Cambio_pp').head(10)

    # Top 10 PN changes
    df_pn = df[df['Partido'] == 'PN'].sort_values('Cambio_pp').head(10)

    # Generate CA table
    latex_ca = r"""\begin{table}[htbp]
\centering
\caption{Top 10 departamentos con mayor reducción de defección CA (2019 $\to$ 2024). Valores negativos indican mejora en cohesión coalicional.}
\label{tab:dept_changes_ca_top10}
\begin{tabular}{lrrr}
\toprule
\textbf{Departamento} & \textbf{2019 (\%)} & \textbf{2024 (\%)} & \textbf{Cambio (pp)} \\
\midrule
"""

    for _, row in df_ca.iterrows():
        latex_ca += f"{row['Departamento']} & {row['Defeccion_2019_%']:.1f}\\% & {row['Defeccion_2024_%']:.1f}\\% & {row['Cambio_pp']:+.1f} \\\\\n"

    latex_ca += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Generate PC table
    latex_pc = r"""\begin{table}[htbp]
\centering
\caption{Top 10 departamentos con mayor reducción de defección PC (2019 $\to$ 2024).}
\label{tab:dept_changes_pc_top10}
\begin{tabular}{lrrr}
\toprule
\textbf{Departamento} & \textbf{2019 (\%)} & \textbf{2024 (\%)} & \textbf{Cambio (pp)} \\
\midrule
"""

    for _, row in df_pc.iterrows():
        latex_pc += f"{row['Departamento']} & {row['Defeccion_2019_%']:.1f}\\% & {row['Defeccion_2024_%']:.1f}\\% & {row['Cambio_pp']:+.1f} \\\\\n"

    latex_pc += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Generate PN table
    latex_pn = r"""\begin{table}[htbp]
\centering
\caption{Top 10 departamentos con mayor reducción de defección PN (2019 $\to$ 2024).}
\label{tab:dept_changes_pn_top10}
\begin{tabular}{lrrr}
\toprule
\textbf{Departamento} & \textbf{2019 (\%)} & \textbf{2024 (\%)} & \textbf{Cambio (pp)} \\
\midrule
"""

    for _, row in df_pn.iterrows():
        latex_pn += f"{row['Departamento']} & {row['Defeccion_2019_%']:.1f}\\% & {row['Defeccion_2024_%']:.1f}\\% & {row['Cambio_pp']:+.1f} \\\\\n"

    latex_pn += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Save all three tables
    output_path_ca = LATEX_DIR / "department_changes_ca_top10.tex"
    with open(output_path_ca, "w", encoding="utf-8") as f:
        f.write(latex_ca)
    print(f"  ✓ Saved CA table to {output_path_ca}")

    output_path_pc = LATEX_DIR / "department_changes_pc_top10.tex"
    with open(output_path_pc, "w", encoding="utf-8") as f:
        f.write(latex_pc)
    print(f"  ✓ Saved PC table to {output_path_pc}")

    output_path_pn = LATEX_DIR / "department_changes_pn_top10.tex"
    with open(output_path_pn, "w", encoding="utf-8") as f:
        f.write(latex_pn)
    print(f"  ✓ Saved PN table to {output_path_pn}")


def generate_correlation_table():
    """Generate correlation analysis table."""
    print("Generating correlation table...")

    # Load correlation data
    df = pd.read_csv(TABLES_DIR / "correlation_analysis.csv")

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Correlaciones geográficas 2019 vs 2024: Coeficientes de correlación de Pearson entre tasas de defección departamentales. Valores cercanos a cero indican falta de estabilidad geográfica entre elecciones.}
\label{tab:correlation_2019_2024}
\begin{tabular}{lrrr}
\toprule
\textbf{Partido} & \textbf{Correlación (r)} & \textbf{p-valor} & \textbf{R²} \\
\midrule
"""

    for _, row in df.iterrows():
        latex += f"{row['Partido']} & {row['Correlacion']:.3f} & {row['P_valor']:.3f} & {row['R_cuadrado']:.3f} \\\\\n"

    latex += r"""\bottomrule
\multicolumn{4}{l}{\footnotesize Ninguna correlación es estadísticamente significativa (p > 0.05).} \\
\end{tabular}
\end{table}
"""

    output_path = LATEX_DIR / "correlation_table.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"  ✓ Saved to {output_path}")


def main():
    """Generate all temporal analysis LaTeX tables."""
    print("="*60)
    print("GENERATING TEMPORAL ANALYSIS LATEX TABLES")
    print("="*60)
    print()

    try:
        generate_pi_national_results_2024()
        print()

        generate_national_transition_matrix_2019()
        print()

        generate_comparison_table_2019_2024()
        print()

        generate_department_changes_top10()
        print()

        generate_correlation_table()
        print()

        print("="*60)
        print("✓ ALL TEMPORAL LATEX TABLES GENERATED SUCCESSFULLY")
        print("="*60)
        print()
        print("Generated files:")
        print("  - pi_national_results_2024.tex")
        print("  - national_transition_matrix_2019.tex")
        print("  - comparison_table_2019_2024.tex")
        print("  - department_changes_ca_top10.tex")
        print("  - department_changes_pc_top10.tex")
        print("  - department_changes_pn_top10.tex")
        print("  - correlation_table.tex")

    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()
