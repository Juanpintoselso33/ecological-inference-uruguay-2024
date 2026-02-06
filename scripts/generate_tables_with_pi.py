#!/usr/bin/env python3
"""
Script para generar tablas LaTeX con PI incluido.

Este script regenera las tablas críticas con Partido Independiente separado.

Uso:
    python generate_tables_with_pi.py
"""

import pandas as pd
import pickle
from pathlib import Path


def generate_national_matrix_with_pi(pkl_path: str, output_path: str) -> None:
    """
    Genera matriz de transición nacional con PI incluido.

    Args:
        pkl_path: Path al archivo national_transfers_with_pi.pkl
        output_path: Path al archivo LaTeX de salida
    """
    # Load results
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    # Extract posterior means and CIs
    posterior_mean = results['posterior_mean']
    ci_lower = results['credible_interval_lower']
    ci_upper = results['credible_interval_upper']

    # Origin parties (rows)
    origins = ['CA', 'FA', 'OTROS', 'PC', 'PI', 'PN']
    # Destination parties (columns)
    destinations = ['FA', 'PN', 'Blancos']

    latex = r"""\begin{table}[htbp]
\centering
\caption{Matriz de Transición Nacional (Primera Vuelta $\rightarrow$ Ballotage) con PI}
\label{tab:national_matrix_pi}
\small
\begin{tabular}{lS[table-format=2.1]S[table-format=2.1]S[table-format=2.1]}
\toprule
{\textbf{Origen}} & {\textbf{FA (\%)}} & {\textbf{PN (\%)}} & {\textbf{Blancos (\%)}} \\
\midrule
"""

    for i, origin in enumerate(origins):
        means = [posterior_mean[i, j] * 100 for j in range(3)]
        latex += f"{origin} & {means[0]:.1f} & {means[1]:.1f} & {means[2]:.1f} \\\\\n"

        # Add CI row
        cis = [f"[{ci_lower[i,j]*100:.1f}, {ci_upper[i,j]*100:.1f}]" for j in range(3)]
        latex += f" & {{{cis[0]}}} & {{{cis[1]}}} & {{{cis[2]}}} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{flushleft}
\small \textit{Nota:} CA: Cabildo Abierto; FA: Frente Amplio; PC: Partido Colorado; PI: Partido Independiente; PN: Partido Nacional. Los valores representan medias posteriores con intervalos creíbles al 95\% entre corchetes. Max $\hat{R} < 1.01$ indica convergencia MCMC adecuada.
\end{flushleft}
\end{table}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f"OK: Tabla matriz nacional con PI generada: {output_path}")


def generate_urban_rural_table_with_pi(csv_path: str, output_path: str) -> None:
    """
    Genera tabla urbano/rural con PI incluido.

    Args:
        csv_path: Path a urban_rural_with_pi.csv
        output_path: Path al archivo LaTeX de salida
    """
    df = pd.read_csv(csv_path)

    # Convert to percentages
    value_cols = [c for c in df.columns if c not in ['stratum', 'n_circuits']]
    for col in value_cols:
        df[col] = (df[col] * 100).round(1)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Tasas de Defección de Partidos de la Coalición por Clasificación Urbana/Rural (con PI)}
\label{tab:urban_rural_pi}
\small
\begin{tabular}{lS[table-format=4]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]}
\toprule
{\textbf{Estrato}} & {\textbf{N}} & \multicolumn{2}{c}{\textbf{CA}} & \multicolumn{2}{c}{\textbf{PC}} & \multicolumn{2}{c}{\textbf{PI}} & \multicolumn{2}{c}{\textbf{PN}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
 & & {\textbf{$\rightarrow$FA (\%)}} & {\textbf{$\rightarrow$PN (\%)}} & {\textbf{$\rightarrow$FA (\%)}} & {\textbf{$\rightarrow$PN (\%)}} & {\textbf{$\rightarrow$FA (\%)}} & {\textbf{$\rightarrow$PN (\%)}} & {\textbf{$\rightarrow$FA (\%)}} & {\textbf{$\rightarrow$PN (\%)}} \\
\midrule
"""

    for _, row in df.iterrows():
        stratum = 'Urbano' if row['stratum'] == 'urbano' else 'Rural'
        n = int(row['n_circuits'])

        # Extract values (assuming column names like ca_to_fa_mean, ca_to_fa_lower, etc.)
        latex += f"{stratum} & {n} & "
        latex += f"{row['ca_to_fa_mean']:.1f} & {row['ca_to_pn_mean']:.1f} & "
        latex += f"{row['pc_to_fa_mean']:.1f} & {row['pc_to_pn_mean']:.1f} & "
        latex += f"{row['pi_to_fa_mean']:.1f} & {row['pi_to_pn_mean']:.1f} & "
        latex += f"{row['pn_to_fa_mean']:.1f} & {row['pn_to_pn_mean']:.1f} \\\\\n"

        # Add CI row
        latex += " & & "
        latex += f"{{[{row['ca_to_fa_lower']:.1f}, {row['ca_to_fa_upper']:.1f}]}} & "
        latex += f"{{[{row['ca_to_pn_lower']:.1f}, {row['ca_to_pn_upper']:.1f}]}} & "
        latex += f"{{[{row['pc_to_fa_lower']:.1f}, {row['pc_to_fa_upper']:.1f}]}} & "
        latex += f"{{[{row['pc_to_pn_lower']:.1f}, {row['pc_to_pn_upper']:.1f}]}} & "
        latex += f"{{[{row['pi_to_fa_lower']:.1f}, {row['pi_to_fa_upper']:.1f}]}} & "
        latex += f"{{[{row['pi_to_pn_lower']:.1f}, {row['pi_to_pn_upper']:.1f}]}} & "
        latex += f"{{[{row['pn_to_fa_lower']:.1f}, {row['pn_to_fa_upper']:.1f}]}} & "
        latex += f"{{[{row['pn_to_pn_lower']:.1f}, {row['pn_to_pn_upper']:.1f}]}} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{flushleft}
\small \textit{Nota:} CA: Cabildo Abierto; PC: Partido Colorado; PI: Partido Independiente; PN: Partido Nacional; FA: Frente Amplio. Los valores representan medias posteriores con intervalos creíbles al 95\% entre corchetes. Max $\hat{R} < 1.01$ indica convergencia MCMC adecuada.
\end{flushleft}
\end{table}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f"OK: Tabla urbano/rural con PI generada: {output_path}")


def generate_region_table_with_pi(csv_path: str, output_path: str) -> None:
    """
    Genera tabla regional con PI incluido.

    Args:
        csv_path: Path a region_with_pi.csv
        output_path: Path al archivo LaTeX de salida
    """
    df = pd.read_csv(csv_path)

    # Convert to percentages
    value_cols = [c for c in df.columns if c not in ['region', 'n_circuits']]
    for col in value_cols:
        df[col] = (df[col] * 100).round(1)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Variación Regional en las Defecciones de la Coalición (con PI)}
\label{tab:region_pi}
\small
\begin{tabular}{lS[table-format=4]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]S[table-format=2.1]}
\toprule
{\textbf{Región}} & {\textbf{N}} & \multicolumn{2}{c}{\textbf{CA}} & \multicolumn{2}{c}{\textbf{PC}} & \multicolumn{2}{c}{\textbf{PI}} & \multicolumn{2}{c}{\textbf{PN}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
 & & {\textbf{$\rightarrow$FA (\%)}} & {\textbf{$\rightarrow$PN (\%)}} & {\textbf{$\rightarrow$FA (\%)}} & {\textbf{$\rightarrow$PN (\%)}} & {\textbf{$\rightarrow$FA (\%)}} & {\textbf{$\rightarrow$PN (\%)}} & {\textbf{$\rightarrow$FA (\%)}} & {\textbf{$\rightarrow$PN (\%)}} \\
\midrule
"""

    for _, row in df.iterrows():
        region = 'Área Metropolitana' if row['region'] == 'Metropolitana' else 'Interior'
        n = int(row['n_circuits'])

        latex += f"{region} & {n} & "
        latex += f"{row['ca_to_fa']:.1f} & {row['ca_to_pn']:.1f} & "
        latex += f"{row['pc_to_fa']:.1f} & {row['pc_to_pn']:.1f} & "
        latex += f"{row['pi_to_fa']:.1f} & {row['pi_to_pn']:.1f} & "
        latex += f"{row['pn_to_fa']:.1f} & {row['pn_to_pn']:.1f} \\\\\n"

        # Add CI row (only for FA transfers)
        latex += " & & "
        latex += f"{{[{row['ca_to_fa_lower']:.1f}, {row['ca_to_fa_upper']:.1f}]}} & {{--}} & "
        latex += f"{{[{row['pc_to_fa_lower']:.1f}, {row['pc_to_fa_upper']:.1f}]}} & {{--}} & "
        latex += f"{{[{row['pi_to_fa_lower']:.1f}, {row['pi_to_fa_upper']:.1f}]}} & {{--}} & "
        latex += f"{{[{row['pn_to_fa_lower']:.1f}, {row['pn_to_fa_upper']:.1f}]}} & {{--}} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{flushleft}
\small \textit{Nota:} CA: Cabildo Abierto; PC: Partido Colorado; PI: Partido Independiente; PN: Partido Nacional; FA: Frente Amplio. Área Metropolitana incluye Montevideo y Canelones. Los valores representan medias posteriores con intervalos creíbles al 95\% entre corchetes para transferencias a FA.
\end{flushleft}
\end{table}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f"OK: Tabla regional con PI generada: {output_path}")


def generate_vote_impact_table_with_pi(dept_csv: str, output_path: str) -> None:
    """
    Genera tabla de impacto de votos con PI incluido.

    Args:
        dept_csv: Path a transfers_by_department_with_pi.csv
        output_path: Path al archivo LaTeX de salida
    """
    df = pd.read_csv(dept_csv)

    # Calculate total estimated transfers
    parties = ['ca', 'pc', 'pi', 'pn']
    totals = {}

    for party in parties:
        if f'{party}_votes' in df.columns and f'{party}_to_fa' in df.columns:
            totals[party] = {
                'votes': df[f'{party}_votes'].sum(),
                'to_fa_rate': (df[f'{party}_to_fa'] * df[f'{party}_votes']).sum() / df[f'{party}_votes'].sum(),
                'estimated_fa': (df[f'{party}_to_fa'] * df[f'{party}_votes']).sum()
            }

    latex = r"""\begin{table}[htbp]
\centering
\caption{Impacto de Transferencias de Votos a FA por Partido (con PI)}
\label{tab:vote_impact_pi}
\begin{tabular}{lS[table-format=6.0]S[table-format=2.1]S[table-format=6.0]}
\toprule
{\textbf{Partido}} & {\textbf{Votos 1ra Vuelta}} & {\textbf{Tasa Defección (\%)}} & {\textbf{Votos Est. a FA}} \\
\midrule
"""

    party_names = {'ca': 'Cabildo Abierto', 'pc': 'Partido Colorado',
                   'pi': 'Partido Independiente', 'pn': 'Partido Nacional'}

    for party in parties:
        if party in totals:
            latex += f"{party_names[party]} & {int(totals[party]['votes'])} & "
            latex += f"{totals[party]['to_fa_rate']*100:.1f} & {int(totals[party]['estimated_fa'])} \\\\\n"

    latex += r"""\midrule
"""

    total_votes = sum(t['votes'] for t in totals.values())
    total_to_fa = sum(t['estimated_fa'] for t in totals.values())
    total_rate = (total_to_fa / total_votes) * 100

    latex += f"\\textbf{{TOTAL}} & {int(total_votes)} & {total_rate:.1f} & {int(total_to_fa)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{flushleft}
\small \textit{Nota:} Votos estimados a FA calculados como producto de tasa de defección por votos totales en primera vuelta. PI representa Partido Independiente, con ~41,000 votos en primera vuelta.
\end{flushleft}
\end{table}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f"OK: Tabla de impacto con PI generada: {output_path}")


def main():
    """Genera todas las tablas con PI."""

    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'outputs' / 'results'
    tables_dir = project_root / 'outputs' / 'tables'
    latex_dir = tables_dir / 'latex'

    latex_dir.mkdir(parents=True, exist_ok=True)

    print("Generando tablas LaTeX con PI...\n")

    # 1. National matrix
    try:
        generate_national_matrix_with_pi(
            str(results_dir / 'national_transfers_with_pi.pkl'),
            str(latex_dir / 'national_transition_matrix_pi.tex')
        )
    except Exception as e:
        print(f"ERROR: Error generando matriz nacional: {e}")

    # 2. Urban/Rural
    try:
        generate_urban_rural_table_with_pi(
            str(tables_dir / 'urban_rural_with_pi.csv'),
            str(latex_dir / 'urban_rural_comparison_table_pi.tex')
        )
    except Exception as e:
        print(f"ERROR: Error generando tabla urbano/rural: {e}")

    # 3. Regional
    try:
        generate_region_table_with_pi(
            str(tables_dir / 'region_with_pi.csv'),
            str(latex_dir / 'region_comparison_table_pi.tex')
        )
    except Exception as e:
        print(f"ERROR: Error generando tabla regional: {e}")

    # 4. Vote impact
    try:
        generate_vote_impact_table_with_pi(
            str(tables_dir / 'transfers_by_department_with_pi.csv'),
            str(latex_dir / 'vote_impact_table_pi.tex')
        )
    except Exception as e:
        print(f"ERROR: Error generando tabla de impacto: {e}")

    print("\nOK: Todas las tablas con PI han sido generadas exitosamente.")
    print(f"  Ubicación: {latex_dir}")


if __name__ == '__main__':
    main()
