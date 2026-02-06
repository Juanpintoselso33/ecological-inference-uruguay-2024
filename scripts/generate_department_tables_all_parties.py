"""
Generate LaTeX tables for department-level PC and PN defection analysis.

This script creates top 10 department tables for Partido Colorado and Partido Nacional
defections to FA, complementing the existing CA department table.
"""

import pandas as pd
from pathlib import Path

def format_ci(lower, upper):
    """Format 95% credible interval."""
    return f"[{lower:.1f}, {upper:.1f}]"

def create_pc_department_table(df: pd.DataFrame, output_path: Path):
    """Create LaTeX table for top 10 PC defection departments."""

    # Sort by PC→FA defection rate
    df_sorted = df.sort_values('pc_to_fa', ascending=False).head(10)

    # Calculate estimated votes to FA
    df_sorted['pc_votes_to_fa'] = (df_sorted['pc_votes'] * df_sorted['pc_to_fa']).astype(int)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Top 10 Departments by Partido Colorado Defection to Frente Amplio}",
        r"\label{tab:dept_pc_top10}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Department} & \textbf{N Circuits} & \textbf{PC Votes} & \textbf{PC$\rightarrow$FA (\%)} & \textbf{95\% CI} & \textbf{Est. Votes to FA} & \textbf{Max $\hat{R}$} \\",
        r"\midrule"
    ]

    for _, row in df_sorted.iterrows():
        dept = row['departamento']
        n = int(row['n_circuits'])
        pc_votes = int(row['pc_votes'])
        pc_fa_pct = row['pc_to_fa'] * 100
        ci = format_ci(row['pc_to_fa_lower'] * 100, row['pc_to_fa_upper'] * 100)
        est_votes = int(row['pc_votes_to_fa'])
        max_rhat = row['max_rhat']

        line = f"{dept} & {n:,} & {pc_votes:,} & {pc_fa_pct:.1f} & {ci} & {est_votes:,} & {max_rhat:.4f} \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize \textit{Note:} PC, Partido Colorado; FA, Frente Amplio. Departments ranked by PC$\rightarrow$FA transition probability. Estimated votes calculated as PC primera vuelta votes $\times$ transition probability. Max $\hat{R}$ values < 1.01 indicate MCMC convergence.",
        r"\end{flushleft}",
        r"\end{table}"
    ])

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"✓ Created PC department table: {output_path}")

def create_pn_department_table(df: pd.DataFrame, output_path: Path):
    """Create LaTeX table for top 10 PN defection departments."""

    # Sort by PN→FA defection rate
    df_sorted = df.sort_values('pn_to_fa', ascending=False).head(10)

    # Calculate estimated votes to FA
    df_sorted['pn_votes_to_fa'] = (df_sorted['pn_votes'] * df_sorted['pn_to_fa']).astype(int)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Top 10 Departments by Partido Nacional Defection to Frente Amplio}",
        r"\label{tab:dept_pn_top10}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Department} & \textbf{N Circuits} & \textbf{PN Votes} & \textbf{PN$\rightarrow$FA (\%)} & \textbf{95\% CI} & \textbf{Est. Votes to FA} & \textbf{Max $\hat{R}$} \\",
        r"\midrule"
    ]

    for _, row in df_sorted.iterrows():
        dept = row['departamento']
        n = int(row['n_circuits'])
        pn_votes = int(row['pn_votes'])
        pn_fa_pct = row['pn_to_fa'] * 100
        ci = format_ci(row['pn_to_fa_lower'] * 100, row['pn_to_fa_upper'] * 100)
        est_votes = int(row['pn_votes_to_fa'])
        max_rhat = row['max_rhat']

        line = f"{dept} & {n:,} & {pn_votes:,} & {pn_fa_pct:.1f} & {ci} & {est_votes:,} & {max_rhat:.4f} \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize \textit{Note:} PN, Partido Nacional; FA, Frente Amplio. Departments ranked by PN$\rightarrow$FA transition probability. Estimated votes calculated as PN primera vuelta votes $\times$ transition probability. Max $\hat{R}$ values < 1.01 indicate MCMC convergence.",
        r"\end{flushleft}",
        r"\end{table}"
    ])

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"✓ Created PN department table: {output_path}")

def main():
    """Main execution function."""

    # Paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "outputs" / "tables" / "transfers_by_department.csv"
    latex_dir = base_dir / "outputs" / "tables" / "latex"

    latex_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading department data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} departments")
    print(f"Columns: {df.columns.tolist()}")

    # Generate tables
    pc_output = latex_dir / "department_pc_top10_table.tex"
    pn_output = latex_dir / "department_pn_top10_table.tex"

    create_pc_department_table(df, pc_output)
    create_pn_department_table(df, pn_output)

    print("\n✓ Department tables generated successfully!")
    print(f"  - PC table: {pc_output}")
    print(f"  - PN table: {pn_output}")

if __name__ == "__main__":
    main()
