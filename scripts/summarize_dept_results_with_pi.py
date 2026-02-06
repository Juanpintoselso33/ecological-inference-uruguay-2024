"""
Summarize departmental results with PI included.
"""

import pandas as pd
import numpy as np

def main():
    # Load results
    df = pd.read_csv('outputs/tables/transfers_by_department_with_pi.csv')

    print("="*80)
    print("ANÁLISIS DEPARTAMENTAL CON PI SEPARADO - RESUMEN")
    print("="*80)
    print(f"Total departamentos: {len(df)}")
    print(f"Total circuitos: {df['n_circuits'].sum():,}")
    print()

    # PI Summary
    print("="*80)
    print("RESUMEN PARTIDO INDEPENDIENTE (PI)")
    print("="*80)
    total_pi = df['pi_votes'].sum()
    print(f"Total votos PI: {total_pi:,}")
    print(f"Proporción nacional: {total_pi/df['pn_votes'].sum()*100:.2f}% (relativo a PN)")
    print()

    # PI to FA transfers (sorted)
    print("PI -> FA (defección, ordenado de mayor a menor):")
    print("-"*80)
    df_sorted = df.sort_values('pi_to_fa', ascending=False)

    for _, row in df_sorted.iterrows():
        mean = row['pi_to_fa'] * 100
        lower = row['pi_to_fa_lower'] * 100
        upper = row['pi_to_fa_upper'] * 100
        votes = row['pi_votes']
        abs_transfer = mean/100 * votes

        print(f"{row['departamento']:20s}: {mean:5.1f}% [{lower:5.1f}% - {upper:5.1f}%] "
              f"({votes:4.0f} votos PI, ~{abs_transfer:4.0f} a FA)")

    print()

    # Summary statistics
    print("="*80)
    print("ESTADÍSTICAS DE PI -> FA")
    print("="*80)
    weighted_mean = np.average(df['pi_to_fa'], weights=df['pi_votes'])
    print(f"Media ponderada (por votos PI): {weighted_mean*100:.1f}%")
    print(f"Media simple: {df['pi_to_fa'].mean()*100:.1f}%")
    print(f"Mediana: {df['pi_to_fa'].median()*100:.1f}%")
    print(f"Desviación estándar: {df['pi_to_fa'].std()*100:.1f}%")
    print(f"Mínimo: {df['pi_to_fa'].min()*100:.1f}% ({df.loc[df['pi_to_fa'].idxmin(), 'departamento']})")
    print(f"Máximo: {df['pi_to_fa'].max()*100:.1f}% ({df.loc[df['pi_to_fa'].idxmax(), 'departamento']})")

    print()

    # Comparison with other parties
    print("="*80)
    print("COMPARACIÓN ENTRE PARTIDOS (defección a FA)")
    print("="*80)

    parties = ['CA', 'PC', 'PI', 'PN']
    for party in parties:
        col_fa = f'{party.lower()}_to_fa'
        col_votes = f'{party.lower()}_votes'

        total_votes = df[col_votes].sum()
        weighted_mean = np.average(df[col_fa], weights=df[col_votes])

        print(f"{party:5s}: {weighted_mean*100:5.1f}% defección a FA "
              f"({total_votes:,} votos totales)")

    print()

    # Geographic patterns
    print("="*80)
    print("PATRONES GEOGRÁFICOS - MONTEVIDEO VS INTERIOR")
    print("="*80)

    montevideo = df[df['departamento'] == 'Montevideo']
    if len(montevideo) > 0:
        mvd = montevideo.iloc[0]
        print(f"Montevideo:")
        print(f"  PI -> FA: {mvd['pi_to_fa']*100:.1f}%")
        print(f"  PI votos: {mvd['pi_votes']:,.0f}")
        print(f"  CA -> FA: {mvd['ca_to_fa']*100:.1f}%")
        print(f"  PC -> FA: {mvd['pc_to_fa']*100:.1f}%")
        print(f"  PN -> FA: {mvd['pn_to_fa']*100:.1f}%")
    else:
        print("Montevideo: No data available")

    print()

    interior = df[df['departamento'] != 'Montevideo']
    if len(interior) > 0:
        interior_pi_mean = np.average(interior['pi_to_fa'], weights=interior['pi_votes'])
        interior_ca_mean = np.average(interior['ca_to_fa'], weights=interior['ca_votes'])
        interior_pc_mean = np.average(interior['pc_to_fa'], weights=interior['pc_votes'])
        interior_pn_mean = np.average(interior['pn_to_fa'], weights=interior['pn_votes'])

        print(f"Interior (18 departamentos):")
        print(f"  PI -> FA: {interior_pi_mean*100:.1f}% (ponderado)")
        print(f"  PI votos totales: {interior['pi_votes'].sum():,.0f}")
        print(f"  CA -> FA: {interior_ca_mean*100:.1f}%")
        print(f"  PC -> FA: {interior_pc_mean*100:.1f}%")
        print(f"  PN -> FA: {interior_pn_mean*100:.1f}%")

    print()

    # MCMC Diagnostics
    print("="*80)
    print("DIAGNÓSTICOS MCMC")
    print("="*80)
    print(f"R-hat máximo: {df['max_rhat'].max():.4f} (debe ser < 1.01)")
    print(f"ESS mínimo: {df['min_ess'].min():.0f} (debe ser > 1000 idealmente)")

    problematic = df[df['max_rhat'] > 1.01]
    if len(problematic) > 0:
        print(f"\nDepartamentos con R-hat > 1.01:")
        for _, row in problematic.iterrows():
            print(f"  {row['departamento']}: R-hat = {row['max_rhat']:.4f}")
    else:
        print("\nTodos los departamentos tienen R-hat < 1.01 ✓")

    low_ess = df[df['min_ess'] < 1000]
    if len(low_ess) > 0:
        print(f"\nDepartamentos con ESS < 1000:")
        for _, row in low_ess.iterrows():
            print(f"  {row['departamento']}: ESS = {row['min_ess']:.0f}")

    print()
    print("="*80)
    print("NOTA: Este es un análisis con 1000 samples, 2 chains.")
    print("Para resultados definitivos, esperar análisis nacional con 4000 samples, 4 chains.")
    print("="*80)


if __name__ == '__main__':
    main()
