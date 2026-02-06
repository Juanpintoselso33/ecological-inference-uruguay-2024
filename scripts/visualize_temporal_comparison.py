"""
Visualización de comparación temporal 2019 vs 2024
==================================================

Genera gráficos para explorar cambios en transferencias de votos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import io

# Configurar encoding para Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
PLOT_DIR = BASE_DIR / "outputs" / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Cargar datos de comparación temporal."""
    diff_df = pd.read_csv(OUTPUT_DIR / "temporal_differences_rigorous.csv")
    het_df = pd.read_csv(OUTPUT_DIR / "heterogeneity_analysis.csv")
    return diff_df, het_df


def plot_forest_plots():
    """Crear forest plots para cada transferencia."""
    diff_df, _ = load_data()

    # Agrupar por comparison
    comparisons = diff_df['comparison'].unique()

    for comp in comparisons:
        subset = diff_df[diff_df['comparison'] == comp].copy()

        # Ordenar por delta
        subset = subset.sort_values('delta')

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plotear intervalos de confianza
        y_pos = np.arange(len(subset))

        ax.scatter(subset['delta'], y_pos, color='darkblue', s=100, zorder=3)
        ax.errorbar(subset['delta'], y_pos,
                   xerr=[subset['delta'] - subset['delta_ci_lower'],
                         subset['delta_ci_upper'] - subset['delta']],
                   fmt='none', ecolor='steelblue', capsize=5, linewidth=2, zorder=2)

        # Línea de no efecto
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Sin cambio')

        # Configurar
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subset['departamento'])
        ax.set_xlabel('Cambio en transferencia (2024 vs 2019)', fontsize=12, fontweight='bold')
        ax.set_title(f'Forest Plot: {comp.upper()}\nCambio en transferencia de votos con IC 95%',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"forest_plot_{comp}.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Guardado: forest_plot_{comp}.png")


def plot_change_distributions():
    """Graficar distribuciones de cambios por transferencia."""
    diff_df, _ = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    comparisons = sorted(diff_df['comparison'].unique())

    for ax, comp in zip(axes, comparisons):
        subset = diff_df[diff_df['comparison'] == comp]
        deltas = subset['delta'].values

        ax.hist(deltas, bins=8, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(deltas.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {deltas.mean():.3f}')
        ax.axvline(0, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Sin cambio')
        ax.set_xlabel('Cambio (Δ)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=11)
        ax.set_title(f'{comp.upper()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Distribución de cambios en transferencias (2019 vs 2024)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "change_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Guardado: change_distributions.png")


def plot_heterogeneity_summary():
    """Graficar resumen de heterogeneidad."""
    _, het_df = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: I-squared
    ax = axes[0]
    colors = ['darkred' if i > 75 else 'orange' if i > 50 else 'yellow' for i in het_df['i_squared']]
    bars = ax.barh(het_df['comparison'], het_df['i_squared'], color=colors, edgecolor='black', linewidth=2)

    ax.set_xlabel('I² (%)', fontsize=12, fontweight='bold')
    ax.set_title('Heterogeneidad entre departamentos\n(I²: % variación debida a heterogeneidad real)',
                fontsize=12, fontweight='bold')
    ax.set_xlim(0, 105)

    # Añadir valores
    for i, (comp, val) in enumerate(zip(het_df['comparison'], het_df['i_squared'])):
        ax.text(val + 2, i, f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

    ax.axvline(75, color='red', linestyle='--', alpha=0.5, label='Muy Alta (>75%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 2: Tau-squared
    ax = axes[1]
    bars = ax.barh(het_df['comparison'], het_df['tau_squared'], color='steelblue', edgecolor='black', linewidth=2)

    ax.set_xlabel('τ² (Varianza entre-estudios)', fontsize=12, fontweight='bold')
    ax.set_title('Varianza de efectos entre departamentos\n(τ²: cuantifica dispersión real)',
                fontsize=12, fontweight='bold')

    # Añadir valores
    for i, (comp, val) in enumerate(zip(het_df['comparison'], het_df['tau_squared'])):
        ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Análisis de heterogeneidad (Meta-análisis)',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "heterogeneity_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Guardado: heterogeneity_summary.png")


def plot_mean_changes_comparison():
    """Comparar cambios medios entre transferencias."""
    diff_df, _ = load_data()

    fig, ax = plt.subplots(figsize=(10, 6))

    comparisons = sorted(diff_df['comparison'].unique())
    means = [diff_df[diff_df['comparison'] == c]['delta'].mean() for c in comparisons]
    stds = [diff_df[diff_df['comparison'] == c]['delta'].std() for c in comparisons]

    x_pos = np.arange(len(comparisons))
    colors = ['darkred' if m < -0.35 else 'orange' if m < -0.20 else 'yellow' for m in means]

    bars = ax.bar(x_pos, means, yerr=stds, color=colors, edgecolor='black', linewidth=2,
                 capsize=10, error_kw={'linewidth': 2})

    ax.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Sin cambio')
    ax.set_ylabel('Cambio promedio (Δ)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Transferencia', fontsize=12, fontweight='bold')
    ax.set_title('Magnitud de cambios en transferencias\n(Media ± Desviación Estándar)',
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.upper() for c in comparisons], fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    # Añadir valores
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean - 0.05, f'{mean:.3f}\n±{std:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mean_changes_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Guardado: mean_changes_comparison.png")


def plot_significance_summary():
    """Graficar resumen de significancia por comparación."""
    sig_df = pd.read_csv(OUTPUT_DIR / "significance_tests.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    comparisons = sorted(sig_df['comparison'].unique())

    for ax, comp in zip(axes, comparisons):
        subset = sig_df[sig_df['comparison'] == comp]

        classification_counts = subset['classification'].value_counts()

        # Ordenar categorías
        categories = ["SIGNIFICATIVO (Bonferroni)", "MARGINAL (p < 0.05)", "NO SIGNIFICATIVO"]
        counts = [classification_counts.get(cat, 0) for cat in categories]
        colors_sig = ['darkgreen', 'orange', 'lightgray']

        bars = ax.bar(range(len(categories)), counts, color=colors_sig, edgecolor='black', linewidth=2)

        ax.set_ylabel('Número de departamentos', fontsize=11, fontweight='bold')
        ax.set_title(f'{comp.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 20)
        ax.grid(True, alpha=0.3, axis='y')

        # Añadir valores
        for i, count in enumerate(counts):
            ax.text(i, count + 0.2, str(count), ha='center', fontsize=11, fontweight='bold')

    plt.suptitle('Significancia estadística por transferencia\n(Corrección Bonferroni: α = 0.000877)',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "significance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Guardado: significance_summary.png")


def plot_scatter_2019_vs_2024():
    """Graficar scatter 2019 vs 2024 para cada transferencia."""
    diff_df, _ = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    comparisons = sorted(diff_df['comparison'].unique())

    for ax, comp in zip(axes, comparisons):
        subset = diff_df[diff_df['comparison'] == comp].copy()

        # Extraer valores 2019 y 2024
        col_2019 = f"{comp.split('_')[0]}_to_fa_2019"
        col_2024 = f"{comp.split('_')[0]}_to_fa_2024"

        x = subset[col_2019]
        y = subset[col_2024]

        # Color por significancia
        sig_df = pd.read_csv(OUTPUT_DIR / "significance_tests.csv")
        sig_subset = sig_df[sig_df['comparison'] == comp].set_index('departamento')

        colors = ['darkred' if sig_subset.loc[dept, 'classification'] == "SIGNIFICATIVO (Bonferroni)" else 'steelblue'
                 for dept in subset['departamento']]

        scatter = ax.scatter(x, y, s=150, c=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Linea diagonal (sin cambio)
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, alpha=0.5, label='Sin cambio')

        # Etiquetas
        for pos, (idx, row) in enumerate(subset.iterrows()):
            ax.annotate(row['departamento'][:3], (x.values[pos], y.values[pos]),
                       fontsize=8, alpha=0.7, ha='center')

        ax.set_xlabel('2019', fontsize=11, fontweight='bold')
        ax.set_ylabel('2024', fontsize=11, fontweight='bold')
        ax.set_title(f'{comp.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(min_val - 0.05, max_val + 0.05)
        ax.set_ylim(min_val - 0.05, max_val + 0.05)

    plt.suptitle('Transferencias 2019 vs 2024 por departamento\n(Rojo: cambio significativo Bonferroni)',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "scatter_2019_vs_2024.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Guardado: scatter_2019_vs_2024.png")


def main():
    """Ejecutar todas las visualizaciones."""
    print("\n" + "="*70)
    print("GENERANDO VISUALIZACIONES - COMPARACION TEMPORAL")
    print("="*70 + "\n")

    try:
        plot_forest_plots()
        plot_change_distributions()
        plot_heterogeneity_summary()
        plot_mean_changes_comparison()
        plot_significance_summary()
        plot_scatter_2019_vs_2024()

        print("\n" + "="*70)
        print("[OK] TODAS LAS VISUALIZACIONES GENERADAS EXITOSAMENTE")
        print("="*70)
        print(f"\nArchivos guardados en: {PLOT_DIR}")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
