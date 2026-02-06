"""
Visualization of Regional Comparison: Area Metropolitana vs Interior

Creates publication-quality figures comparing CA defection patterns across regions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_regional_comparison_plot():
    """Create comparison plot of CA transfers by region."""

    # Load data
    df = pd.read_csv('outputs/tables/region_detailed_results.csv')

    # Filter CA transfers
    ca_data = df[df['origin_party'] == 'CA'].copy()

    # Prepare data for plotting
    metro = ca_data[ca_data['region'] == 'Area_Metropolitana'].set_index('destination_party')
    interior = ca_data[ca_data['region'] == 'Interior'].set_index('destination_party')

    # Set up figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors
    colors = {'FA': '#FF4444', 'PN': '#4444FF', 'BLANCOS': '#888888'}

    # Plot 1: Bar chart comparison
    ax1 = axes[0]

    destinations = ['FA', 'PN', 'BLANCOS']
    x = np.arange(len(destinations))
    width = 0.35

    metro_means = [metro.loc[d, 'mean'] * 100 for d in destinations]
    interior_means = [interior.loc[d, 'mean'] * 100 for d in destinations]

    metro_errors = [
        [(metro.loc[d, 'mean'] - metro.loc[d, 'lower_95']) * 100 for d in destinations],
        [(metro.loc[d, 'upper_95'] - metro.loc[d, 'mean']) * 100 for d in destinations]
    ]

    interior_errors = [
        [(interior.loc[d, 'mean'] - interior.loc[d, 'lower_95']) * 100 for d in destinations],
        [(interior.loc[d, 'upper_95'] - interior.loc[d, 'mean']) * 100 for d in destinations]
    ]

    bars1 = ax1.bar(x - width/2, metro_means, width, label='Area Metropolitana',
                    color='#3498db', alpha=0.8, yerr=metro_errors, capsize=5)
    bars2 = ax1.bar(x + width/2, interior_means, width, label='Interior',
                    color='#e74c3c', alpha=0.8, yerr=interior_errors, capsize=5)

    ax1.set_xlabel('Destino', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Tasa de Transferencia (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Tasas de Transferencia de Cabildo Abierto por Región', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(destinations)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 70)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)

    # Plot 2: Focus on CA -> FA defection
    ax2 = axes[1]

    regions = ['Area\nMetropolitana', 'Interior']
    fa_means = [metro.loc['FA', 'mean'] * 100, interior.loc['FA', 'mean'] * 100]
    fa_errors = [
        [(metro.loc['FA', 'mean'] - metro.loc['FA', 'lower_95']) * 100,
         (interior.loc['FA', 'mean'] - interior.loc['FA', 'lower_95']) * 100],
        [(metro.loc['FA', 'upper_95'] - metro.loc['FA', 'mean']) * 100,
         (interior.loc['FA', 'upper_95'] - interior.loc['FA', 'mean']) * 100]
    ]

    bars = ax2.bar(regions, fa_means, color=['#3498db', '#e74c3c'], alpha=0.8,
                   yerr=fa_errors, capsize=8, width=0.6)

    ax2.set_ylabel('Tasa de Transferencia CA → FA (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Defección de CA al Frente Amplio:\nComparación Regional',
                 fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 70)

    # Add value labels with CI
    for i, (bar, mean, lower, upper) in enumerate(zip(bars, fa_means,
                                                       [metro.loc['FA', 'lower_95'] * 100, interior.loc['FA', 'lower_95'] * 100],
                                                       [metro.loc['FA', 'upper_95'] * 100, interior.loc['FA', 'upper_95'] * 100])):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{mean:.1f}%\n[{lower:.1f}, {upper:.1f}]',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add difference annotation
    diff = fa_means[1] - fa_means[0]
    ax2.text(0.5, 65, f'Diferencia: +{diff:.1f} pp\n(Interior mayor)',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()

    # Save
    output_dir = Path('outputs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'regional_comparison_ca_transfers.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    plt.show()


def create_all_transitions_heatmap():
    """Create heatmap showing all transitions by region."""

    # Load data
    df = pd.read_csv('outputs/tables/region_detailed_results.csv')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    regions = ['Area_Metropolitana', 'Interior']
    titles = ['Area Metropolitana\n(Montevideo + Canelones)', 'Interior\n(17 Departments)']

    for ax, region, title in zip(axes, regions, titles):
        # Filter data for this region
        region_data = df[df['region'] == region].copy()

        # Pivot to matrix form
        matrix = region_data.pivot(index='origin_party',
                                   columns='destination_party',
                                   values='mean')

        # Ensure order
        origin_order = ['CA', 'FA', 'OTROS', 'PC', 'PN']
        dest_order = ['FA', 'PN', 'BLANCOS']
        matrix = matrix.loc[origin_order, dest_order]

        # Create heatmap
        im = ax.imshow(matrix.values * 100, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

        # Set ticks
        ax.set_xticks(np.arange(len(dest_order)))
        ax.set_yticks(np.arange(len(origin_order)))
        ax.set_xticklabels(dest_order, fontsize=11)
        ax.set_yticklabels(origin_order, fontsize=11)

        # Labels
        ax.set_xlabel('Balotaje Noviembre', fontsize=12, fontweight='bold')
        ax.set_ylabel('Primera Vuelta Octubre', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

        # Add text annotations
        for i in range(len(origin_order)):
            for j in range(len(dest_order)):
                value = matrix.values[i, j] * 100
                text = ax.text(j, i, f'{value:.1f}%',
                             ha="center", va="center",
                             color="black" if value < 50 else "white",
                             fontsize=10, fontweight='bold')

        # Highlight CA row
        ax.add_patch(plt.Rectangle((-0.5, -0.5), len(dest_order), 1,
                                   fill=False, edgecolor='blue', linewidth=3))

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.02)
    cbar.set_label('Tasa de Transferencia (%)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Path('outputs/figures/regional_heatmap_all_transfers.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    plt.show()


def create_vote_impact_plot():
    """Create plot showing absolute vote impact by region."""

    # Load comparison data
    df_comp = pd.read_csv('outputs/tables/region_comparison.csv')

    # Calculate absolute votes
    metro = df_comp[df_comp['region'] == 'Area_Metropolitana'].iloc[0]
    interior = df_comp[df_comp['region'] == 'Interior'].iloc[0]

    # From log output:
    metro_ca_votes = 28774
    interior_ca_votes = 30638

    metro_to_fa = metro_ca_votes * metro['ca_to_fa_mean']
    metro_to_pn = metro_ca_votes * metro['ca_to_pn_mean']

    interior_to_fa = interior_ca_votes * interior['ca_to_fa_mean']
    interior_to_pn = interior_ca_votes * interior['ca_to_pn_mean']

    fig, ax = plt.subplots(figsize=(12, 7))

    regions = ['Area\nMetropolitana', 'Interior']
    x = np.arange(len(regions))
    width = 0.35

    to_fa = [metro_to_fa, interior_to_fa]
    to_pn = [metro_to_pn, interior_to_pn]

    bars1 = ax.bar(x - width/2, to_fa, width, label='Transferido a FA',
                   color='#FF4444', alpha=0.8)
    bars2 = ax.bar(x + width/2, to_pn, width, label='Transferido a PN (Coalición)',
                   color='#4444FF', alpha=0.8)

    ax.set_ylabel('Número de Votos', fontsize=13, fontweight='bold')
    ax.set_title('Impacto Absoluto de Votos: Transferencias CA por Región\n(Octubre → Noviembre 2024)',
                fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add net loss annotations
    metro_net = metro_to_fa - metro_to_pn
    interior_net = interior_to_fa - interior_to_pn

    ax.text(0, max(to_fa + to_pn) * 0.95,
           f'Pérdida neta coalición:\n{metro_net:,.0f} votos',
           ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.text(1, max(to_fa + to_pn) * 0.95,
           f'Pérdida neta coalición:\n{interior_net:,.0f} votos',
           ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save
    output_path = Path('outputs/figures/regional_vote_impact.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    plt.show()


if __name__ == '__main__':
    print("Creating regional comparison visualizations...")
    print("\n1. CA Transfer Rates Comparison")
    create_regional_comparison_plot()

    print("\n2. All Transitions Heatmap")
    create_all_transitions_heatmap()

    print("\n3. Absolute Vote Impact")
    create_vote_impact_plot()

    print("\nAll visualizations complete!")
