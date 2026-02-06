# -*- coding: utf-8 -*-
"""
Generate publication-quality figures for geographic stratified analyses.
Includes urban/rural comparison, regional comparison, department-level map, and combined summary.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from pathlib import Path
import sys

# Handle encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# Set publication style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Define paths
output_dir = Path('outputs/tables')
figure_dir = Path('outputs/figures')
figure_dir.mkdir(parents=True, exist_ok=True)

# Color palette for colorblind accessibility
colorblind_palette = sns.color_palette("colorblind")

print("Loading data files...")

# Load urban/rural data (with PI if available, fallback to original)
try:
    urban_rural_df = pd.read_csv(output_dir / 'urban_rural_with_pi.csv')
    print(f"Urban/rural data (with PI) shape: {urban_rural_df.shape}")
except FileNotFoundError:
    urban_rural_df = pd.read_csv(output_dir / 'urban_rural_comparison.csv')
    print(f"Urban/rural data shape: {urban_rural_df.shape}")

# Load region data (with PI if available, fallback to original)
try:
    region_df = pd.read_csv(output_dir / 'region_with_pi.csv')
    print(f"Region data (with PI) shape: {region_df.shape}")
except FileNotFoundError:
    region_df = pd.read_csv(output_dir / 'region_comparison.csv')
    print(f"Region data shape: {region_df.shape}")

# Load department data (with PI if available, fallback to original)
try:
    department_df = pd.read_csv(output_dir / 'transfers_by_department_with_pi.csv')
    print(f"Department data (with PI) shape: {department_df.shape}")
except FileNotFoundError:
    department_df = pd.read_csv(output_dir / 'transfers_by_department.csv')
    print(f"Department data shape: {department_df.shape}")

# Load detailed urban/rural for additional comparisons
try:
    urban_rural_detailed = pd.read_csv(output_dir / 'urban_rural_detailed_comparison.csv')
    print(f"Urban/rural detailed data loaded successfully (shape: {urban_rural_detailed.shape})")
except FileNotFoundError:
    print("Warning: urban_rural_detailed_comparison.csv not found, some figures may be limited")

print("\n" + "="*80)
print("FIGURE 1: Urban vs Rural Forest Plot (CA to FA Defection)")
print("="*80)

fig1, ax1 = plt.subplots(figsize=(10, 5))

# Prepare forest plot data
strata = urban_rural_df['stratum'].tolist()
ca_to_fa_means = (urban_rural_df['ca_to_fa'] * 100).tolist()  # Convert to percentage
ca_to_fa_lower = (urban_rural_df['ca_to_fa_ci_lower'] * 100).tolist()
ca_to_fa_upper = (urban_rural_df['ca_to_fa_ci_upper'] * 100).tolist()

# Calculate error bar values
errors_lower = np.array(ca_to_fa_means) - np.array(ca_to_fa_lower)
errors_upper = np.array(ca_to_fa_upper) - np.array(ca_to_fa_means)

# Create forest plot
y_pos = np.arange(len(strata))
colors = [colorblind_palette[0], colorblind_palette[1]]

# Plot error bars and point estimates separately for each stratum
for i, (mean, lower, upper, color) in enumerate(zip(ca_to_fa_means, ca_to_fa_lower, ca_to_fa_upper, colors)):
    err_lower = mean - lower
    err_upper = upper - mean
    ax1.errorbar(mean, i,
                 xerr=[[err_lower], [err_upper]],
                 fmt='D', markersize=10, capsize=8, capthick=2,
                 elinewidth=2, ecolor=color, color=color, zorder=3)

# Add vertical reference line at overall mean
overall_mean = urban_rural_df['ca_to_fa'].mean() * 100
ax1.axvline(overall_mean, color='gray', linestyle='--', linewidth=1.5,
            label=f'Media general ({overall_mean:.1f}%)', alpha=0.7)

# Check for overlapping intervals
urban_ci = (ca_to_fa_lower[0], ca_to_fa_upper[0])
rural_ci = (ca_to_fa_lower[1], ca_to_fa_upper[1])
overlap = not (urban_ci[1] < rural_ci[0] or rural_ci[1] < urban_ci[0])

# Add annotation for overlapping intervals
if not overlap:
    ax1.text(0.98, 0.02, 'ICs 95% no superpuestos',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.set_yticks(y_pos)
ax1.set_yticklabels([s.capitalize() for s in strata])
ax1.set_xlabel('Tasa de Defección CA→FA (%)', fontsize=12)
ax1.set_title('Defección de CA a FA: Comparación Urbano vs Rural', fontsize=13, fontweight='bold')
ax1.grid(True, axis='x', alpha=0.3)
ax1.set_xlim(30, 75)

plt.tight_layout()
fig1.savefig(figure_dir / 'urban_rural_ca_defection_forest.png', dpi=300, bbox_inches='tight')
print(f"Guardado: outputs/figures/urban_rural_ca_defection_forest.png")
plt.close()

print("\n" + "="*80)
print("FIGURE 2: Regional Comparison (Bar Chart)")
print("="*80)

fig2, ax2 = plt.subplots(figsize=(12, 6))

# Prepare regional data
regions = region_df['region'].tolist()
ca_to_fa_reg = (region_df['ca_to_fa_mean'] * 100).tolist()
ca_to_fa_lower_reg = (region_df['ca_to_fa_lower'] * 100).tolist()
ca_to_fa_upper_reg = (region_df['ca_to_fa_upper'] * 100).tolist()

ca_to_pn_reg = (region_df['ca_to_pn_mean'] * 100).tolist()
ca_to_pn_lower_reg = (region_df['ca_to_pn_lower'] * 100).tolist()
ca_to_pn_upper_reg = (region_df['ca_to_pn_upper'] * 100).tolist()

# PC->FA data (if available)
pc_to_fa_reg = []
pc_to_fa_lower_reg = []
pc_to_fa_upper_reg = []

# Try to get PC->FA from detailed results
try:
    detailed_region = pd.read_csv(output_dir / 'region_detailed_results.csv', index_col=0)
    for region_name in regions:
        # Normalize region name for lookup
        lookup_name = region_name.replace('_', ' ')
        if lookup_name in detailed_region.index:
            row = detailed_region.loc[lookup_name]
            pc_to_fa_reg.append(row['pc_to_fa_mean'] * 100)
            pc_to_fa_lower_reg.append(row['pc_to_fa_lower'] * 100)
            pc_to_fa_upper_reg.append(row['pc_to_fa_upper'] * 100)
        else:
            pc_to_fa_reg.append(0)
            pc_to_fa_lower_reg.append(0)
            pc_to_fa_upper_reg.append(0)
except:
    pc_to_fa_reg = [0] * len(regions)
    pc_to_fa_lower_reg = [0] * len(regions)
    pc_to_fa_upper_reg = [0] * len(regions)

# Calculate errors
errors_ca_fa_lower = np.array(ca_to_fa_reg) - np.array(ca_to_fa_lower_reg)
errors_ca_fa_upper = np.array(ca_to_fa_upper_reg) - np.array(ca_to_fa_reg)

errors_ca_pn_lower = np.array(ca_to_pn_reg) - np.array(ca_to_pn_lower_reg)
errors_ca_pn_upper = np.array(ca_to_pn_upper_reg) - np.array(ca_to_pn_reg)

errors_pc_fa_lower = np.array(pc_to_fa_reg) - np.array(pc_to_fa_lower_reg)
errors_pc_fa_upper = np.array(pc_to_fa_upper_reg) - np.array(pc_to_fa_reg)

# Create grouped bar plot
x_pos = np.arange(len(regions))
bar_width = 0.25

# Plot bars
bars1 = ax2.bar(x_pos - bar_width, ca_to_fa_reg, bar_width,
                label='CA->FA', color=colorblind_palette[0], alpha=0.8)
bars2 = ax2.bar(x_pos, ca_to_pn_reg, bar_width,
                label='CA->PN', color=colorblind_palette[1], alpha=0.8)
bars3 = ax2.bar(x_pos + bar_width, pc_to_fa_reg, bar_width,
                label='PC->FA', color=colorblind_palette[2], alpha=0.8)

# Add error bars
ax2.errorbar(x_pos - bar_width, ca_to_fa_reg,
             yerr=[errors_ca_fa_lower, errors_ca_fa_upper],
             fmt='none', ecolor='black', capsize=5, elinewidth=1.5, alpha=0.6)
ax2.errorbar(x_pos, ca_to_pn_reg,
             yerr=[errors_ca_pn_lower, errors_ca_pn_upper],
             fmt='none', ecolor='black', capsize=5, elinewidth=1.5, alpha=0.6)
ax2.errorbar(x_pos + bar_width, pc_to_fa_reg,
             yerr=[errors_pc_fa_lower, errors_pc_fa_upper],
             fmt='none', ecolor='black', capsize=5, elinewidth=1.5, alpha=0.6)

ax2.set_xlabel('Región', fontsize=12, fontweight='bold')
ax2.set_ylabel('Tasa de Defección (%)', fontsize=12, fontweight='bold')
ax2.set_title('Defecciones de Partidos de la Coalición por Región', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([r.replace('_', ' ') for r in regions])
ax2.legend(loc='upper right', framealpha=0.95)
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_ylim(0, max(ca_to_fa_reg + ca_to_pn_reg + pc_to_fa_reg) + 15)

plt.tight_layout()
fig2.savefig(figure_dir / 'region_coalition_defections.png', dpi=300, bbox_inches='tight')
print(f"Saved: outputs/figures/region_coalition_defections.png")
plt.close()

print("\n" + "="*80)
print("FIGURE 3: Department-Level Visualization")
print("="*80)

# Create a bar chart instead showing departments sorted by CA->FA defection
fig3, ax3 = plt.subplots(figsize=(14, 8))

dept_sorted = department_df.sort_values('ca_to_fa', ascending=True)

y_pos = np.arange(len(dept_sorted))
ca_to_fa_dept = (dept_sorted['ca_to_fa'] * 100).values
ca_to_fa_lower_dept = (dept_sorted['ca_to_fa_lower'] * 100).values
ca_to_fa_upper_dept = (dept_sorted['ca_to_fa_upper'] * 100).values

errors_lower = ca_to_fa_dept - ca_to_fa_lower_dept
errors_upper = ca_to_fa_upper_dept - ca_to_fa_dept

# Create colormap for gradient coloring
norm = plt.Normalize(vmin=ca_to_fa_dept.min(), vmax=ca_to_fa_dept.max())
cmap = plt.cm.RdYlBu_r
colors_dept = cmap(norm(ca_to_fa_dept))

ax3.barh(y_pos, ca_to_fa_dept, color=colors_dept, edgecolor='black', linewidth=0.5)
ax3.errorbar(ca_to_fa_dept, y_pos,
            xerr=[errors_lower, errors_upper],
            fmt='none', ecolor='black', capsize=4, elinewidth=1, alpha=0.6)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(dept_sorted['departamento'].values, fontsize=10)
ax3.set_xlabel('Tasa de Defección CA→FA (%)', fontsize=12, fontweight='bold')
ax3.set_title('Tasa de Defección CA→FA por Departamento', fontsize=13, fontweight='bold')
ax3.grid(True, axis='x', alpha=0.3)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax3, orientation='vertical', pad=0.02)
cbar.set_label('Tasa de Defección (%)', fontsize=11)

plt.tight_layout()
fig3.savefig(figure_dir / 'department_ca_defection_map.png', dpi=300, bbox_inches='tight')
print(f"Saved: outputs/figures/department_ca_defection_map.png")
plt.close()

print("\n" + "="*80)
print("FIGURE 4: Combined Summary Figure (2x2 Panel)")
print("="*80)

fig4 = plt.figure(figsize=(16, 12))
gs = fig4.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# Panel A: Urban/Rural forest plot
ax_a = fig4.add_subplot(gs[0, 0])

y_pos_a = np.arange(len(strata))
for i, (mean, lower, upper, color) in enumerate(zip(ca_to_fa_means, ca_to_fa_lower, ca_to_fa_upper, colors)):
    err_lower = mean - lower
    err_upper = upper - mean
    ax_a.errorbar(mean, i,
                  xerr=[[err_lower], [err_upper]],
                  fmt='D', markersize=8, capsize=6, capthick=1.5,
                  elinewidth=1.5, ecolor=color, color=color, zorder=3)

ax_a.axvline(overall_mean, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

ax_a.set_yticks(y_pos_a)
ax_a.set_yticklabels([s.capitalize() for s in strata])
ax_a.set_xlabel('Tasa de Defección CA→FA (%)', fontsize=11)
ax_a.set_title('A. Comparación Urbano vs Rural (CA→FA)', fontsize=12, fontweight='bold')
ax_a.grid(True, axis='x', alpha=0.3)
ax_a.set_xlim(30, 75)

# Panel B: Regional bar chart
ax_b = fig4.add_subplot(gs[0, 1])

x_pos_b = np.arange(len(regions))
bar_width_b = 0.25

ax_b.bar(x_pos_b - bar_width_b, ca_to_fa_reg, bar_width_b,
         label='CA->FA', color=colorblind_palette[0], alpha=0.8)
ax_b.bar(x_pos_b, ca_to_pn_reg, bar_width_b,
         label='CA->PN', color=colorblind_palette[1], alpha=0.8)
ax_b.bar(x_pos_b + bar_width_b, pc_to_fa_reg, bar_width_b,
         label='PC->FA', color=colorblind_palette[2], alpha=0.8)

ax_b.errorbar(x_pos_b - bar_width_b, ca_to_fa_reg,
             yerr=[errors_ca_fa_lower, errors_ca_fa_upper],
             fmt='none', ecolor='black', capsize=4, elinewidth=1, alpha=0.6)
ax_b.errorbar(x_pos_b, ca_to_pn_reg,
             yerr=[errors_ca_pn_lower, errors_ca_pn_upper],
             fmt='none', ecolor='black', capsize=4, elinewidth=1, alpha=0.6)
ax_b.errorbar(x_pos_b + bar_width_b, pc_to_fa_reg,
             yerr=[errors_pc_fa_lower, errors_pc_fa_upper],
             fmt='none', ecolor='black', capsize=4, elinewidth=1, alpha=0.6)

ax_b.set_xlabel('Región', fontsize=11, fontweight='bold')
ax_b.set_ylabel('Tasa de Defección (%)', fontsize=11, fontweight='bold')
ax_b.set_title('B. Comparación Regional', fontsize=12, fontweight='bold')
ax_b.set_xticks(x_pos_b)
ax_b.set_xticklabels([r.replace('_', ' ') for r in regions], fontsize=10)
ax_b.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax_b.grid(True, axis='y', alpha=0.3)
ax_b.set_ylim(0, max(ca_to_fa_reg + ca_to_pn_reg + pc_to_fa_reg) + 12)

# Panel C: All coalition transfers by stratum
ax_c = fig4.add_subplot(gs[1, 0])

# Get all transitions from detailed comparison
transitions = urban_rural_detailed['transition'].tolist()
urban_means = (urban_rural_detailed['urban_mean'] * 100).tolist()
urban_lower = (urban_rural_detailed['urban_ci_lower'] * 100).tolist()
urban_upper = (urban_rural_detailed['urban_ci_upper'] * 100).tolist()
rural_means = (urban_rural_detailed['rural_mean'] * 100).tolist()
rural_lower = (urban_rural_detailed['rural_ci_lower'] * 100).tolist()
rural_upper = (urban_rural_detailed['rural_ci_upper'] * 100).tolist()

x_pos_c = np.arange(len(transitions))
bar_width_c = 0.35

urban_errors_lower = np.array(urban_means) - np.array(urban_lower)
urban_errors_upper = np.array(urban_upper) - np.array(urban_means)
rural_errors_lower = np.array(rural_means) - np.array(rural_lower)
rural_errors_upper = np.array(rural_upper) - np.array(rural_means)

ax_c.bar(x_pos_c - bar_width_c/2, urban_means, bar_width_c,
         label='Urbano', color=colorblind_palette[3], alpha=0.8)
ax_c.bar(x_pos_c + bar_width_c/2, rural_means, bar_width_c,
         label='Rural', color=colorblind_palette[4], alpha=0.8)

ax_c.errorbar(x_pos_c - bar_width_c/2, urban_means,
             yerr=[urban_errors_lower, urban_errors_upper],
             fmt='none', ecolor='black', capsize=4, elinewidth=1, alpha=0.6)
ax_c.errorbar(x_pos_c + bar_width_c/2, rural_means,
             yerr=[rural_errors_lower, rural_errors_upper],
             fmt='none', ecolor='black', capsize=4, elinewidth=1, alpha=0.6)

ax_c.set_xlabel('Transición de Partido', fontsize=11, fontweight='bold')
ax_c.set_ylabel('Tasa de Transición (%)', fontsize=11, fontweight='bold')
ax_c.set_title('C. Todas las Transiciones de la Coalición por Estrato', fontsize=12, fontweight='bold')
ax_c.set_xticks(x_pos_c)
# Format transition labels
transition_labels = [t.replace(' -> ', '\n>\n') for t in transitions]
ax_c.set_xticklabels(transition_labels, fontsize=9)
ax_c.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax_c.grid(True, axis='y', alpha=0.3)

# Panel D: Vote impact (absolute numbers)
ax_d = fig4.add_subplot(gs[1, 1])

# Calculate absolute number of defectors
dept_vote_impact = department_df[['departamento', 'ca_votes', 'ca_to_fa']].copy()
dept_vote_impact['ca_to_fa_votes'] = dept_vote_impact['ca_votes'] * dept_vote_impact['ca_to_fa']
dept_vote_impact_sorted = dept_vote_impact.sort_values('ca_to_fa_votes', ascending=True).tail(10)

y_pos_d = np.arange(len(dept_vote_impact_sorted))
votes = dept_vote_impact_sorted['ca_to_fa_votes'].values

colors_d = plt.cm.viridis(np.linspace(0.2, 0.9, len(dept_vote_impact_sorted)))

ax_d.barh(y_pos_d, votes, color=colors_d, edgecolor='black', linewidth=0.5)
ax_d.set_yticks(y_pos_d)
ax_d.set_yticklabels(dept_vote_impact_sorted['departamento'].values, fontsize=10)
ax_d.set_xlabel('Votos Estimados de Defección CA→FA', fontsize=11, fontweight='bold')
ax_d.set_title('D. Top 10 Departamentos por Impacto de Votos (CA→FA)', fontsize=12, fontweight='bold')
ax_d.grid(True, axis='x', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(votes):
    ax_d.text(v + 50, i, f'{v:,.0f}', va='center', fontsize=9)

# Main title
fig4.suptitle('Variación Geográfica en Defecciones de la Coalición',
             fontsize=15, fontweight='bold', y=0.995)

plt.savefig(figure_dir / 'geographic_variation_summary.png', dpi=300, bbox_inches='tight')
print(f"Guardado: outputs/figures/geographic_variation_summary.png")
plt.close()

print("\n" + "="*80)
print("TODAS LAS FIGURAS GENERADAS EXITOSAMENTE")
print("="*80)

# Print summary statistics
print("\nComparación Urbano vs Rural (Defección CA→FA):")
for idx, row in urban_rural_df.iterrows():
    print(f"  {row['stratum'].capitalize()}: {row['ca_to_fa']*100:.1f}% "
          f"(95% CI: {row['ca_to_fa_ci_lower']*100:.1f}%-{row['ca_to_fa_ci_upper']*100:.1f}%)")

print("\nComparación Regional (Defección CA→FA):")
for idx, row in region_df.iterrows():
    region_name = row['region'].replace('_', ' ')
    print(f"  {region_name}: {row['ca_to_fa_mean']*100:.1f}% "
          f"(IC 95%: {row['ca_to_fa_lower']*100:.1f}%-{row['ca_to_fa_upper']*100:.1f}%)")

print("\nRango Departamental (Defección CA→FA):")
print(f"  Más bajo: {department_df['departamento'].iloc[department_df['ca_to_fa'].argmin()]} "
      f"({department_df['ca_to_fa'].min()*100:.1f}%)")
print(f"  Más alto: {department_df['departamento'].iloc[department_df['ca_to_fa'].argmax()]} "
      f"({department_df['ca_to_fa'].max()*100:.1f}%)")
print(f"  Rango: {(department_df['ca_to_fa'].max() - department_df['ca_to_fa'].min())*100:.1f} puntos porcentuales")

print("\n" + "="*80)
print("Archivos de salida guardados en: outputs/figures/")
print("="*80)
