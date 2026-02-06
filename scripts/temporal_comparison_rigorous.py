"""
Comparación Temporal Estadísticamente Rigurosa: 2019 vs 2024
============================================================

Análisis de cambios en transferencias de votos entre elecciones con:
- Propagación de errores para intervalos de confianza
- Tests de significancia con corrección de múltiples tests (Bonferroni)
- Análisis de heterogeneidad (Tau-squared, I-squared)
- Identificación de cambios significativos por departamento

Autor: Electoral Inference Analysis
Fecha: 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import logging
import pickle
import warnings
from typing import Tuple, Dict, List

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de rutas
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TemporalComparison:
    """Clase para realizar comparaciones temporales estadísticamente rigurosas."""

    def __init__(self, verbose: bool = True):
        """Inicializar el comparador temporal."""
        self.verbose = verbose
        self.data_2019 = None
        self.data_2024 = None
        self.departments = None
        self.comparisons = {}

    def load_data(self) -> None:
        """Cargar datos de transferencias para 2019 y 2024."""
        try:
            # Cargar datos 2024
            path_2024 = OUTPUT_DIR / "transfers_by_department.csv"
            self.data_2024 = pd.read_csv(path_2024, index_col=0)
            logger.info(f"✓ Datos 2024 cargados: {len(self.data_2024)} departamentos")

            # Cargar datos 2019
            path_2019 = OUTPUT_DIR / "transfers_by_department_2019.csv"
            self.data_2019 = pd.read_csv(path_2019, index_col=0)
            logger.info(f"✓ Datos 2019 cargados: {len(self.data_2019)} departamentos")

            # Obtener lista de departamentos comunes
            self._harmonize_departments()

        except FileNotFoundError as e:
            logger.error(f"✗ Error cargando datos: {e}")
            raise

    def _harmonize_departments(self) -> None:
        """Armonizar nombres y códigos de departamentos entre 2019 y 2024."""
        # Mapeo de códigos abreviados 2019 a nombres completos
        dept_mapping = {
            'AR': 'Artigas',
            'CA': 'Canelones',
            'CL': 'Cerro Largo',
            'CO': 'Colonia',
            'DU': 'Durazno',
            'FD': 'Flores',
            'FS': 'Florida',
            'LA': 'Lavalleja',
            'MA': 'Maldonado',
            'MO': 'Montevideo',
            'PA': 'Paysandú',
            'RN': 'Río Negro',
            'RO': 'Rocha',
            'RV': 'Rivera',
            'SA': 'Salto',
            'SJ': 'San José',
            'SO': 'Soriano',
            'TA': 'Tacuarembó',
            'TT': 'Treinta y Tres'
        }

        # Renombrar índice de 2019
        self.data_2019.index = self.data_2019.index.map(
            lambda x: dept_mapping.get(x, x)
        )

        # Obtener intersección de departamentos
        depts_2024 = set(self.data_2024.index)
        depts_2019 = set(self.data_2019.index)

        self.departments = sorted(list(depts_2024.intersection(depts_2019)))

        # Filtrar a departamentos comunes
        self.data_2024 = self.data_2024.loc[self.departments]
        self.data_2019 = self.data_2019.loc[self.departments]

        logger.info(f"✓ {len(self.departments)} departamentos armonizados")

    def _propagate_error(
        self,
        mean1: float,
        se1: float,
        mean2: float,
        se2: float
    ) -> Tuple[float, float, float, float]:
        """
        Propagar errores para diferencia entre dos estimaciones.

        Para Δ = X₂ - X₁:
        - E[Δ] = E[X₂] - E[X₁]
        - SE(Δ) = √(SE₁² + SE₂²)
        - CI 95%: [Δ ± 1.96*SE(Δ)]
        """
        delta = mean2 - mean1
        se_delta = np.sqrt(se1**2 + se2**2)

        # Intervalos de confianza 95%
        ci_lower = delta - 1.96 * se_delta
        ci_upper = delta + 1.96 * se_delta

        return delta, se_delta, ci_lower, ci_upper

    def _calculate_se_from_ci(self, point_est: float, ci_lower: float, ci_upper: float) -> float:
        """Calcular SE a partir de intervalo de confianza 95%."""
        # CI = point_est ± 1.96*SE → SE = (CI_upper - CI_lower) / (2*1.96)
        se = (ci_upper - ci_lower) / (2 * 1.96)
        return max(se, 1e-6)  # Evitar división por cero

    def compare_ca_to_fa(self) -> pd.DataFrame:
        """Comparar transferencias CA→FA entre 2019 y 2024."""
        logger.info("\n" + "="*70)
        logger.info("COMPARACIÓN CA → FA (2019 vs 2024)")
        logger.info("="*70)

        results = []

        for dept in self.departments:
            # Obtener datos 2024
            mean_2024 = self.data_2024.loc[dept, 'ca_to_fa']
            ci_lower_2024 = self.data_2024.loc[dept, 'ca_to_fa_lower']
            ci_upper_2024 = self.data_2024.loc[dept, 'ca_to_fa_upper']
            se_2024 = self._calculate_se_from_ci(mean_2024, ci_lower_2024, ci_upper_2024)

            # Obtener datos 2019
            mean_2019 = self.data_2019.loc[dept, 'ca_to_fa']
            ci_lower_2019 = self.data_2019.loc[dept, 'ca_to_fa_lower']
            ci_upper_2019 = self.data_2019.loc[dept, 'ca_to_fa_upper']
            se_2019 = self._calculate_se_from_ci(mean_2019, ci_lower_2019, ci_upper_2019)

            # Propagar errores
            delta, se_delta, ci_lower, ci_upper = self._propagate_error(
                mean_2019, se_2019, mean_2024, se_2024
            )

            # Test de significancia (¿CI incluye 0?)
            significant = not (ci_lower <= 0 <= ci_upper)

            # p-valor aproximado usando normal
            z_stat = abs(delta) / se_delta if se_delta > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(z_stat))

            results.append({
                'departamento': dept,
                'ca_to_fa_2019': mean_2019,
                'ca_to_fa_2024': mean_2024,
                'delta': delta,
                'delta_se': se_delta,
                'delta_ci_lower': ci_lower,
                'delta_ci_upper': ci_upper,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': significant
            })

        df = pd.DataFrame(results)
        self.comparisons['ca_to_fa'] = df

        return df

    def compare_pn_to_fa(self) -> pd.DataFrame:
        """Comparar transferencias PN→FA entre 2019 y 2024."""
        logger.info("\n" + "="*70)
        logger.info("COMPARACIÓN PN → FA (2019 vs 2024)")
        logger.info("="*70)

        results = []

        for dept in self.departments:
            # Obtener datos 2024
            mean_2024 = self.data_2024.loc[dept, 'pn_to_fa']
            ci_lower_2024 = self.data_2024.loc[dept, 'pn_to_fa_lower']
            ci_upper_2024 = self.data_2024.loc[dept, 'pn_to_fa_upper']
            se_2024 = self._calculate_se_from_ci(mean_2024, ci_lower_2024, ci_upper_2024)

            # Obtener datos 2019
            mean_2019 = self.data_2019.loc[dept, 'pn_to_fa']
            ci_lower_2019 = self.data_2019.loc[dept, 'pn_to_fa_lower']
            ci_upper_2019 = self.data_2019.loc[dept, 'pn_to_fa_upper']
            se_2019 = self._calculate_se_from_ci(mean_2019, ci_lower_2019, ci_upper_2019)

            # Propagar errores
            delta, se_delta, ci_lower, ci_upper = self._propagate_error(
                mean_2019, se_2019, mean_2024, se_2024
            )

            # Test de significancia
            significant = not (ci_lower <= 0 <= ci_upper)

            z_stat = abs(delta) / se_delta if se_delta > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(z_stat))

            results.append({
                'departamento': dept,
                'pn_to_fa_2019': mean_2019,
                'pn_to_fa_2024': mean_2024,
                'delta': delta,
                'delta_se': se_delta,
                'delta_ci_lower': ci_lower,
                'delta_ci_upper': ci_upper,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': significant
            })

        df = pd.DataFrame(results)
        self.comparisons['pn_to_fa'] = df

        return df

    def compare_pc_to_fa(self) -> pd.DataFrame:
        """Comparar transferencias PC→FA entre 2019 y 2024."""
        logger.info("\n" + "="*70)
        logger.info("COMPARACIÓN PC → FA (2019 vs 2024)")
        logger.info("="*70)

        results = []

        for dept in self.departments:
            # Obtener datos 2024
            mean_2024 = self.data_2024.loc[dept, 'pc_to_fa']
            ci_lower_2024 = self.data_2024.loc[dept, 'pc_to_fa_lower']
            ci_upper_2024 = self.data_2024.loc[dept, 'pc_to_fa_upper']
            se_2024 = self._calculate_se_from_ci(mean_2024, ci_lower_2024, ci_upper_2024)

            # Obtener datos 2019
            mean_2019 = self.data_2019.loc[dept, 'pc_to_fa']
            ci_lower_2019 = self.data_2019.loc[dept, 'pc_to_fa_lower']
            ci_upper_2019 = self.data_2019.loc[dept, 'pc_to_fa_upper']
            se_2019 = self._calculate_se_from_ci(mean_2019, ci_lower_2019, ci_upper_2019)

            # Propagar errores
            delta, se_delta, ci_lower, ci_upper = self._propagate_error(
                mean_2019, se_2019, mean_2024, se_2024
            )

            # Test de significancia
            significant = not (ci_lower <= 0 <= ci_upper)

            z_stat = abs(delta) / se_delta if se_delta > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(z_stat))

            results.append({
                'departamento': dept,
                'pc_to_fa_2019': mean_2019,
                'pc_to_fa_2024': mean_2024,
                'delta': delta,
                'delta_se': se_delta,
                'delta_ci_lower': ci_lower,
                'delta_ci_upper': ci_upper,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': significant
            })

        df = pd.DataFrame(results)
        self.comparisons['pc_to_fa'] = df

        return df

    def perform_multiple_comparison_correction(self) -> pd.DataFrame:
        """
        Aplicar corrección de Bonferroni para tests múltiples.

        Con 19 departamentos y 3 comparaciones → 57 tests
        α_ajustado = 0.05 / 57 = 0.000877
        """
        n_departments = len(self.departments)
        n_comparisons = len(self.comparisons)
        n_tests = n_departments * n_comparisons

        alpha_bonferroni = 0.05 / n_tests

        logger.info("\n" + "="*70)
        logger.info("CORRECCIÓN DE MÚLTIPLES COMPARACIONES (BONFERRONI)")
        logger.info("="*70)
        logger.info(f"Número de departamentos: {n_departments}")
        logger.info(f"Número de comparaciones (transferencias): {n_comparisons}")
        logger.info(f"Número total de tests: {n_tests}")
        logger.info(f"α original: 0.05 → α ajustado: {alpha_bonferroni:.6f}")

        significance_results = []

        for comparison_type, df in self.comparisons.items():
            for _, row in df.iterrows():
                p_val = row['p_value']
                bonferroni_sig = p_val < alpha_bonferroni

                # Clasificación: significativo, marginal (0.05), no significativo
                if bonferroni_sig:
                    classification = "SIGNIFICATIVO (Bonferroni)"
                elif p_val < 0.05:
                    classification = "MARGINAL (p < 0.05)"
                else:
                    classification = "NO SIGNIFICATIVO"

                significance_results.append({
                    'departamento': row['departamento'],
                    'comparison': comparison_type,
                    'delta': row['delta'],
                    'delta_ci_lower': row['delta_ci_lower'],
                    'delta_ci_upper': row['delta_ci_upper'],
                    'z_statistic': row['z_statistic'],
                    'p_value': p_val,
                    'bonferroni_alpha': alpha_bonferroni,
                    'classification': classification
                })

        significance_df = pd.DataFrame(significance_results)

        # Contar por clasificación
        logger.info("\nResumen de significancia (después de Bonferroni):")
        for comp in self.comparisons.keys():
            subset = significance_df[significance_df['comparison'] == comp]
            sig_count = (subset['classification'] == "SIGNIFICATIVO (Bonferroni)").sum()
            marg_count = (subset['classification'] == "MARGINAL (p < 0.05)").sum()
            nsig_count = (subset['classification'] == "NO SIGNIFICATIVO").sum()

            logger.info(f"\n  {comp}:")
            logger.info(f"    - Significativos (Bonferroni): {sig_count}/{n_departments}")
            logger.info(f"    - Marginales (p < 0.05): {marg_count}/{n_departments}")
            logger.info(f"    - No significativos: {nsig_count}/{n_departments}")

        return significance_df

    def calculate_heterogeneity(self) -> pd.DataFrame:
        """
        Calcular medidas de heterogeneidad tipo meta-análisis.

        Para cada transferencia, calcula:
        - Tau-squared (τ²): varianza entre departamentos del efecto
        - I-squared (I²): proporción de variación debida a heterogeneidad real
        - Q-statistic: test de heterogeneidad
        """
        logger.info("\n" + "="*70)
        logger.info("ANÁLISIS DE HETEROGENEIDAD (ESTILO META-ANÁLISIS)")
        logger.info("="*70)

        heterogeneity_results = []

        for comparison_type, df in self.comparisons.items():
            deltas = df['delta'].values
            ses = df['delta_se'].values

            # Pesos (inverso de varianza)
            weights = 1 / (ses ** 2)

            # Media ponderada
            mu = np.sum(weights * deltas) / np.sum(weights)

            # Q-statistic: suma ponderada de desviaciones al cuadrado
            q_stat = np.sum(weights * (deltas - mu) ** 2)

            # Grados de libertad
            df_q = len(deltas) - 1

            # p-valor del Q-test
            p_q = 1 - stats.chi2.cdf(q_stat, df_q)

            # Tau-squared (estimador DerSimonian-Laird)
            sum_weights = np.sum(weights)
            sum_weights_sq = np.sum(weights ** 2)

            tau_sq_numerator = q_stat - df_q
            tau_sq_denominator = sum_weights - (sum_weights_sq / sum_weights)

            tau_sq = max(0, tau_sq_numerator / tau_sq_denominator) if tau_sq_denominator > 0 else 0

            # I-squared: % de variación debida a heterogeneidad
            # I² = 100% * τ² / (τ² + mediana(σᵢ²))
            se_sq_median = np.median(ses ** 2)
            i_squared = 100 * tau_sq / (tau_sq + se_sq_median) if (tau_sq + se_sq_median) > 0 else 0

            heterogeneity_results.append({
                'comparison': comparison_type,
                'n_studies': len(deltas),
                'mu_pooled': mu,
                'tau_squared': tau_sq,
                'tau': np.sqrt(tau_sq),
                'i_squared': i_squared,
                'q_statistic': q_stat,
                'q_df': df_q,
                'q_p_value': p_q,
                'heterogeneity_level': self._classify_heterogeneity(i_squared)
            })

            logger.info(f"\n{comparison_type}:")
            logger.info(f"  N estudios (departamentos): {len(deltas)}")
            logger.info(f"  Media global (ponderada): {mu:.4f}")
            logger.info(f"  τ² (varianza entre-estudios): {tau_sq:.6f}")
            logger.info(f"  τ (SD entre-estudios): {np.sqrt(tau_sq):.4f}")
            logger.info(f"  I²: {i_squared:.2f}% (heterogeneidad real vs ruido)")
            logger.info(f"  Q-estadístico: {q_stat:.3f} (df={df_q}, p={p_q:.4f})")
            logger.info(f"  Nivel: {self._classify_heterogeneity(i_squared)}")

        return pd.DataFrame(heterogeneity_results)

    @staticmethod
    def _classify_heterogeneity(i_squared: float) -> str:
        """Clasificar nivel de heterogeneidad según Higgins & Thompson."""
        if i_squared < 25:
            return "Baja"
        elif i_squared < 50:
            return "Moderada"
        elif i_squared < 75:
            return "Alta"
        else:
            return "Muy Alta"

    def rank_changes(self) -> Dict[str, pd.DataFrame]:
        """Ranking de cambios por magnitud en cada comparación."""
        logger.info("\n" + "="*70)
        logger.info("RANKING DE CAMBIOS POR MAGNITUD")
        logger.info("="*70)

        ranking_results = {}

        for comparison_type, df in self.comparisons.items():
            # Ordenar por valor absoluto del cambio
            ranked = df.copy()
            ranked['abs_delta'] = ranked['delta'].abs()
            ranked = ranked.sort_values('abs_delta', ascending=False)

            # Añadir ranking
            ranked['rank'] = range(1, len(ranked) + 1)

            # Identificar outliers (cambios > 2 DE del cambio medio)
            mean_delta = ranked['delta'].mean()
            std_delta = ranked['delta'].std()
            ranked['is_outlier'] = (ranked['delta'].abs() - mean_delta) > (2 * std_delta)

            ranking_results[comparison_type] = ranked[[
                'rank', 'departamento', 'delta', 'delta_se',
                'delta_ci_lower', 'delta_ci_upper', 'p_value', 'is_outlier'
            ]]

            logger.info(f"\n{comparison_type} (Top 5):")
            for _, row in ranked.head(5).iterrows():
                outlier_marker = " ⚠️ OUTLIER" if row['is_outlier'] else ""
                logger.info(f"  {row['rank']}. {row['departamento']}: Δ={row['delta']:+.4f} "
                           f"({row['delta_ci_lower']:.4f}, {row['delta_ci_upper']:.4f}){outlier_marker}")

        return ranking_results

    def save_results(self) -> None:
        """Guardar todos los resultados en archivos CSV."""
        logger.info("\n" + "="*70)
        logger.info("GUARDANDO RESULTADOS")
        logger.info("="*70)

        # 1. Diferencias temporales detalladas
        all_differences = []
        for comp_type, df in self.comparisons.items():
            df_copy = df.copy()
            df_copy['comparison'] = comp_type
            all_differences.append(df_copy)

        differences_df = pd.concat(all_differences, ignore_index=True)
        path_differences = OUTPUT_DIR / "temporal_differences_rigorous.csv"
        differences_df.to_csv(path_differences, index=False)
        logger.info(f"✓ {path_differences}")

        # 2. Tests de significancia
        significance_df = self.perform_multiple_comparison_correction()
        path_significance = OUTPUT_DIR / "significance_tests.csv"
        significance_df.to_csv(path_significance, index=False)
        logger.info(f"✓ {path_significance}")

        # 3. Heterogeneidad
        heterogeneity_df = self.calculate_heterogeneity()
        path_heterogeneity = OUTPUT_DIR / "heterogeneity_analysis.csv"
        heterogeneity_df.to_csv(path_heterogeneity, index=False)
        logger.info(f"✓ {path_heterogeneity}")

        # 4. Rankings
        ranking_dfs = self.rank_changes()
        for comp_type, ranked_df in ranking_dfs.items():
            path_ranking = OUTPUT_DIR / f"ranking_changes_{comp_type}.csv"
            ranked_df.to_csv(path_ranking, index=False)
            logger.info(f"✓ {path_ranking}")


def print_summary_statistics():
    """Imprimir resumen de estadísticas clave."""
    logger.info("\n" + "="*70)
    logger.info("RESUMEN EJECUTIVO")
    logger.info("="*70)

    # Cargar resultados
    path_diff = OUTPUT_DIR / "temporal_differences_rigorous.csv"
    path_sig = OUTPUT_DIR / "significance_tests.csv"
    path_het = OUTPUT_DIR / "heterogeneity_analysis.csv"

    if path_diff.exists():
        diff_df = pd.read_csv(path_diff)
        logger.info(f"\nDiferencias detectadas (todos los departamentos):")
        logger.info(f"  - Total de comparaciones: {len(diff_df)}")

        for comp in diff_df['comparison'].unique():
            subset = diff_df[diff_df['comparison'] == comp]
            avg_change = subset['delta'].mean()
            min_change = subset['delta'].min()
            max_change = subset['delta'].max()

            logger.info(f"\n  {comp}:")
            logger.info(f"    - Cambio promedio: {avg_change:+.4f}")
            logger.info(f"    - Rango: {min_change:+.4f} a {max_change:+.4f}")
            logger.info(f"    - Desviación estándar: {subset['delta'].std():.4f}")

    if path_sig.exists():
        sig_df = pd.read_csv(path_sig)
        logger.info(f"\nSignificancia (después de corrección Bonferroni):")
        for comp in sig_df['comparison'].unique():
            subset = sig_df[sig_df['comparison'] == comp]
            sig_count = (subset['classification'] == "SIGNIFICATIVO (Bonferroni)").sum()
            marg_count = (subset['classification'] == "MARGINAL (p < 0.05)").sum()

            logger.info(f"  {comp}:")
            logger.info(f"    - Significativos: {sig_count}")
            logger.info(f"    - Marginales: {marg_count}")

    if path_het.exists():
        het_df = pd.read_csv(path_het)
        logger.info(f"\nHeterogeneidad entre departamentos:")
        for _, row in het_df.iterrows():
            logger.info(f"  {row['comparison']}:")
            logger.info(f"    - I²: {row['i_squared']:.2f}% ({row['heterogeneity_level']})")
            logger.info(f"    - τ²: {row['tau_squared']:.6f}")


def main():
    """Ejecutar análisis temporal completo."""
    logger.info("\n" + "="*70)
    logger.info("COMPARACIÓN TEMPORAL 2019 vs 2024")
    logger.info("Análisis estadísticamente riguroso")
    logger.info("="*70)

    # Crear comparador
    comparator = TemporalComparison(verbose=True)

    # Cargar datos
    try:
        comparator.load_data()
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        return 1

    # Realizar comparaciones
    logger.info("\n" + "-"*70)
    logger.info("FASE 1: Comparaciones de transferencias")
    logger.info("-"*70)

    df_ca_fa = comparator.compare_ca_to_fa()
    df_pn_fa = comparator.compare_pn_to_fa()
    df_pc_fa = comparator.compare_pc_to_fa()

    # Tests de significancia con corrección múltiple
    logger.info("\n" + "-"*70)
    logger.info("FASE 2: Tests de significancia (Bonferroni)")
    logger.info("-"*70)

    comparator.perform_multiple_comparison_correction()

    # Análisis de heterogeneidad
    logger.info("\n" + "-"*70)
    logger.info("FASE 3: Análisis de heterogeneidad")
    logger.info("-"*70)

    comparator.calculate_heterogeneity()

    # Ranking de cambios
    logger.info("\n" + "-"*70)
    logger.info("FASE 4: Ranking de cambios")
    logger.info("-"*70)

    comparator.rank_changes()

    # Guardar resultados
    logger.info("\n" + "-"*70)
    logger.info("FASE 5: Guardando resultados")
    logger.info("-"*70)

    comparator.save_results()

    # Imprimir resumen
    print_summary_statistics()

    logger.info("\n" + "="*70)
    logger.info("✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
    logger.info("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
