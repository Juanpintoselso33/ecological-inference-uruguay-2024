"""
MASTER ORCHESTRATOR - Run all remaining analyses in optimal parallel order

This script executes the full Phase A-C implementation plan in parallel.
"""

import sys
from pathlib import Path
import subprocess
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParallelExecutor:
    """Execute analyses in parallel batches."""

    def __init__(self):
        self.processes = []
        self.results = []

    def run_script(self, script_name, args="", background=False):
        """Run a Python script."""
        cmd = f"python scripts/{script_name} {args}"

        if background:
            logger.info(f"  Iniciando en background: {script_name}")
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append({
                'name': script_name,
                'process': process,
                'start_time': datetime.now()
            })
        else:
            logger.info(f"  Ejecutando: {script_name}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            self.results.append({
                'name': script_name,
                'returncode': result.returncode,
                'stdout': result.stdout[:500],
                'stderr': result.stderr[:500]
            })
            return result.returncode == 0

    def wait_for_completion(self, timeout=None):
        """Wait for all background processes to complete."""
        logger.info(f"Esperando {len(self.processes)} procesos...")

        for proc_info in self.processes:
            try:
                stdout, stderr = proc_info['process'].communicate(timeout=timeout)
                elapsed = (datetime.now() - proc_info['start_time']).total_seconds()

                self.results.append({
                    'name': proc_info['name'],
                    'returncode': proc_info['process'].returncode,
                    'elapsed_time': elapsed,
                    'stdout': stdout[:500] if stdout else "",
                    'stderr': stderr[:500] if stderr else ""
                })

                status = "OK" if proc_info['process'].returncode == 0 else "FAILED"
                logger.info(f"  {proc_info['name']}: {status} ({elapsed:.0f}s)")

            except subprocess.TimeoutExpired:
                proc_info['process'].kill()
                logger.error(f"  {proc_info['name']}: TIMEOUT")

        self.processes = []


def main():
    config = get_config()

    print("="*80)
    print(" " * 20 + "MASTER PARALLEL ORCHESTRATOR")
    print("=" * 80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    executor = ParallelExecutor()

    # =================================================================
    # BATCH 1: Already completed
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("BATCH 1: YA COMPLETADO")
    logger.info("="*70)
    logger.info("  - FA National MCMC")
    logger.info("  - Turnout Dynamics")
    logger.info("  - Goodman vs King")
    logger.info("  - Blank Votes")
    logger.info("  - Turnout Figures")
    logger.info("  - Goodman Figures")

    # =================================================================
    # BATCH 2: Medium-length MCMC analyses (2-4 hours each)
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("BATCH 2: ANÁLISIS MCMC MEDIOS (Iniciando en paralelo)")
    logger.info("="*70)

    # FA Departmental (19 departments × ~6-10 min each = ~2-3 hours)
    executor.run_script("analyze_fa_departmental.py", "--samples 4000 --chains 4", background=True)

    # FA Temporal 2019 vs 2024 (2 years × ~3 hours = ~6 hours, but can run in parallel)
    executor.run_script("compare_fa_2019_2024.py", "--samples 4000 --chains 4", background=True)

    # FA Stratified (4 strata × ~1 hour = ~4 hours)
    executor.run_script("analyze_fa_stratified.py", "--samples 4000 --chains 4", background=True)

    logger.info("  Esperando completación de BATCH 2 (esto tomará ~4-6 horas)...")
    logger.info("  Puede detener este script y ejecutar batch 3 manualmente más tarde")

    # Wait for these to complete (with 8-hour timeout)
    executor.wait_for_completion(timeout=8*3600)

    # =================================================================
    # BATCH 3: Quick analyses and visualizations
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("BATCH 3: ANÁLISIS RÁPIDOS Y VISUALIZACIONES")
    logger.info("="*70)

    # Posterior Predictive Checks
    executor.run_script("posterior_predictive_checks.py")

    # Generate all remaining figures
    # Note: FA figures script will be created separately
    logger.info("  Generación de figuras FA (crear script generate_fa_figures.py)")

    # =================================================================
    # SUMMARY
    # =================================================================
    logger.info("\n" + "="*70)
    logger.info("RESUMEN DE EJECUCIÓN")
    logger.info("="*70)

    success_count = sum(1 for r in executor.results if r['returncode'] == 0)
    total_count = len(executor.results)

    logger.info(f"\nCompletados: {success_count}/{total_count}")

    for result in executor.results:
        status = "✓" if result['returncode'] == 0 else "✗"
        elapsed = result.get('elapsed_time', 0)
        logger.info(f"  {status} {result['name']:<40s} ({elapsed:.0f}s)")

    print("\n" + "="*80)
    print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
