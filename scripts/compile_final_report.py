"""
Compilation validation script for LaTeX report.

Verifies all tables, figures, and references exist before compiling PDF.
Generates summary statistics for the report.
"""

import re
from pathlib import Path
from typing import List, Tuple


class ReportValidator:
    """Validator for LaTeX report compilation readiness."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.latex_dir = base_dir / "reports" / "latex"
        self.figures_dir = base_dir / "outputs" / "figures"
        self.tables_dir = base_dir / "outputs" / "tables" / "latex"

        self.errors = []
        self.warnings = []

    def check_figures(self) -> bool:
        """Verify all required figures exist."""
        print("\n=== Checking Figures ===")

        required_figures = [
            "urban_rural_ca_defection_forest.png",
            "region_coalition_defections.png",
            "department_ca_defection_map.png",
            "geographic_variation_summary.png",
            "all_coalition_parties_comparison.png",
            "coalition_defection_heatmap.png",
            "coalition_vote_impact.png",
        ]

        all_exist = True
        for fig in required_figures:
            path = self.figures_dir / fig
            if path.exists():
                size_kb = path.stat().st_size / 1024
                print(f"  [OK] {fig} ({size_kb:.1f} KB)")
            else:
                self.errors.append(f"Missing figure: {fig}")
                print(f"  [MISSING] {fig}")
                all_exist = False

        return all_exist

    def check_tables(self) -> bool:
        """Verify all required LaTeX tables exist."""
        print("\n=== Checking Tables ===")

        required_tables = [
            "national_transition_matrix.tex",
            "urban_rural_comparison_table.tex",
            "region_comparison_table.tex",
            "department_top10_table.tex",
            "department_pc_top10_table.tex",
            "department_pn_top10_table.tex",
            "mcmc_diagnostics_table.tex",
            "vote_impact_table.tex",
        ]

        all_exist = True
        for table in required_tables:
            path = self.tables_dir / table
            if path.exists():
                lines = path.read_text(encoding='utf-8').count('\n')
                print(f"  [OK] {table} ({lines} lines)")
            else:
                self.errors.append(f"Missing table: {table}")
                print(f"  [MISSING] {table}")
                all_exist = False

        return all_exist

    def check_main_tex(self) -> bool:
        """Verify main.tex structure."""
        print("\n=== Checking main.tex ===")

        main_path = self.latex_dir / "main.tex"
        if not main_path.exists():
            self.errors.append("main.tex not found")
            return False

        content = main_path.read_text(encoding='utf-8')

        # Check for section includes
        required_includes = [
            r'\input{sections/methodology}',
            r'\input{sections/results}',
            r'\input{sections/discussion}',
        ]

        all_present = True
        for include in required_includes:
            if include in content:
                print(f"  [OK] {include}")
            else:
                self.errors.append(f"Missing include: {include}")
                print(f"  [MISSING] {include}")
                all_present = False

        # Check bibliography
        if r'\bibliography{references}' in content:
            print(f"  [OK] Bibliography reference")
        else:
            self.warnings.append("Bibliography reference not found")
            print(f"  [WARNING] Bibliography reference not found")

        return all_present

    def check_references(self) -> Tuple[List[str], List[str]]:
        """Check for undefined references in LaTeX files."""
        print("\n=== Checking References ===")

        latex_files = list(self.latex_dir.glob("**/*.tex"))
        all_refs = set()
        all_labels = set()

        # Extract all \ref{} commands
        ref_pattern = re.compile(r'\\ref\{([^}]+)\}')
        label_pattern = re.compile(r'\\label\{([^}]+)\}')

        for file in latex_files:
            content = file.read_text(encoding='utf-8')
            all_refs.update(ref_pattern.findall(content))
            all_labels.update(label_pattern.findall(content))

        # Find undefined references
        undefined = all_refs - all_labels

        if undefined:
            print(f"  [WARNING] Found {len(undefined)} potentially undefined references:")
            for ref in sorted(undefined):
                print(f"    - {ref}")
                self.warnings.append(f"Potentially undefined reference: {ref}")
        else:
            print(f"  [OK] All {len(all_refs)} references appear to be defined")

        print(f"\n  Total labels: {len(all_labels)}")
        print(f"  Total references: {len(all_refs)}")

        return list(undefined), list(all_labels)

    def generate_statistics(self):
        """Generate report statistics."""
        print("\n=== Report Statistics ===")

        sections_dir = self.latex_dir / "sections"

        for section_file in ["methodology.tex", "results.tex", "discussion.tex"]:
            path = sections_dir / section_file
            if path.exists():
                content = path.read_text(encoding='utf-8')
                lines = content.count('\n')
                words = len(content.split())

                # Count party mentions
                ca_count = len(re.findall(r'\bCA\b|Cabildo Abierto', content, re.IGNORECASE))
                pc_count = len(re.findall(r'\bPC\b|Partido Colorado', content, re.IGNORECASE))
                pn_count = len(re.findall(r'\bPN\b|Partido Nacional', content, re.IGNORECASE))

                print(f"\n  {section_file}:")
                print(f"    Lines: {lines}")
                print(f"    Words: {words}")
                print(f"    CA mentions: {ca_count}")
                print(f"    PC mentions: {pc_count}")
                print(f"    PN mentions: {pn_count}")

    def validate(self) -> bool:
        """Run all validation checks."""
        print("=" * 60)
        print("LaTeX Report Validation")
        print("=" * 60)

        figures_ok = self.check_figures()
        tables_ok = self.check_tables()
        main_ok = self.check_main_tex()
        undefined, labels = self.check_references()

        self.generate_statistics()

        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        if self.errors:
            print(f"\n[ERROR] {len(self.errors)} errors found:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\n[WARNING] {len(self.warnings)} warnings found:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors:
            print("\n[SUCCESS] ALL CHECKS PASSED - Ready to compile!")
            print("\nTo compile, run:")
            print("  cd reports/latex")
            print("  pdflatex main.tex")
            print("  bibtex main")
            print("  pdflatex main.tex")
            print("  pdflatex main.tex")
            return True
        else:
            print(f"\n[FAILED] Fix {len(self.errors)} errors before compiling")
            return False


def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent
    validator = ReportValidator(base_dir)
    success = validator.validate()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
