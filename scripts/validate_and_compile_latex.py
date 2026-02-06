"""
Script de validación y compilación LaTeX con mejoras 2024-2025.

Valida que todas las mejoras modernas estén aplicadas y compila el documento.
"""

import re
import subprocess
from pathlib import Path
from typing import List, Tuple


class LatexValidator:
    """Validador de mejoras LaTeX modernas."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.latex_dir = base_dir / "reports" / "latex"
        self.main_tex = self.latex_dir / "main.tex"

        self.checks_passed = []
        self.checks_failed = []

    def check_modern_packages(self) -> bool:
        """Verifica que los paquetes modernos estén presentes."""
        print("\n=== Verificando Paquetes Modernos ===")

        content = self.main_tex.read_text(encoding='utf-8')

        required_packages = {
            'microtype': 'Mejora micro-tipografía automáticamente',
            'placeins': 'Evita drift de floats al apéndice',
            'subcaption': 'Moderno para subfiguras',
            'siunitx': 'Alineación decimal profesional',
            'setspace': 'Espaciado de líneas académico',
        }

        all_present = True
        for package, description in required_packages.items():
            pattern = rf'\\usepackage(\[.*?\])?\{{{package}\}}'
            if re.search(pattern, content):
                print(f"  [OK] {package} - {description}")
                self.checks_passed.append(f"Paquete {package} presente")
            else:
                print(f"  [FALTA] {package} - {description}")
                self.checks_failed.append(f"Paquete {package} faltante")
                all_present = False

        return all_present

    def check_float_configuration(self) -> bool:
        """Verifica configuración de floats."""
        print("\n=== Verificando Configuración de Floats ===")

        content = self.main_tex.read_text(encoding='utf-8')

        required_settings = [
            (r'\\renewcommand\{\\topfraction\}', 'topfraction configurado'),
            (r'\\renewcommand\{\\textfraction\}', 'textfraction configurado'),
            (r'\\renewcommand\{\\floatpagefraction\}', 'floatpagefraction configurado'),
            (r'\\FloatBarrier', 'FloatBarrier usado antes de bibliografía'),
        ]

        all_present = True
        for pattern, description in required_settings:
            if re.search(pattern, content):
                print(f"  [OK] {description}")
                self.checks_passed.append(description)
            else:
                print(f"  [FALTA] {description}")
                self.checks_failed.append(description)
                all_present = False

        return all_present

    def check_paragraph_configuration(self) -> bool:
        """Verifica configuración de párrafos."""
        print("\n=== Verificando Configuración de Párrafos ===")

        content = self.main_tex.read_text(encoding='utf-8')

        has_parindent = r'\\setlength\{\\parindent\}' in content
        has_parskip = r'\\setlength\{\\parskip\}' in content or r'\\usepackage\{parskip\}' in content

        if has_parindent:
            # Extraer valor
            match = re.search(r'\\setlength\{\\parindent\}\{(\d+)pt\}', content)
            if match:
                value = int(match.group(1))
                print(f"  [OK] Sangría configurada: {value}pt (estándar académico: 15pt)")
                self.checks_passed.append(f"Sangría académica: {value}pt")
            else:
                print(f"  [OK] Sangría configurada")
                self.checks_passed.append("Sangría configurada")

        if has_parskip:
            print(f"  [OK] Espaciado entre párrafos configurado")
            self.checks_passed.append("Espaciado entre párrafos configurado")

        if not (has_parindent or has_parskip):
            print(f"  [ADVERTENCIA] Sin configuración explícita de párrafos")
            self.checks_failed.append("Sin configuración de párrafos")
            return False

        return True

    def check_widows_orphans(self) -> bool:
        """Verifica prevención de widows/orphans."""
        print("\n=== Verificando Prevención Widows/Orphans ===")

        content = self.main_tex.read_text(encoding='utf-8')

        required_penalties = [
            ('widowpenalty', 'Prevención de widows'),
            ('clubpenalty', 'Prevención de orphans'),
            ('displaywidowpenalty', 'Prevención en ecuaciones'),
        ]

        all_present = True
        for penalty, description in required_penalties:
            if f'\\{penalty}=' in content:
                # Extraer valor
                match = re.search(rf'\\{penalty}=(\d+)', content)
                if match:
                    value = int(match.group(1))
                    print(f"  [OK] {description}: {value}")
                    self.checks_passed.append(f"{description}: {value}")
            else:
                print(f"  [FALTA] {description}")
                self.checks_failed.append(description)
                all_present = False

        # Verificar raggedbottom
        if r'\raggedbottom' in content:
            print(f"  [OK] \\raggedbottom configurado")
            self.checks_passed.append("raggedbottom configurado")
        else:
            print(f"  [ADVERTENCIA] Sin \\raggedbottom (páginas pueden quedar estiradas)")

        return all_present

    def check_spanish_metadata(self) -> bool:
        """Verifica que metadatos estén en español."""
        print("\n=== Verificando Idioma en Metadatos ===")

        content = self.main_tex.read_text(encoding='utf-8')

        # Buscar pdfauthor y pdftitle
        pdfauthor_match = re.search(r'pdfauthor=\{([^}]+)\}', content)
        pdftitle_match = re.search(r'pdftitle=\{([^}]+)\}', content)

        spanish_issues = []

        if pdfauthor_match:
            author = pdfauthor_match.group(1)
            # Verificar si tiene caracteres típicamente ingleses
            if any(word in author.lower() for word in ['electoral analysis project', 'project']):
                spanish_issues.append(f"pdfauthor en inglés: {author}")
                print(f"  [ADVERTENCIA] pdfauthor puede estar en inglés: {author}")
            else:
                print(f"  [OK] pdfauthor en español: {author}")
                self.checks_passed.append("pdfauthor en español")

        if pdftitle_match:
            title = pdftitle_match.group(1)
            if any(word in title.lower() for word in ['analysis', 'elections', 'ecological', 'inference']):
                spanish_issues.append(f"pdftitle en inglés: {title}")
                print(f"  [ADVERTENCIA] pdftitle puede estar en inglés: {title}")
            else:
                print(f"  [OK] pdftitle en español: {title}")
                self.checks_passed.append("pdftitle en español")

        return len(spanish_issues) == 0

    def generate_report(self):
        """Genera reporte final."""
        print("\n" + "=" * 60)
        print("RESUMEN DE VALIDACIÓN")
        print("=" * 60)

        print(f"\n[ÉXITOS] {len(self.checks_passed)} verificaciones pasadas:")
        for check in self.checks_passed[:10]:  # Mostrar primeros 10
            print(f"  + {check}")
        if len(self.checks_passed) > 10:
            print(f"  ... y {len(self.checks_passed) - 10} más")

        if self.checks_failed:
            print(f"\n[ADVERTENCIAS] {len(self.checks_failed)} items a revisar:")
            for check in self.checks_failed:
                print(f"  - {check}")

        if not self.checks_failed:
            print("\n[SUCCESS] Todas las mejoras modernas están aplicadas!")
            print("\nEl documento está listo para compilación profesional.")
        else:
            print(f"\n[ADVERTENCIA] Hay {len(self.checks_failed)} items pendientes.")

    def validate(self) -> bool:
        """Ejecuta todas las validaciones."""
        print("=" * 60)
        print("Validación de Mejoras LaTeX Modernas")
        print("=" * 60)

        packages_ok = self.check_modern_packages()
        floats_ok = self.check_float_configuration()
        paragraphs_ok = self.check_paragraph_configuration()
        widows_ok = self.check_widows_orphans()
        spanish_ok = self.check_spanish_metadata()

        self.generate_report()

        return packages_ok and floats_ok and paragraphs_ok and widows_ok


def main():
    """Función principal."""
    base_dir = Path(__file__).parent.parent
    validator = LatexValidator(base_dir)

    success = validator.validate()

    if success:
        print("\n" + "=" * 60)
        print("LISTO PARA COMPILAR")
        print("=" * 60)
        print("\nPara compilar el PDF actualizado:")
        print("  cd reports/latex")
        print("  pdflatex main.tex")
        print("  bibtex main")
        print("  pdflatex main.tex")
        print("  pdflatex main.tex")
        print("\nO usa el script: compile.bat")
        return 0
    else:
        print("\n[INFO] El documento se puede compilar, pero hay items pendientes de optimización.")
        return 0


if __name__ == "__main__":
    exit(main())
