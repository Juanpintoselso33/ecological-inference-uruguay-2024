# Inferencia Ecologica de las Elecciones Uruguayas 2024

Analisis de transferencia de votos entre la primera vuelta (27 octubre) y el ballotage (24 noviembre) de las elecciones presidenciales uruguayas de 2024, a nivel de circuito electoral, usando metodos bayesianos de inferencia ecologica.

## Descripcion

Este proyecto implementa metodos estadisticos de inferencia ecologica para estimar como los votantes de cada partido en la primera vuelta distribuyeron sus votos en el ballotage. El analisis incluye:

- **King's Ecological Inference**: Metodo bayesiano implementado con PyMC v5 (4000 samples, 4 chains)
- **Goodman Ecological Regression**: Metodo baseline OLS para validacion cruzada
- **Analisis temporal**: Comparacion sistematica con elecciones 2019
- **Estratificacion**: Analisis departamental (19 deptos), urbano/rural, metropolitano/interior
- **Analisis de sensibilidad**: Variacion de priors, remocion de outliers, bootstrap
- **Posterior Predictive Checks**: Validacion del ajuste del modelo
- **Patrones de voto en blanco**: Hipotesis de protesta vs desmovilizacion

## Resultados Principales

**Nacional 2024** (modelo con PI):
- CA defeccion a FA: **46.5%** [95% CI: 43.0%, 49.9%]
- CA retencion por PN: **49.1%** [95% CI: 45.6%, 52.7%]
- PC lealtad a coalicion: **87.9%** [95% CI: 86.9%, 88.8%]
- PN lealtad: **91.8%** [95% CI: 91.2%, 92.3%]

**Variacion Geografica**:
- Mayor defeccion CA: Rio Negro (95.7%), Treinta y Tres (70.2%)
- Menor defeccion CA: Tacuarembo (5.3%), Cerro Largo (14.8%)
- Circuitos rurales: 60.8% defeccion vs 39.0% urbanos

**Comparacion 2019-2024**:
- Paradoja: coalicion mejoro cohesion pero perdio por colapso de CA (-78% votos)
- PC lealtad mejoro: 75.9% (2019) -> 87.9% (2024)
- Nula estabilidad temporal geografica (r < 0.13)

## Estructura del Proyecto

```
.
├── data/
│   ├── raw/                    # Datos originales de Corte Electoral
│   ├── processed/              # Datos limpios (7,271 circuitos 2024, 7,213 en 2019)
│   └── external/shapefiles/    # Mapas de circuitos electorales
├── src/
│   ├── data/                   # Descarga y procesamiento de datos
│   ├── models/                 # Modelos EI (King bayesiano, Goodman OLS)
│   ├── visualization/          # Generacion de graficos y mapas
│   └── utils/                  # Utilidades (config, logging, validacion)
├── scripts/                    # Scripts ejecutables (analisis, visualizacion)
├── tests/                      # Tests unitarios
├── outputs/
│   ├── figures/                # Visualizaciones @ 300 DPI (PNG + PDF)
│   └── tables/                 # CSV + LaTeX tables
├── reports/
│   └── latex/                  # Fuentes LaTeX del informe (100 paginas)
└── config.yaml                 # Configuracion centralizada del proyecto
```

## Instalacion

### Requisitos

- Python 3.13+
- PyMC v5 (inferencia bayesiana)
- pandas, numpy, scipy (procesamiento)
- matplotlib, seaborn (visualizacion)
- geopandas (analisis geoespacial)
- arviz (diagnosticos MCMC)

```bash
conda activate ds
python -c "import pymc; import pandas; import arviz; print('OK')"
```

## Uso

### Pipeline completo desde cero

```bash
# 1. Descargar datos de Corte Electoral
python scripts/download_data.py

# 2. Procesar y limpiar datos
python scripts/clean_data.py

# 3. Fusionar primera vuelta + ballotage
python scripts/merge_data.py

# 4. Analisis nacional (King's EI con PI)
python scripts/national_analysis_with_pi.py --samples 4000 --chains 4

# 5. Analisis departamental
python scripts/analyze_by_department.py

# 6. Comparacion temporal 2019 vs 2024
python scripts/compare_2019_2024.py

# 7. Generar visualizaciones (48+ figuras @ 300 DPI)
python scripts/generate_all_figures.py

# 8. Generar tablas LaTeX
python scripts/generate_latex_tables.py

# 9. Compilar informe PDF
cd reports/latex && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Analisis adicionales

```bash
# Comparacion Goodman vs King's EI
python scripts/compare_goodman_vs_king.py

# Analisis de sensibilidad (priors, outliers, bootstrap)
python scripts/sensitivity_analysis.py

# Patrones de voto en blanco
python scripts/analyze_blank_votes.py

# Posterior Predictive Checks
python scripts/posterior_predictive_checks.py

# Dinamica de participacion
python scripts/analyze_turnout_dynamics.py

# Escenarios 2026
python scripts/project_2026_scenarios.py
```

## Fuentes de Datos

- **Primera vuelta 2024**: [Corte Electoral](https://www.gub.uy/corte-electoral/datos-y-estadisticas/estadisticas/resultados-elecciones-nacionales-del-2024)
- **Ballotage 2024**: [Corte Electoral](https://www.gub.uy/corte-electoral/datos-y-estadisticas/estadisticas/resultados-elecciones-nacionales-del-2024)
- **Datos 2019**: Archivo historico Corte Electoral
- **Shapefiles**: IDE Uruguay / OpenData Uruguay

## Metodologia

### King's Ecological Inference (Bayesiano)

Modelo que estima la matriz de transferencia de votos respetando bounds naturales:
- Prior: Dirichlet uniforme sobre probabilidades de transferencia
- Likelihood: Multinomial por circuito
- Inferencia: MCMC via PyMC v5 (NUTS sampler)
- Diagnosticos: R-hat < 1.01, ESS > 1000

### Matriz de Transferencia

```
                  Ballotage (Nov 2024)
                FA      PN    Blanco/Nulo
Primera   FA    98.9%    0.5%     0.6%
Vuelta    PN     6.3%   91.8%     1.9%
(Oct)     PC    10.0%   87.9%     2.1%
          CA    46.5%   49.1%     4.4%
          PI     ...     ...      ...
```

### Validaciones

- Suma de votos = total reportado por Corte Electoral
- Proporciones en [0, 1], filas suman 1.0
- Convergencia MCMC (R-hat < 1.01, ESS > 1000)
- Analisis de sensibilidad: CV < 10% entre variantes
- Posterior Predictive Checks

## Informe

El informe completo (100 paginas) se compila desde `reports/latex/main.tex` e incluye:

1. Introduccion y contexto politico
2. Metodologia (King's EI, Goodman, datos)
3. Resultados nacionales 2024
4. Resultados departamentales
5. Comparacion temporal 2019-2024
6. Analisis de sensibilidad y robustez
7. Patrones de voto en blanco
8. Discusion y conclusiones
9. Apendice: Posterior Predictive Checks
10. Apendice: Covariables demograficas (exploratorio)

## Referencias

- King, G. (1997). *A Solution to the Ecological Inference Problem*. Princeton University Press.
- Rosen, O., Jiang, W., King, G. & Tanner, M. A. (2001). "Bayesian and Frequentist Inference for Ecological Inference"
- Gonzalez, L. E. (1991). *Political Structures and Democracy in Uruguay*. University of Notre Dame Press.
- Bottinelli, O. (2021). "Las Elecciones Uruguayas de 2019"
- [PyMC v5](https://www.pymc.io/) | [ArviZ](https://arviz-devs.github.io/arviz/)

## Licencia

Este proyecto es de codigo abierto para fines academicos y de investigacion.
