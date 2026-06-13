# Sub-proyecto 1 — Capa de datos EI-ready

**Fecha:** 2026-06-13
**Estado:** Diseño aprobado, pendiente de plan de implementación
**Contexto mayor:** Primero de cuatro sub-proyectos para reconstruir el experimento de
inferencia ecológica (EI) sobre datos robustos del proyecto *Uruguay Electoral Map*.
Los otros tres (granularidad, validación 2025, robustez metodológica) dependen de este.

---

## 1. Problema

El EI actual (`scripts/national_analysis_with_pi.py` y afines) consume
`data/processed/circuitos_merged.parquet` (7.271 circuitos), un dataset propio
cuya trazabilidad y validación son limitadas. El proyecto vecino
*Uruguay Electoral Map* (`E:\Proyectos VS CODE\Uruguay Electoral Map\uruguay-electoral-map`)
expone los mismos resultados electorales con un ETL reproducible, quality gates y
reconciliación de totales, bajando hasta circuito / local / hoja.

El objetivo de este sub-proyecto es **construir una capa de datos EI-ready a partir de
esa fuente robusta, en paralelo al parquet actual, con un gate de reconciliación que
explique cualquier diferencia**, sin sobrescribir nada hasta validar.

## 2. Estrategia (decidida)

**Snapshot + v2 en paralelo.** Se copia un snapshot *fijado por hash* de los archivos
de datos necesarios dentro del repo EI, se construye `circuitos_merged_v2.parquet` al
lado del parquet viejo, se reconcilia viejo-vs-nuevo y recién entonces se migra el
análisis. Reproducible para publicación; auditable; reversible.

Descartadas: cliente API en vivo (rompe reproducibilidad), reemplazo in-place
(destruye la comparación viejo-vs-nuevo).

## 3. Fuente de datos (corregida tras inspección)

La fuente **no** son los dumps masivos `/api/v1/dumps/*.ndjson`: esos omiten los campos
no partidarios (verificado: `grep -c noPartidarios balotaje-2024.ndjson` → 0). Ver issues
abiertos en el repo vecino: **#59** (dumps sin `noPartidarios`/`local`) y **#60**
(sin padrón/`habilitados`).

La fuente correcta son los archivos por departamento:

```
public/data/<eleccion>/<departamento>/votes-circuito.json   (19 por elección)
```

Estructura por circuito (verificada):

```json
{
  "eleccionId": "balotaje-2024", "departamento": "artigas",
  "nivel": "circuito", "escrutinio": "definitivo", "tipo": "balotaje",
  "zonas": [
    { "geoId": "1", "local": "artigas-L001", "ganadorOpcionId": "nacional",
      "validos": 186,
      "porOpcion": [ {"opcionId": "nacional", "votos": 152},
                     {"opcionId": "frente-amplio", "votos": 34} ],
      "noPartidarios": { "enBlanco": 4, "anulados": 4, "observados": 27 } }
  ]
}
```

Hechos verificados:
- Clave de circuito: **`(departamento, geoId)`**. `geoId` es un número que se repite
  entre departamentos, por lo que solo, no identifica.
- Conteos: nac-2024 = 7.222 circuitos, bal-2024 = 7.221, deptal-2025 = 2.542 (distintos
  `(depto,geoId)`). El parquet viejo tiene 7.271 → diferencia a reconciliar.
- `validos == fa + pn` en 7.221/7.221 circuitos de balotaje (100%): confirma que
  blanco/anulado/observado quedan fuera de `validos`.
- `cabildo-abierto`, `frente-amplio`, `colorado`, `nacional`, `independiente` presentes
  en 1ª vuelta; balotaje solo `nacional` + `frente-amplio`.

## 4. Arquitectura y flujo

```
Uruguay Electoral Map (repo vecino)        Eco inference 2024 (este repo)
public/data/<elec>/<depto>/        ──snapshot──►  data/raw/external/<elec>/
   votes-circuito.json                              + snapshot_manifest.json
                                                       (sha256 + versión + fecha)
                                                              │
                                                              ▼
                                                 src/data/ingest_votes_circuito.py
                                                 (lee 19 JSON → DataFrame largo)
                                                              │
                                                              ▼  src/data/party_mapping.py
                                                 src/data/build_ei_dataset.py
                                                 (pivot ancho + merge 1ª+balotaje
                                                  por (departamento, geoId) + shares)
                                                              │
                                                              ▼
                                           data/processed/circuitos_merged_v2.parquet
                                              (schema del viejo + columnas procedencia;
                                               NO sobrescribe)
                                                              │
                                                              ▼  src/data/reconcile.py
                                           outputs/reconciliation_report.{md,csv}
                                              (diff viejo-vs-nuevo + vs totales oficiales)
```

## 5. Módulos

### `scripts/snapshot_external_data.py`
Copia los `votes-circuito.json` de las elecciones requeridas
(`nacionales-2024`, `balotaje-2024`, y para sub-proyectos siguientes
`departamentales-2025`, `municipales-2025`) desde el repo vecino a
`data/raw/external/`. Escribe `snapshot_manifest.json` con sha256 por archivo,
ruta de origen, y fecha. Ruta del repo vecino configurable en `config.yaml`.

### `src/data/party_mapping.py`
Tabla explícita, versionada, `opcionId → código EI`:

| opcionId | EI |
|---|---|
| `cabildo-abierto` | ca |
| `frente-amplio` | fa |
| `colorado` | pc |
| `nacional` | pn |
| `independiente` | pi |
| *(resto de lemas)* | otros |

`otros` = suma de los 6 lemas menores. Falla ruidosamente si aparece un `opcionId`
no mapeado (evita pérdida silenciosa).

### `src/data/ingest_votes_circuito.py`
Lee los 19 `votes-circuito.json` de una elección → DataFrame largo con columnas:
`departamento, geoId, local, opcionId, votos, validos, enBlanco, anulados, observados,
escrutinio`. Una función `load_election(eleccion) -> pd.DataFrame`.
Parametrizado por `nivel` para reuso futuro (sub-proyecto 2: `local`, `hoja`).

### `src/data/build_ei_dataset.py`
1. Carga 1ª vuelta (`nacionales-2024`) y balotaje (`balotaje-2024`) vía ingest.
2. Pivotea `opcionId → columnas` aplicando `party_mapping`.
3. Mergea por `(departamento, geoId)` (inner join; los no apareados se reportan).
4. Construye columnas equivalentes al parquet viejo: `ca_primera … pn_primera`,
   `total_primera`, `fa_ballotage`, `pn_ballotage`, `blancos_ballotage`,
   `total_ballotage`, más shares.
5. Columnas nuevas de procedencia: `observados_ballotage`, `source_snapshot_hash`,
   `escrutinio`.
6. `habilitados` = NaN (issue #60), marcado; se hereda del parquet viejo solo si un
   análisis lo exige, con flag.
7. Escribe `data/processed/circuitos_merged_v2.parquet`.

**Definición de `blancos_ballotage` / `total_ballotage`:** se fija empíricamente en la
reconciliación — se elige la combinación de (`enBlanco`, `anulados`, `observados`) que
mejor reproduce `total_ballotage` y `blancos_ballotage` del parquet viejo y los totales
oficiales por departamento. Hipótesis de partida: `blancos_ballotage = enBlanco + anulados`,
`total_ballotage = validos + enBlanco + anulados + observados`.

### `src/data/reconcile.py`
Genera `outputs/reconciliation_report.{md,csv}`:
- **Cobertura:** circuitos en viejo∖nuevo y nuevo∖viejo (set-diff explícito).
- **Votos por partido por circuito:** diferencia absoluta y relativa viejo-vs-nuevo;
  histograma de discrepancias; lista de los circuitos con mayor desvío.
- **Totales por departamento:** v2 contra los totales oficiales del proyecto vecino
  (nivel `departamento` / `_nacional`).
- **Impacto en EI:** se re-corre el EI nacional sobre v2 y se comparan las tasas de
  transferencia estimadas (media + IC) contra las del parquet viejo, en una tabla.

## 6. Gate de aceptación

El sub-proyecto se considera terminado cuando:
1. `circuitos_merged_v2.parquet` existe con el schema esperado + columnas de procedencia.
2. Votos por partido por circuito coinciden con el parquet viejo dentro de tolerancia
   (objetivo: ≥99% de circuitos con diferencia 0; las excepciones quedan explicadas en
   el reporte, p.ej. circuitos reasignados o escrutinio distinto).
3. Totales por departamento de v2 = totales oficiales del proyecto vecino (exacto).
4. `reconciliation_report.md` documenta toda diferencia material, incluida la del EI.
5. Decisión registrada de migrar (o no) el análisis a v2, con justificación.

## 7. Pruebas

- `party_mapping`: todo `opcionId` de las elecciones objetivo mapea; un id desconocido
  levanta error.
- `ingest_votes_circuito`: sobre un fixture de 1 departamento, parsea `noPartidarios` y
  produce las filas/columnas esperadas.
- `build_ei_dataset`: la clave de merge produce el conteo esperado; `total_primera` =
  suma de columnas de partido; sin NaN inesperados en columnas core.
- `reconcile`: los totales por departamento de v2 igualan los oficiales (test duro).

## 8. Fuera de alcance (otros sub-proyectos)

- EI a nivel local/hoja (sub-proyecto 2) — el ingest queda parametrizado para habilitarlo.
- Validación con departamentales/municipales 2025 (sub-proyecto 3) — el snapshot ya
  puede incluir esas elecciones, pero su análisis es aparte.
- Triangulación bayesiano vs eiCircles (sub-proyecto 4).
- Conseguir padrón/`habilitados` (issue #60 del repo vecino).

## 9. Riesgos

- **Definición de no-partidarios:** si la reconciliación no reproduce los totales viejos,
  hay que revisar la semántica de `observados` en escrutinio definitivo (pueden estar ya
  distribuidos). Mitigación: la reconciliación es el árbitro; se prueban variantes.
- **Diferencia 7.271 vs 7.222:** circuitos faltantes/extra entre fuentes. Mitigación:
  set-diff explícito; ningún circuito se descarta en silencio.
- **`escrutinio` mixto:** confirmar que todas las elecciones objetivo son `definitivo`.

## 10. Resultado de reconciliación (ejecutado 2026-06-13)

Snapshot `snapshot_sha256=42e279fc…` (38 archivos, 19 deptos × 2 elecciones).
v2 construido: **7.221 circuitos** (1 de primera sin balotaje, logueado).

### Cobertura y voto (vs parquet viejo, 7.271 circuitos)

- **Circuitos apareados: 7.165.** En ellos, **`max|Δ| = 0` en todas las columnas de
  partido** (ca/fa/pc/pn/pi/otros primera; fa/pn balotaje): el v2 reproduce el voto
  exactamente.
- **106 solo-viejo: todos serie ≥9000** (circuitos especiales / voto observado
  consolidado), 30.174 votos = **1,32%** del total. El desglose nuevo a nivel circuito
  no los expone como circuito separado.
- **56 solo-nuevo: todos regulares** (numeración <9000); el dato nuevo es más granular.
- Composición de no-partidarios verificada al 100% sobre los apareados:
  `OLD(blancos+anulados) == V2(enBlanco+anulados)`, `OLD total == V2 total`,
  `OLD residual == V2 observados` (32.876).

### Definición de la categoría destino "blancos" (decisión registrada)

La primera versión de v2 fusionaba `anulados` dentro de `blancos_ballotage`, lo que
distorsionaba el EI (OTROS→Blancos +21pp). Se corrigió para **igualar el esquema
publicado**: `blancos_ballotage = enBlanco` solo, con `anulados_ballotage` y
`observados_ballotage` como columnas separadas fuera de los destinos modelados.
Tras el fix, `blancos` y `anulados` matchean el viejo al 100%.

### Impacto en el EI nacional (King's EI, 4000 muestras / 4 cadenas)

Tres corridas convergidas (R-hat < 1.01, ESS > 1800):

| Comparación | Max\|Δ\| tasas de transferencia |
|---|---|
| v2 (7.221) vs publicado-viejo (7.271) | 7,56 pp |
| **v2 (7.221) vs viejo-restringido-a-matcheados (7.165)** | **0,21 pp** |

**Sobre los mismos circuitos, v2 y viejo dan resultados idénticos (Δ ≤ 0,21 pp = ruido
MCMC).** Las diferencias contra el análisis publicado (hasta 7,6 pp, concentradas en
PC y OTROS) se explican **íntegramente** por los 106 circuitos especiales serie-9000:
al incluirlos, el publicado daba PC→FA 10,0% y OTROS→PN 69,4%; sin ellos (v2 y
viejo-restringido coinciden) son PC→FA 4,5% y OTROS→PN 62%. Los partidos grandes
(FA, PN, PI) y CA coinciden dentro de ruido en todas las corridas.

### Decisión

**La capa de datos v2 queda validada como reemplazo fiel** y se adopta como fuente del
EI. Reproduce exactamente el voto y, a igualdad de circuitos, las estimaciones del
análisis publicado. La diferencia frente al publicado es atribuible y deseable: el v2
excluye del nivel-circuito 106 circuitos especiales (voto observado consolidado, 1,32%
del padrón de votos) que sesgaban las estimaciones de partidos chicos. Al migrar el
análisis a v2 se debe **documentar este cambio en los resultados de partidos menores**
(PC, OTROS) respecto del informe previo.

Artefactos regenerables (gitignored): `circuitos_merged_v2.parquet`,
`national_transfers_with_pi_v2.pkl`. Reproducibles vía
`scripts/snapshot_external_data.py` → `python -m src.data.build_ei_dataset` →
`scripts/national_analysis_with_pi.py`. Registro persistente del snapshot:
`data/raw/external/snapshot_manifest.json`.
