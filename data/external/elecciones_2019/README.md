# 2019 Uruguayan Election Results - Circuit Level Data

This directory contains official election results from the Uruguayan Electoral Court (Corte Electoral) at the circuit level (CRV - Comisión Receptora de Votos), downloaded from the government's open data portal.

## Data Files

### 1. elecciones_nacionales_2019_circuito.xlsx
**First Round (Primera Vuelta) - October 27, 2019**

- **Source**: Corte Electoral Uruguay Open Data Catalog
- **File Size**: ~350 KB
- **Data Points**: 7,229 circuit records
- **Geographic Coverage**: 19 departments across Uruguay

#### Sheets:
- **Datos**: Main data sheet with circuit-level results
- **Metadatos**: Metadata describing the data structure

#### Columns:
- `Departamento`: Department code (e.g., 'MO' = Montevideo, 'CA' = Canelones)
- `CRV`: Circuit identification number
- `Serie`: Serial identifier
- `TotalHabilitados`: Total eligible voters
- `TotalVotosEmitidos`: Total votes cast
- `TotalVotosNOObservados`: Votes without observations
- `TotalVotosObservados`: Votes with observations (observations refer to contested ballots)
- `TotalAnulados`: Annulled votes
- `TotalEnBlanco`: Blank votes
- `TotalSoloSi`: Votes for "Yes" on the constitutional referendum (security question)

#### Data Quality:
- Covers all 19 departments
- 7,229 circuit-level records
- Maximum CRV ID: 9078

### 2. balotaje_2019_circuito.xlsx
**Runoff Election (Balotaje) - November 24, 2019**

- **Source**: Corte Electoral Uruguay Open Data Catalog
- **File Size**: ~408 KB
- **Data Points**: 7,220 circuit records
- **Geographic Coverage**: 19 departments across Uruguay

#### Sheets:
- **Datos**: Main data sheet with circuit-level runoff results
- **Metadatos**: Metadata describing the data structure
- **Listas de valores**: Value lists for reference data

#### Columns:
- `Departamento`: Department code
- `CRV`: Circuit identification number
- `Serie`: Serial identifier
- `Total_Habilitados`: Total eligible voters
- `Total_Votos_Emitidos`: Total votes cast
- `Total_Votos_NO_Observados`: Votes without observations
- `Total_Votos_Observados`: Votes with observations
- `Total_Anulados`: Annulled votes
- `Total_EN_Blanco`: Blank votes
- `Total_Martinez_Villar`: Votes for the FA (Frente Amplio) ticket: Martínez-Villar
- `Total_Lacalle Pou_Argimon`: Votes for the PN (Partido Nacional) ticket: Lacalle Pou-Argimón

#### Data Quality:
- Covers all 19 departments
- 7,220 circuit-level records
- Maximum CRV ID: 9024

## Department Codes

The data uses the following two-letter abbreviations for departments:

| Code | Department          |
|------|-------------------|
| MO   | Montevideo        |
| CA   | Canelones         |
| MA   | Maldonado         |
| RO   | Rocha             |
| TT   | Treinta y Tres    |
| CL   | Cerro Largo       |
| RV   | Río Negro         |
| AR   | Artigas           |
| SA   | Salto             |
| PA   | Paysandú          |
| RN   | Rivera            |
| SO   | Soriano           |
| CO   | Colonia           |
| SJ   | San José          |
| FS   | Florida           |
| FD   | Flores            |
| DU   | Durazno           |
| LA   | Lavalleja         |
| TA   | Tacuarembó        |

## Use Cases

These files are suitable for:

- Control variables for econometric analysis
- Calculating electoral support at circuit level
- Urban/rural electoral comparisons
- Regional political patterns analysis
- Voter turnout analysis
- Analysis of blank and annulled votes by region
- Temporal comparison with other election cycles

## Data Preparation Notes

1. The Primera Vuelta data includes votes on a constitutional referendum (TotalSoloSi)
2. Circuit identifiers (CRV) can be used to merge with geographic or demographic data
3. Column naming conventions differ slightly between the two files (CamelCase vs snake_case)
4. Both files contain metadata sheets with additional documentation
5. Some circuits may have lower vote totals due to geographic isolation or special circumstances

## Source Information

**Data Source**: Corte Electoral Uruguay - Open Data Catalog (Catálogo de Datos Abiertos)

**URLs**:
- Primera Vuelta: https://catalogodatos.gub.uy/dataset/corte-electoral-elecciones-nacionales-2019
- Balotaje: https://catalogodatos.gub.uy/dataset/corte-electoral-balotaje_2019

**License**: Licencia de DAG de Uruguay (Uruguay's Open Government License)
https://www.gub.uy/agencia-gobierno-electronico-sociedad-informacion-conocimiento/sites/agencia-gobierno-electronico-sociedad-informacion-conocimiento/files/documentos/publicaciones/licencia_de_datos_abiertos_0.pdf

**Maintainer**: Daniel Drinfeld Rosenberg (Corte Electoral)
Email: ddrinfeld@corteelectoral.gub.uy

## Data Access Methods

### Excel (Native Format)
The files can be opened directly in Excel, LibreOffice Calc, or similar spreadsheet applications.

### Python/Pandas
```python
import pandas as pd

# Load primera vuelta data
df_primera = pd.read_excel('elecciones_nacionales_2019_circuito.xlsx', sheet_name='Datos')

# Load balotaje data
df_balotaje = pd.read_excel('balotaje_2019_circuito.xlsx', sheet_name='Datos')
```

### R
```r
library(readxl)

# Load primera vuelta data
df_primera <- read_excel('elecciones_nacionales_2019_circuito.xlsx', sheet = 'Datos')

# Load balotaje data
df_balotaje <- read_excel('balotaje_2019_circuito.xlsx', sheet = 'Datos')
```

## Related Data

Additional election data available from Corte Electoral:
- 2014 National Elections
- 2020 Departmental and Municipal Elections
- 2024 National Elections
- Internal Party Elections (2019, 2024)
- Constitutional referendums

## Last Updated

Downloaded: February 4, 2026
Source Last Updated: March 7, 2024
