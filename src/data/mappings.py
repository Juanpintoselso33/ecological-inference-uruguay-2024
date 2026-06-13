"""
Mapeos canónicos entre los identificadores del proyecto Uruguay Electoral Map
y los códigos usados por el pipeline de inferencia ecológica.
"""

# opcionId (lema) -> código EI. Los lemas no listados se agregan en 'otros'.
_EXPLICIT_PARTY_MAP = {
    "cabildo-abierto": "ca",
    "frente-amplio": "fa",
    "colorado": "pc",
    "nacional": "pn",
    "independiente": "pi",
}

# Lemas menores conocidos de 2024 (validan que 'otros' es deliberado, no un typo).
_KNOWN_OTROS = {
    "identidad-soberana",
    "constitucional-ambientalista",
    "ecologista-radical-intransigente",
    "asamblea-popular",
    "por-los-cambios-necesarios-pcn",
    "avanzar-republicano",
}

PARTY_MAP = dict(_EXPLICIT_PARTY_MAP)
for _lema in _KNOWN_OTROS:
    PARTY_MAP[_lema] = "otros"


def map_opcion(opcion_id: str) -> str:
    """Devuelve el código EI para un opcionId. Levanta KeyError si es desconocido."""
    try:
        return PARTY_MAP[opcion_id]
    except KeyError as exc:
        raise KeyError(
            f"opcionId desconocido: {opcion_id!r}. Agregalo a mappings.PARTY_MAP "
            f"como código de partido o como 'otros'."
        ) from exc


# slug del repo vecino -> nombre con acento usado en el parquet histórico.
DEPARTAMENTO_DISPLAY = {
    "artigas": "Artigas",
    "canelones": "Canelones",
    "cerro_largo": "Cerro Largo",
    "colonia": "Colonia",
    "durazno": "Durazno",
    "flores": "Flores",
    "florida": "Florida",
    "lavalleja": "Lavalleja",
    "maldonado": "Maldonado",
    "montevideo": "Montevideo",
    "paysandu": "Paysandú",
    "rio_negro": "Río Negro",
    "rivera": "Rivera",
    "rocha": "Rocha",
    "salto": "Salto",
    "san_jose": "San José",
    "soriano": "Soriano",
    "tacuarembo": "Tacuarembó",
    "treinta_y_tres": "Treinta y Tres",
}


def departamento_display(slug: str) -> str:
    """slug lowercase del repo vecino -> nombre con acento. KeyError si desconocido."""
    try:
        return DEPARTAMENTO_DISPLAY[slug]
    except KeyError as exc:
        raise KeyError(f"departamento desconocido: {slug!r}") from exc
