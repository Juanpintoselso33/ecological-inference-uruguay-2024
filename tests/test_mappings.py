# tests/test_mappings.py
import pytest
from src.data.mappings import PARTY_MAP, DEPARTAMENTO_DISPLAY, map_opcion, departamento_display


def test_party_map_known_lemas():
    assert map_opcion("cabildo-abierto") == "ca"
    assert map_opcion("frente-amplio") == "fa"
    assert map_opcion("colorado") == "pc"
    assert map_opcion("nacional") == "pn"
    assert map_opcion("independiente") == "pi"


def test_party_map_minor_lemas_become_otros():
    for lema in ["identidad-soberana", "constitucional-ambientalista",
                 "ecologista-radical-intransigente", "asamblea-popular",
                 "por-los-cambios-necesarios-pcn", "avanzar-republicano"]:
        assert map_opcion(lema) == "otros"


def test_party_map_unknown_raises():
    with pytest.raises(KeyError):
        map_opcion("partido-inventado")


def test_departamento_display_has_accents():
    assert departamento_display("paysandu") == "Paysandú"
    assert departamento_display("rio_negro") == "Río Negro"
    assert departamento_display("san_jose") == "San José"
    assert departamento_display("cerro_largo") == "Cerro Largo"
    assert departamento_display("treinta_y_tres") == "Treinta y Tres"


def test_departamento_display_count():
    assert len(DEPARTAMENTO_DISPLAY) == 19
