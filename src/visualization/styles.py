"""
Sistema Centralizado de Estilos para Visualizaciones
Estilo profesional tipo Tableau sin grid
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional

# ==================== PALETAS DE COLORES ====================

# Paleta principal - Colores vibrantes pero profesionales
PARTY_COLORS = {
    'FA': '#9B59B6',      # Morado/Púrpura (Frente Amplio)
    'PN': '#27AE60',      # Verde esmeralda (Partido Nacional)
    'PC': '#E74C3C',      # Rojo coral (Partido Colorado)
    'CA': '#F39C12',      # Naranja/Dorado (Cabildo Abierto)
    'PI': '#3498DB',      # Azul brillante (Partido Independiente)
    'OTROS': '#95A5A6',   # Gris neutro
    'Blancos': '#BDC3C7', # Gris claro
}

# Paleta secuencial para mapas (verde → amarillo → rojo)
SEQUENTIAL_PALETTE = [
    '#27AE60',  # Verde oscuro (bajo)
    '#52BE80',  # Verde medio
    '#F1C40F',  # Amarillo (medio)
    '#E67E22',  # Naranja (medio-alto)
    '#E74C3C',  # Rojo (alto)
]

# Paleta divergente para cambios (azul ← neutro → rojo)
DIVERGENT_PALETTE = [
    '#3498DB',  # Azul (negativo)
    '#85C1E9',  # Azul claro
    '#F8F9F9',  # Casi blanco (neutral)
    '#F5B7B1',  # Rojo claro
    '#E74C3C',  # Rojo (positivo)
]

# Paleta categórica moderna (inspirada en Tableau 10)
CATEGORICAL_PALETTE = [
    '#1F77B4',  # Azul
    '#FF7F0E',  # Naranja
    '#2CA02C',  # Verde
    '#D62728',  # Rojo
    '#9467BD',  # Púrpura
    '#8C564B',  # Marrón
    '#E377C2',  # Rosa
    '#7F7F7F',  # Gris
    '#BCBD22',  # Verde oliva
    '#17BECF',  # Cian
]

# ==================== CONFIGURACIÓN DE ESTILO ====================

def setup_professional_style():
    """
    Configura estilo profesional global para matplotlib y seaborn.
    Inspirado en Tableau con modificaciones para publicaciones.
    """
    # Reiniciar a defaults
    mpl.rcdefaults()

    # Configuración general
    plt.style.use('seaborn-v0_8-whitegrid')

    # Personalización del estilo
    custom_params = {
        # Figura
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.format': 'png',

        # Ejes
        'axes.facecolor': 'white',
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 1.0,
        'axes.grid': False,  # SIN GRID por defecto
        'axes.axisbelow': True,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.titlepad': 15,
        'axes.labelpad': 8,
        'axes.labelweight': 'normal',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,

        # Grid (cuando se active manualmente)
        'grid.color': '#E5E5E5',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,

        # Ticks
        'xtick.major.size': 4,
        'xtick.minor.size': 0,
        'xtick.major.width': 1,
        'xtick.labelsize': 10,
        'xtick.color': '#333333',
        'ytick.major.size': 4,
        'ytick.minor.size': 0,
        'ytick.major.width': 1,
        'ytick.labelsize': 10,
        'ytick.color': '#333333',

        # Leyenda
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.edgecolor': '#CCCCCC',
        'legend.facecolor': 'white',
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        'legend.borderpad': 0.8,
        'legend.labelspacing': 0.6,

        # Fuentes
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif'],
        'font.size': 10,
        'font.weight': 'normal',

        # Líneas
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'lines.markeredgewidth': 0,

        # Parches (barras, etc)
        'patch.linewidth': 0.5,
        'patch.edgecolor': '#CCCCCC',
        'patch.force_edgecolor': False,

        # Otros
        'mathtext.default': 'regular',
    }

    mpl.rcParams.update(custom_params)

    # Configurar paleta seaborn
    sns.set_palette(CATEGORICAL_PALETTE)


def apply_minimal_grid(ax, axis='y', alpha=0.2):
    """
    Aplica grid minimalista solo cuando sea necesario.

    Args:
        ax: Matplotlib axis
        axis: 'x', 'y', o 'both'
        alpha: Transparencia del grid (0-1)
    """
    ax.grid(True, axis=axis, color='#E5E5E5', linestyle='-',
            linewidth=0.5, alpha=alpha, zorder=0)
    ax.set_axisbelow(True)


def remove_spines(ax, top=True, right=True, left=False, bottom=False):
    """
    Remueve spines específicos para un look más limpio.

    Args:
        ax: Matplotlib axis
        top, right, left, bottom: Booleanos para remover cada spine
    """
    if top:
        ax.spines['top'].set_visible(False)
    if right:
        ax.spines['right'].set_visible(False)
    if left:
        ax.spines['left'].set_visible(False)
    if bottom:
        ax.spines['bottom'].set_visible(False)


def add_value_labels(ax, orientation='vertical', padding=3, fontsize=9,
                     format_str='{:.1f}%', color='#333333'):
    """
    Añade etiquetas de valores en barras.

    Args:
        ax: Matplotlib axis
        orientation: 'vertical' o 'horizontal'
        padding: Distancia de la barra
        fontsize: Tamaño de fuente
        format_str: Formato de los números
        color: Color del texto
    """
    if orientation == 'vertical':
        for container in ax.containers:
            ax.bar_label(container, fmt=format_str, padding=padding,
                        fontsize=fontsize, color=color, weight='normal')
    else:  # horizontal
        for container in ax.containers:
            ax.bar_label(container, fmt=format_str, padding=padding,
                        fontsize=fontsize, color=color, weight='normal')


def format_percentage_axis(ax, axis='y', decimals=0):
    """
    Formatea eje como porcentajes.

    Args:
        ax: Matplotlib axis
        axis: 'x' o 'y'
        decimals: Decimales a mostrar
    """
    from matplotlib.ticker import FuncFormatter

    def to_percent(y, position):
        return f'{100 * y:.{decimals}f}%'

    if axis == 'y':
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(to_percent))


def add_subtle_background(ax, color='#F8F9FA', alpha=0.3):
    """
    Añade fondo sutil detrás del plot.

    Args:
        ax: Matplotlib axis
        color: Color del fondo
        alpha: Transparencia
    """
    ax.set_facecolor(color)
    ax.patch.set_alpha(alpha)


def create_color_gradient(color_start, color_end, n_colors=10):
    """
    Crea gradiente de colores entre dos colores.

    Args:
        color_start: Color inicial (hex)
        color_end: Color final (hex)
        n_colors: Número de colores en el gradiente

    Returns:
        Lista de colores hex
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors

    # Convertir a RGB
    c1 = mcolors.hex2color(color_start)
    c2 = mcolors.hex2color(color_end)

    # Crear gradiente
    gradient = []
    for i in range(n_colors):
        ratio = i / (n_colors - 1)
        r = c1[0] + ratio * (c2[0] - c1[0])
        g = c1[1] + ratio * (c2[1] - c1[1])
        b = c1[2] + ratio * (c2[2] - c1[2])
        gradient.append(mcolors.rgb2hex((r, g, b)))

    return gradient


def get_party_color(party: str, alpha: float = 1.0) -> str:
    """
    Obtiene color de un partido político.

    Args:
        party: Código del partido
        alpha: Transparencia (0-1)

    Returns:
        Color en formato hex o rgba
    """
    color = PARTY_COLORS.get(party.upper(), '#95A5A6')

    if alpha < 1.0:
        from matplotlib.colors import to_rgba
        rgba = to_rgba(color, alpha)
        return rgba

    return color


def apply_tableau_style(ax, title=None, xlabel=None, ylabel=None):
    """
    Aplica estilo completo tipo Tableau a un axis.

    Args:
        ax: Matplotlib axis
        title: Título opcional
        xlabel: Etiqueta X opcional
        ylabel: Etiqueta Y opcional
    """
    # Remover spines superiores y derecho
    remove_spines(ax, top=True, right=True, left=False, bottom=False)

    # Grid minimalista solo en Y
    apply_minimal_grid(ax, axis='y', alpha=0.15)

    # Títulos y labels
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold',
                    pad=15, loc='left', color='#2C3E50')

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, fontweight='normal',
                     labelpad=8, color='#34495E')

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, fontweight='normal',
                     labelpad=8, color='#34495E')

    # Ajustar colores de ticks
    ax.tick_params(colors='#7F8C8D', which='both')

    # Fondo sutil
    ax.set_facecolor('#FAFBFC')


def save_publication_figure(fig, filepath, dpi=300, bbox_inches='tight',
                            transparent=False):
    """
    Guarda figura en formato publicación (PNG + PDF).

    Args:
        fig: Matplotlib figure
        filepath: Ruta sin extensión
        dpi: Resolución
        bbox_inches: Recorte
        transparent: Fondo transparente
    """
    from pathlib import Path

    filepath = Path(filepath)
    base = filepath.parent / filepath.stem

    # Guardar PNG
    fig.savefig(f"{base}.png", dpi=dpi, bbox_inches=bbox_inches,
               facecolor='white' if not transparent else 'none',
               edgecolor='none', transparent=transparent)

    # Guardar PDF
    fig.savefig(f"{base}.pdf", bbox_inches=bbox_inches,
               facecolor='white' if not transparent else 'none',
               edgecolor='none', transparent=transparent)


# ==================== INICIALIZACIÓN ====================

# Aplicar estilo por defecto al importar
setup_professional_style()


# ==================== EJEMPLOS DE USO ====================

def example_bar_chart():
    """Ejemplo de gráfico de barras con el estilo."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Artigas', 'Canelones', 'Cerro Largo', 'Colonia', 'Montevideo']
    values = [0.45, 0.52, 0.38, 0.48, 0.56]

    bars = ax.bar(categories, values, color=CATEGORICAL_PALETTE[:5],
                   edgecolor='white', linewidth=1.5)

    apply_tableau_style(ax,
                       title='Retención FA por Departamento',
                       ylabel='Retención (%)')

    format_percentage_axis(ax, axis='y')
    add_value_labels(ax, orientation='vertical')

    plt.tight_layout()
    return fig, ax


def example_line_chart():
    """Ejemplo de gráfico de líneas con el estilo."""
    fig, ax = plt.subplots(figsize=(12, 6))

    years = np.arange(2010, 2025)
    fa = 45 + 5 * np.random.randn(15).cumsum() / 10
    pn = 42 + 5 * np.random.randn(15).cumsum() / 10

    ax.plot(years, fa, color=PARTY_COLORS['FA'], linewidth=2.5,
            label='Frente Amplio', marker='o', markersize=6)
    ax.plot(years, pn, color=PARTY_COLORS['PN'], linewidth=2.5,
            label='Partido Nacional', marker='s', markersize=6)

    apply_tableau_style(ax,
                       title='Evolución Electoral 2010-2024',
                       xlabel='Año',
                       ylabel='Porcentaje de Votos')

    ax.legend(loc='best', frameon=True, shadow=False)

    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    # Mostrar ejemplos
    fig1, ax1 = example_bar_chart()
    fig2, ax2 = example_line_chart()
    plt.show()
