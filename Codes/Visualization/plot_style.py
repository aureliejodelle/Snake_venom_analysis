"""
plot_style.py - shared Matplotlib styling for all figures.

Single source of truth so every plot (topology, statistics, significance,
persistence, pI, …)

Usage (after importing matplotlib / pyplot):

    from plot_style import use_computer_modern
    use_computer_modern()
"""
from __future__ import annotations

import matplotlib as mpl

CLASS_DISPLAY = {
    "SNMP":       "SVMP",
    "PLA":        "PLA$_2$",
    "3ft":        "3FTx",
    "serine":     "serine",
    "kallikrein": "kallikrein",
    "kunitz":     "kunitz",
}


def disp(name: str) -> str:
    return CLASS_DISPLAY.get(name, name)


def disp_list(names) -> list:
    """Map ``disp`` over an iterable of class names."""
    return [disp(n) for n in names]


CLASS_COLORS = {
    "3ft":        "#C0392B",   # red
    "PLA":        "#2980B9",   # blue
    "SNMP":       "#27AE60",   # green
    "kallikrein": "#8E44AD",   # purple
    "kunitz":     "#E67E22",   # orange
    "serine":     "#16A085",   # teal
}

# Fallback sequence for any class not in CLASS_COLORS.
PALETTE_FALLBACK = [
    "#D35400", "#F39C12", "#1ABC9C", "#E91E63", "#795548", "#607D8B",
]


def class_color(name: str, i: int = 0) -> str:
    """Canonical colour for a class; falls back to PALETTE_FALLBACK[i] if the
    class is not in CLASS_COLORS."""
    return CLASS_COLORS.get(name, PALETTE_FALLBACK[i % len(PALETTE_FALLBACK)])


def class_colors(names) -> list:
    """Colour list for an ordered iterable of class names (order-independent for
    known classes; positional fallback only for unknown ones)."""
    return [class_color(n, i) for i, n in enumerate(names)]


def use_computer_modern() -> None:
    """Apply Computer Modern serif fonts + consistent sizes to Matplotlib.

    """
    mpl.rcParams.update({
        # Computer Modern typography 
        "text.usetex":                 False,  
        "font.family":                 "serif",
        "font.serif":                  ["cmr10", "CMU Serif",
                                        "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset":            "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus":          False,   
        # Consistent font sizes 
        "font.size":                   12,
        "axes.titlesize":              13,
        "axes.labelsize":              12,
        "xtick.labelsize":             11,
        "ytick.labelsize":             11,
        "legend.fontsize":             10,
        #Shared cosmetics 
        "axes.spines.top":             False,
        "axes.spines.right":           False,
        #Editable text in vector output 
        "pdf.fonttype":                42,
        "ps.fonttype":                 42,
    })
