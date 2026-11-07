"""Shared matplotlib style: the report body is Times, so every figure uses Times text and STIX math.

Figures are built at their final width (5.5 in, one NeurIPS column) so LaTeX never rescales them.
Panels carry only an (a), (b) tag; the LaTeX caption does the describing.
"""
import matplotlib

WIDTH = 5.5


def apply():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "savefig.bbox": "tight",
    })


def tag(ax, letter):
    ax.set_title(f"({letter})", loc="left", fontsize=8)
