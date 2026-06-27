"""Shared matplotlib styling for tutorial figures."""
import matplotlib.pyplot as plt


colors = [
    "royalblue", "green", "orange", "red", "navy", "brown", "teal", "pink", "gray",
]


def set_static_styles():
    plt.rcParams.update({
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "Times", "DejaVu Serif", "Bitstream Vera Serif",
            "Computer Modern Roman", "New Century Schoolbook", "Century Schoolbook L",
            "Utopia", "ITC Bookman", "Bookman", "Nimbus Roman No9 L", "Palatino",
            "Charter", "serif",
        ],
    })


def create_subplots(nrow=1, ncol=1, sharex=False, sharey=False):
    set_static_styles()
    width, height = 8 * ncol, 6 * nrow
    fig, axs = plt.subplots(nrow, ncol, figsize=(width, height), sharex=sharex, sharey=sharey)
    if nrow * ncol == 1:
        axs = [[axs]]
    elif nrow == 1:
        axs = [axs]
    elif ncol == 1:
        axs = [[ax] for ax in axs]
    return fig, axs
