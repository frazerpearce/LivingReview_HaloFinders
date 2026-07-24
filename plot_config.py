"""Shared Matplotlib defaults for LivingReview figures."""

PLOT_RCPARAMS = {
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.grid": False,
}


def apply_plot_style(plt) -> None:
    """Apply the project-wide figure style to a pyplot module."""
    plt.rcParams.update(PLOT_RCPARAMS)
