import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "olive": "#bcbd22",
    "cyan": "#17becf",
    "gray": "#7f7f7f",
    "black": "#1c2128",
}

COLOR_LIST = list(COLORS.values())


def plot_spectrum(
    spectrums: list,
    labels: list,
    ground_truth,
    angle_grids,
    num_signal,
    peak_threshold=0.5,
    x_label="Angle",
    y_label="Spectrum",
):
    """Plot spatial spectrum

    Args:
        spectrum (list): Spatial spectrum estimated by the algorithm
        labels (list): Labels for each spectrum
        ground_truth: True incident angles
        angle_grids: Angle grids corresponding to the spatial spectrum
        num_signals: Number of signals
        peak_threshold: Threshold used to find
            peaks
        x_label: x-axis label
        y_label: y-axis label
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # set ticks
    grids_min = angle_grids[0]
    grids_max = angle_grids[-1]
    major_space = (grids_max - grids_min + 1) / 6
    minor_space = major_space / 5
    major_ticks = np.arange(grids_min, grids_max, major_space)
    minor_ticks = np.arange(grids_min, grids_max, minor_space)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    for i, spec in enumerate(spectrums):
        spectrum = spec.result["result"]

        # peaks idx and peak heights
        angle_result = spec.angle_result(num_signal, peak_threshold)
        angles = angle_result["angles"]
        heights = angle_result["heights"]

        # plot spectrum
        ax.plot(angle_grids, spectrum, label=labels[i], color=COLOR_LIST[i])

        # plot peaks
        ax.scatter(angles, heights, color=COLOR_LIST[i], marker="v")
        for j, angle in enumerate(angles):
            ax.annotate(angle, xy=(angle, heights[j]))

    # ground truth
    for i, angle in enumerate(ground_truth):
        if i == 0:
            ax.axvline(
                x=angle,
                color=COLORS["gray"],
                linestyle="--",
                label="True DOAs",
            )
        else:
            ax.axvline(x=angle, color=COLORS["gray"], linestyle="--")

    ax.scatter([], [], color=COLORS["black"], marker="v", label="Estimated")
    # set labels
    ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()
