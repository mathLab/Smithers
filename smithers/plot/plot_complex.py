import matplotlib.pyplot as plt
import numpy as np

def _enforce_ratio(goal_ratio, supx, infx, supy, infy):
    """
    Computes the right value of `supx,infx,supy,infy` to obtain the desired
    ratio in :func:`plot_eigs`. Ratio is defined as
    ::
        dx = supx - infx
        dy = supy - infy
        max(dx,dy) / min(dx,dy)

    :param float goal_ratio: the desired ratio.
    :param float supx: the old value of `supx`, to be adjusted.
    :param float infx: the old value of `infx`, to be adjusted.
    :param float supy: the old value of `supy`, to be adjusted.
    :param float infy: the old value of `infy`, to be adjusted.
    :return tuple: a tuple which contains the updated values of
        `supx,infx,supy,infy` in this order.
    """

    dx = supx - infx
    if dx == 0:
        dx = 1.0e-16
    dy = supy - infy
    if dy == 0:
        dy = 1.0e-16
    ratio = max(dx, dy) / min(dx, dy)

    if ratio >= goal_ratio:
        if dx < dy:
            goal_size = dy / goal_ratio

            supx += (goal_size - dx) / 2
            infx -= (goal_size - dx) / 2
        elif dy < dx:
            goal_size = dx / goal_ratio

            supy += (goal_size - dy) / 2
            infy -= (goal_size - dy) / 2

    return (supx, infx, supy, infy)

def _plot_limits(numbers, narrow_view):
    if narrow_view:
        supx = max(numbers.real) + 0.05
        infx = min(numbers.real) - 0.05

        supy = max(numbers.imag) + 0.05
        infy = min(numbers.imag) - 0.05

        return _enforce_ratio(8, supx, infx, supy, infy)
    else:
        return np.max(np.ceil(np.abs(numbers)))

def plot(
        numbers,
        show_axes=True,
        show_unit_circle=True,
        figsize=(8, 8),
        title="",
        narrow_view=False,
        dpi=None,
        filename=None,
        legend_label=None
    ):
        """
        Plot complex numbers (for instance, eigenvalues).
        :param np.ndarray numbers: Array of complex numbers to be plotted.
        :param bool show_axes: if True, the cartesian axes will be showed in the
            plot. Default is True.
        :param bool show_unit_circle: if True, the circle with unitary radius
            and center in the origin will be showed. Default is True.
        :param tuple(int,int) figsize: Tuple defining the figure size in inches.
            Default is (8, 8).
        :param str title: Title of the plot.
        :param bool narrow_view: if True, the plot will show only the smallest
            rectangular area which contains all the eigenvalues, with a padding
            of 0.05. Not compatible with `show_axes=True`. Default is False.
        :param int dpi: If not None, the given value is passed to
            ``plt.figure`` in order to specify the DPI of the resulting figure.
        :param str filename: If specified, the plot is saved at `filename`.
        """
        if dpi is not None:
            plt.figure(figsize=figsize, dpi=dpi)
        else:
            plt.figure(figsize=figsize)

        plt.title(title)
        plt.gcf()
        ax = plt.gca()

        points = ax.plot(
            numbers.real, numbers.imag, "bo"
        )

        if narrow_view:
            supx, infx, supy, infy = _plot_limits(numbers, narrow_view)

            # set limits for axis
            ax.set_xlim((infx, supx))
            ax.set_ylim((infy, supy))

            # x and y axes
            if show_axes:
                endx = np.min([supx, 1.0])
                ax.annotate(
                    "",
                    xy=(endx, 0.0),
                    xytext=(np.max([infx, -1.0]), 0.0),
                    arrowprops=dict(arrowstyle=("->" if endx == 1.0 else "-")),
                )

                endy = np.min([supy, 1.0])
                ax.annotate(
                    "",
                    xy=(0.0, endy),
                    xytext=(0.0, np.max([infy, -1.0])),
                    arrowprops=dict(arrowstyle=("->" if endy == 1.0 else "-")),
                )
        else:
            # set limits for axis
            limit = _plot_limits(numbers, narrow_view)

            ax.set_xlim((-limit, limit))
            ax.set_ylim((-limit, limit))

            # x and y axes
            if show_axes:
                ax.annotate(
                    "",
                    xy=(np.max([limit * 0.8, 1.0]), 0.0),
                    xytext=(np.min([-limit * 0.8, -1.0]), 0.0),
                    arrowprops=dict(arrowstyle="->"),
                )
                ax.annotate(
                    "",
                    xy=(0.0, np.max([limit * 0.8, 1.0])),
                    xytext=(0.0, np.min([-limit * 0.8, -1.0])),
                    arrowprops=dict(arrowstyle="->"),
                )

        plt.ylabel("Imaginary")
        plt.xlabel("Real")

        if show_unit_circle:
            unit_circle = plt.Circle(
                (0.0, 0.0),
                1.0,
                color="green",
                fill=False,
                linestyle="--",
            )
            ax.add_artist(unit_circle)

        # Dashed grid
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle("-.")
        ax.grid(True)

        ax.set_aspect("equal")

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
