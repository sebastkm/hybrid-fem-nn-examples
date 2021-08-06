import matplotlib.pyplot as plt


def auto_adjust_limits(aspect_ratio=0.8):
    ax = plt.gca()
    ax.autoscale()
    # recompute the ax.dataLim
    ax.relim()
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0) * aspect_ratio)
    plt.draw()
