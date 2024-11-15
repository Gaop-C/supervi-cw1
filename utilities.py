import numpy as np
import matplotlib.pyplot as plt
import requests


def set_axes(ax, x_major_scale, y_major_scale, x_subscale, y_subscale, xlim_left=None, xlim_right=None, ylim_bottom=None, ylim_top=None):
    # Make the axes pass through the origin
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    # Hide the top and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.set_xlim(xlim_left, xlim_right)
    ax.set_ylim(ylim_bottom, ylim_top)

    # Set the main scale and subscale of axes
    ax.xaxis.set_major_locator(plt.MultipleLocator(x_major_scale))
    ax.yaxis.set_major_locator(plt.MultipleLocator(y_major_scale))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(x_subscale))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(y_subscale))

    # remove the origin scale
    ax.set_xticks(np.delete(ax.get_xticks(),
                            np.where(ax.get_xticks() == 0)))
    ax.set_yticks(np.delete(ax.get_yticks(),
                            np.where(ax.get_yticks() == 0)))

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')


def read_data(spot):
    csv_data = np.genfromtxt(spot.raw, delimiter=",", names=True)
    return csv_data
