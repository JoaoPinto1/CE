import matplotlib.pyplot as plt 
import numpy as np


def create_interactive_plot(title, xlabel, ylabel, lim_x, lim_y):
    plt.ion()
    # creating subplot and figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(*lim_y)
    ax.set_xlim(*lim_x)
    line1, = ax.plot([], [])
    # setting labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return line1, fig, ax

def update_graph(x, y, graph, fig, ax):
    graph.set_xdata(np.append(graph.get_xdata(), x))
    graph.set_ydata(np.append(graph.get_ydata(), y))
    ax.draw_artist(ax.patch)
    ax.draw_artist(graph)
    fig.canvas.draw()
    fig.canvas.flush_events()