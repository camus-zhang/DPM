
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from DpMixture import *


class DpPlot(object):
    def __init__(self, ax, DP):
        self.mean,   = ax.plot([], [], 'g-', lw=3)
        self.sample, = ax.plot([], [], 'r-'  )
        self.points, = ax.plot([], [], 'r+', markersize=10, markeredgewidth=2)
        self.x      = np.linspace(-1, 1, 2000)
        self.y      = [0] * len(self.x)
        self.ax     = ax
        self.DP     = DP

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(0, 3)
        self.ax.grid(True)

    def init(self):
        self.mean.set_data([], [])
        self.sample.set_data([], [])
        self.points.set_data([], [])
        return self.points, self.mean, self.sample,
    
    def add_point(self, x):
        self.DP.add_point(x)
        plot.points.set_data(self.DP.data, [0.1]*len(self.DP.data))

    def __call__(self, i):
        # Continuously run, do gibbs sampling and update plots
        if i == 0:
            return self.init()

        self.DP.gibbs(10)
        
        pdf = self.DP.pdf(self.x)
        self.sample.set_data(self.x, pdf)

        decay = 0.01
        self.y = [e * (1-decay) for e in self.y] + decay*pdf
        self.mean.set_data(self.x, self.y)

        return self.points, self.mean, self.sample,

def onclick(event):
    global plot
    if event.xdata:
        plot.add_point(event.xdata)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

DP = DpMixture(GaussCompnt, (0, 1, 0.1, 1), 1.)
plot = DpPlot(ax, DP)

anim = FuncAnimation(fig, plot, frames=1000000, init_func=plot.init, interval=1, repeat=True)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()       


