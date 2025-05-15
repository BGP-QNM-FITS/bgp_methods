import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class PlotConfig:
    def __init__(self):
        self.style = "stylesheet.mplstyle"
        self.fig_width = 246.0 * (1.0 / 72.27)
        self.fig_width_2 = 510.0 * (1.0 / 72.27)
        self.fig_height = self.fig_width / 1.618
        self.fig_height_2 = self.fig_width_2 / 1.618
        self.colors = [
            "#395470",  # soft viridis-style blue
            "#5A7A87",  # pastel twilight blue-teal
            "#A4C9A7",  # pastel sage
            "#D3C76A",  # pastel olive
            "#E9DF83",  # slightly darker pastel yellow
        ]
        self.colors2 = [
            "#8B5FBF",  # muted violet
            "#A15C9E",  # dusty magenta
            "#C26C88",  # muted rose
            "#D87570",  # soft reddish-pink
            "#DE6A5E",  # soft muted red
        ]
        self.colormap = LinearSegmentedColormap.from_list("custom_colormap", self.colors)
        self.colormap2 = LinearSegmentedColormap.from_list("custom_colormap2", self.colors2)

    def apply_style(self):
        plt.style.use(self.style)
