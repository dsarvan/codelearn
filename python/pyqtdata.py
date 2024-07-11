#!/usr/bin/env python
# File: pyqtdata.py
# Name: D.Saravanan
# Date: 20/04/2024

""" Script data visualization application using PyQt6, Matplotlib and Seaborn """

import sys

import matplotlib.pyplot as plt
import seaborn as sns
from PyQt6 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

plt.style.use("classic")
plt.rc("text", usetex="True")
plt.rc("figure", titlesize=12)
plt.rc("pgf", texsystem="pdflatex")
plt.rc("axes", labelsize=12, titlesize=12)
plt.rc("font", family="serif", weight="normal", size=10)


class MainWindow(QtWidgets.QMainWindow):
    """Subclass of QMainWindow to customize application's main window"""

    def __init__(self, iris):
        super().__init__()
        self.iris = iris
        self.setWindowTitle("Iris Dataset Visualization")
        self.setGeometry(100, 100, 1200, 900)
        self.initUI(self.iris)

    def initUI(self, iris):
        self.graphWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.graphWidget)
        layout = QtWidgets.QVBoxLayout(self.graphWidget)
        self.addScatterPlot(layout, iris)
        self.addHistogram(layout, iris)
        self.addBoxPlot(layout, iris)

    def addPlotCanvas(self, layout, plot_func):
        canvas = FigureCanvasQTAgg(Figure(figsize=(6, 4)))
        plot_func(canvas.figure)
        layout.addWidget(canvas)

    def addScatterPlot(self, layout, iris):
        def scatterplot(figure):
            ax = figure.add_subplot(111)
            sns.scatterplot(iris, x="sepal_length", y="sepal_width", hue="species", ax=ax)
            ax.set_title("Sepal length vs Sepal width")

        self.addPlotCanvas(layout, scatterplot)

    def addHistogram(self, layout, iris):
        def histogram(figure):
            ax = figure.add_subplot(111)
            sns.histplot(iris, x="petal_length", hue="species", kde=True, ax=ax)
            ax.set_title("Histogram of Petal length")

        self.addPlotCanvas(layout, histogram)

    def addBoxPlot(self, layout, iris):
        def boxplot(figure):
            ax = figure.add_subplot(111)
            sns.boxplot(iris, x="species", y="petal_width", ax=ax)
            ax.set_title("Box plot of Petal width")

        self.addPlotCanvas(layout, boxplot)


def main():
    """pyqt application"""

    app = QtWidgets.QApplication(sys.argv)

    iris = sns.load_dataset("iris")

    window = MainWindow(iris)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    sys.exit(main())
