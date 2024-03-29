#!/usr/bin/env python
# File: pyqtplot.py
# Name: D.Saravanan
# Date: 16/05/2023

""" Script to plotting with PyQtGraph """

import sys

import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    """Subclass of QMainWindow to customize application's main window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Plotting with PyQtGraph")

        # set the size parameters (width, height) pixels
        self.setFixedSize(QtCore.QSize(400, 300))

        # set the central widget of the window
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]

        # set the background color using hex notation #121317 as a string
        self.graphWidget.setBackground("#121317")

        # set the main plot title, text color, text size, text weight, text style
        self.graphWidget.setTitle(
            "Temperature Plot", color="#dcdcdc", size="10pt", bold=True, italic=False
        )

        # set the axis labels (position and text), style parameters
        styles = {"color": "#dcdcdc", "font-size": "10pt"}
        self.graphWidget.setLabel("left", "Temperature (C)", **styles)
        self.graphWidget.setLabel("bottom", "Hour (H)", **styles)

        # set the legend which represents given line
        self.graphWidget.addLegend()

        # set the background grid for both the x and y axis
        self.graphWidget.showGrid(x=True, y=True)

        # set the axis limits within the specified ranges and padding
        self.graphWidget.setXRange(1, 10, padding=0.1)
        self.graphWidget.setYRange(29, 45, padding=0.1)

        # line color in 3-tuple of int values, line width in pixels, solidline style
        lvalue = pg.mkPen(
            color=(220, 220, 220), width=1, style=QtCore.Qt.PenStyle.SolidLine
        )

        # plot data: x, y values with lines drawn using Qt's QPen types & marker '+'
        self.graphWidget.plot(
            hour,
            temperature,
            name="Sensor 1",
            pen=lvalue,
            symbol="+",
            symbolSize=8,
            symbolBrush=("r"),
        )


def main():
    """Need one (and only one) QApplication instance per application.
    Pass in sys.argv to allow command line arguments for the application.
    If no command line arguments than QApplication([]) is required."""
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()  # an instance of the class MainWindow
    window.show()  # windows are hidden by default

    sys.exit(app.exec())  # start the event loop


if __name__ == "__main__":
    main()
