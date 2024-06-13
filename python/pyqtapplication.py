#!/usr/bin/env python
# File: pyqtapplication.py
# Name: D.Saravanan
# Date: 19/04/2024

""" Script to build PyQt6 application """

import sys

from PyQt6 import QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    """Subclass of QMainWindow to customize application's main window"""

    def __init__(self):
        super().__init__()

        # main widget and layout
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)


def main():
    """pyqt application"""

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    sys.exit(main())
