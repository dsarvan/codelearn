#!/usr/bin/env python
# File: pysideapplication.py
# Name: D.Saravanan
# Date: 19/05/2024

""" Script to build PySide6 application """

import sys

from PySide6 import QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    """Subclass of QMainWindow to customize application's main window"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # main widget and layout
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout(widget)

    def add_widget_with_label(self, layout, widget, label_text):
        """function to add widget with label"""
        hbox = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(label_text)
        hbox.addWidget(label)
        hbox.addWidget(widget)
        layout.addLayout(hbox)


def main():
    """pyside application"""

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    sys.exit(main())
