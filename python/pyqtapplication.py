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

        # QLabel widget to display text/images
        self.label = QtWidgets.QLabel("Application PyQt6!")
        self.add_widget_with_label(main_layout, self.label, "QLabel:")

        # QPushButton widget clickable button trigger actions
        self.button = QtWidgets.QPushButton("Click Me")
        self.button.clicked.connect(self.button_clicked)
        self.add_widget_with_label(main_layout, self.button, "QPushButton:")

        # QLineEdit widget allows users enter and edit single line text
        self.line_edit = QtWidgets.QLineEdit()
        self.add_widget_with_label(main_layout, self.line_edit, "QLineEdit:")

        # QComboBox widget combination of dropdown and text field
        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems(["Option 1", "Option 2", "Option 3"])
        self.add_widget_with_label(main_layout, self.combo_box, "QComboBox:")

    def button_clicked(self):
        """function set text when clicked"""
        self.label.setText("Button Clicked!")

    def add_widget_with_label(self, layout, widget, label_text):
        """function to add widget with label"""
        hbox = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(label_text)
        hbox.addWidget(label)
        hbox.addWidget(widget)
        layout.addLayout(hbox)


def main():
    """pyqt application"""

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    sys.exit(main())
