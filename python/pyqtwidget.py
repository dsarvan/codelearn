#!/usr/bin/env python
# File: pyqtwidget.py
# Name: D.Saravanan
# Date: 19/04/2024

""" Script to build PyQt6 application """

import sys

from PyQt6 import QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    """Subclass of QMainWindow to customize application's main window"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # main widget and layout
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout(widget)

        # QLabel widget to display text/images
        self.label = QtWidgets.QLabel("Application PyQt6!")
        self.add_widget_with_label(layout, self.label, "QLabel:")

        # QPushButton widget clickable button trigger actions
        self.button = QtWidgets.QPushButton("Click Me")
        self.button.clicked.connect(self.button_clicked)
        self.add_widget_with_label(layout, self.button, "QPushButton:")

        # QLineEdit widget allows users enter and edit single line text
        self.line_edit = QtWidgets.QLineEdit()
        self.add_widget_with_label(layout, self.line_edit, "QLineEdit:")

        # QComboBox widget combination of dropdown and text field
        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems(["Option 1", "Option 2", "Option 3"])
        self.add_widget_with_label(layout, self.combo_box, "QComboBox:")

        # QCheckBox widget box that users can check/uncheck
        self.check_box = QtWidgets.QCheckBox("Check Me")
        self.add_widget_with_label(layout, self.check_box, "QCheckBox:")

        # QRadioButton widget selecting one option from set
        self.radio_button = QtWidgets.QRadioButton("Radio Button")
        self.add_widget_with_label(layout, self.radio_button, "QRadioButton:")

        # QTextEdit widget multi-line text edit, text format, copy/past, scroll
        self.text_edit = QtWidgets.QTextEdit()
        self.add_widget_with_label(layout, self.text_edit, "QTextEdit:")

        # QSlider widget provides slider control like adjusting volume
        self.slider = QtWidgets.QSlider()
        self.add_widget_with_label(layout, self.slider, "QSlider:")

        # QSpinBox widget lets users select number from given range
        self.spin_box = QtWidgets.QSpinBox()
        self.add_widget_with_label(layout, self.spin_box, "QSpinBox:")

        # QProgressBar widget displays task progress like file upload/download
        self.progress_bar = QtWidgets.QProgressBar()
        self.add_widget_with_label(layout, self.progress_bar, "QProgressBar:")

        # QTableWidget widget displays structured data in tabular format
        self.table_widget = QtWidgets.QTableWidget(8, 5)
        for i in range(8):
            for j in range(5):
                item = QtWidgets.QTableWidgetItem(f"Cell {i+1},{j+1}")
                self.table_widget.setItem(i, j, item)
        self.add_widget_with_label(layout, self.table_widget, "QTableWidget:")

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
