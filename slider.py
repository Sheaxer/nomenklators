from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QLabel, QSlider, QHBoxLayout


class SliderDuo(QWidget):
    changed = pyqtSignal(int)

    def __init__(self, text, default_value, min_value, max_value):
        super().__init__()
        self.textLabel = QLabel(text)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_value, max_value)
        self.slider.setValue(default_value)
        self.value = default_value
        self.label = QLabel(str(self.value))
        self.label.setStyleSheet('QLabel { background: #007AA5; border-radius: 3px;}')
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.label.setMinimumWidth(80)
        self.slider.valueChanged.connect(self.change_value)
        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(self.textLabel)
        hbox.addWidget(self.slider)
        hbox.addSpacing(15)
        hbox.addWidget(self.label)
        hbox.addStretch()

        self.setLayout(hbox)

    def change_value(self, value):
        self.label.setText(str(value))
        self.value = int(value)
        self.changed.emit(int(value))
