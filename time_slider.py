import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, 
                           QVBoxLayout, QLabel, QSlider)
from PyQt6.QtCore import Qt

class TimeSlider(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Slider Demo")
        self.setGeometry(100, 100, 600, 150)  # x, y, width, height
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create time display label
        self.time_label = QLabel("Current Time: 0.000 seconds")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)
        
        # Create slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)  # 1000 steps for smooth movement
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(100)
        layout.addWidget(self.slider)
        
        # Connect slider to update function
        self.slider.valueChanged.connect(self.update_time_display)
        
    def update_time_display(self, value):
        # Convert slider value (0-1000) to seconds (0-10)
        time = (value / 1000) * 10  # 10 seconds total duration
        self.time_label.setText(f"Current Time: {time:.3f} seconds")
        # Print for testing - you'll replace this with your actual data update
        print(f"Time updated to: {time:.3f} seconds")

def main():
    app = QApplication(sys.argv)
    window = TimeSlider()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()