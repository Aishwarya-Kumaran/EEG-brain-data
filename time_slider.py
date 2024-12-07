import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import griddata

class EEGVisualizerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('EEG Brain Topology Visualization')
        self.setGeometry(100, 100, 1000, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        # Create time slider
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        self.time_slider.valueChanged.connect(self.update_plot)
        layout.addWidget(self.time_slider)
        
        # Initialize EEG channel positions (10-20 system)
        self.initialize_channel_positions()
        
        # Initialize sample data
        self.initialize_sample_data()
        
        # Create initial plot
        self.ax = self.fig.add_subplot(111)
        self.colorbar = None  # Initialize colorbar as None
        self.contour = None   # Initialize contour as None
        self.first_plot = True
        self.update_plot()

    def initialize_channel_positions(self):
        # Standard 10-20 system electrode positions (x, y coordinates)
        self.channel_positions = {
            'Fp1': (-0.2, 0.7), 'Fp2': (0.2, 0.7),
            'F7': (-0.5, 0.5), 'F3': (-0.3, 0.5), 'Fz': (0, 0.5), 'F4': (0.3, 0.5), 'F8': (0.5, 0.5),
            'T3': (-0.7, 0), 'C3': (-0.4, 0), 'Cz': (0, 0), 'C4': (0.4, 0), 'T4': (0.7, 0),
            'T5': (-0.5, -0.5), 'P3': (-0.3, -0.5), 'Pz': (0, -0.5), 'P4': (0.3, -0.5), 'T6': (0.5, -0.5),
            'O1': (-0.2, -0.7), 'O2': (0.2, -0.7)
        }
        
        # Extract x and y coordinates for interpolation
        self.x_coords = np.array([pos[0] for pos in self.channel_positions.values()])
        self.y_coords = np.array([pos[1] for pos in self.channel_positions.values()])

    def initialize_sample_data(self):
        # Create sample EEG data
        self.time_points = 100
        self.channels = len(self.channel_positions)
        self.data = np.random.randn(self.channels, self.time_points)
        
        # Create interpolation grid
        x_min, x_max = -0.8, 0.8
        y_min, y_max = -0.8, 0.8
        grid_size = 100
        self.xi = np.linspace(x_min, x_max, grid_size)
        self.yi = np.linspace(y_min, y_max, grid_size)
        self.Xi, self.Yi = np.meshgrid(self.xi, self.yi)
        
        # Update slider
        self.time_slider.setMaximum(self.time_points - 1)

    def create_head_mask(self):
        # Create circular mask for head shape
        center = (0, 0)
        radius = 0.7
        mask = (self.Xi - center[0])**2 + (self.Yi - center[1])**2 <= radius**2
        return mask

    def update_plot(self):
        time_index = self.time_slider.value()
        
        if not self.first_plot:
            # Clear only the contour plot, not the entire axis
            if self.contour:
                for coll in self.contour.collections:
                    coll.remove()
        else:
            self.ax.clear()
            self.first_plot = False
        
        # Get current time point data
        values = self.data[:, time_index]
        
        # Interpolate data
        zi = griddata((self.x_coords, self.y_coords), values, 
                     (self.Xi, self.Yi), method='cubic')
        
        # Apply head mask
        mask = self.create_head_mask()
        zi[~mask] = np.nan
        
        # Plot interpolated data
        self.contour = self.ax.contourf(self.Xi, self.Yi, zi, levels=20, 
                                      cmap='RdBu_r', extend='both')
        
        # Create colorbar only once
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(self.contour, ax=self.ax)
        else:
            self.colorbar.update_normal(self.contour)
        
        # Plot electrode positions
        if self.first_plot or True:  # Always redraw for now
            for name, pos in self.channel_positions.items():
                self.ax.plot(pos[0], pos[1], 'k.', markersize=10)
                self.ax.text(pos[0], pos[1], name, fontsize=8, 
                            ha='center', va='bottom')
            
            # Draw head outline
            circle = plt.Circle((0, 0), 0.7, fill=False, color='black')
            self.ax.add_artist(circle)
            
            # Draw nose
            nose = plt.Polygon([(0, 0.7), (-0.1, 0.8), (0.1, 0.8)], 
                              closed=True, fill=False, color='black')
            self.ax.add_artist(nose)
            
            # Set plot properties
            self.ax.set_xlim(-0.8, 0.8)
            self.ax.set_ylim(-0.8, 0.8)
            self.ax.set_aspect('equal')
            self.ax.axis('off')
        
        self.ax.set_title(f'EEG Topological Map - Time: {time_index}')
        
        # Redraw canvas
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = EEGVisualizerWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
