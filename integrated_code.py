import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider, QLabel, QDialog, QDialogButtonBox
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import griddata
import mne
from matplotlib import colors as mcolors
from matplotlib.collections import PathCollection

class EEGVisualizerWindow(QMainWindow):
    def __init__(self, evoked, epochs, channel_positions):
        super().__init__()
        self.setWindowTitle('Interactive EEG Visualization')
        self.setGeometry(100, 100, 1200, 800)

        self.evoked = evoked
        self.epochs = epochs
        self.channel_positions = channel_positions
        self.channel_names = list(channel_positions.keys())

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
        self.time_slider.setMaximum(len(evoked.times) - 1)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self.update_plot)
        layout.addWidget(self.time_slider)

        # Create label for time display
        self.time_label = QLabel()
        layout.addWidget(self.time_label)

        # Initialize EEG data and plots
        self.initialize_plot()

        # Track the color for each channel
        self.channel_colors = {}

        # Track the active electrodes and their data
        self.selected_electrodes_data = {}

    def initialize_plot(self):
        self.ax = self.fig.add_subplot(111)
        self.overlay_fig, self.overlay_ax = None, None
        self.plotted_channels = {}
        self.active_electrodes = set()
        self.color_map = self.create_color_map()

        self.legend = None  # For storing the current legend
        self.update_plot()

    def create_color_map(self):
        vibrant_colors = [
            "blue", "green", "cyan", "magenta", "orange", "purple", "yellow", 
            "lime", "pink", "teal", "gold", "red", "navy", "violet", "brown", 
            "orchid", "turquoise", "crimson"
        ]
        return {
            name: vibrant_colors[i % len(vibrant_colors)] for i, name in enumerate(self.channel_names)
        }

    def update_plot(self):
        time_index = self.time_slider.value()
        time_point = self.evoked.times[time_index]

        # Update time label
        self.time_label.setText(f'Time: {time_point:.3f} s')

        self.ax.clear()

        # Plot topomap
        mne.viz.plot_topomap(
            self.evoked.data[:, time_index], 
            self.evoked.info, 
            axes=self.ax, 
            show=False, 
            cmap='RdBu_r'
        )

        # Only create the legend once and place it in the same position
        if self.legend is None:
            sm = plt.cm.ScalarMappable(cmap='RdBu_r')
            sm.set_array([])  # Create empty array to use the colormap
            self.legend = self.fig.colorbar(sm, ax=self.ax, orientation='vertical', label='Electrode Color Legend')

        # Overlay clickable electrodes
        scatter = self.ax.scatter(
            [pos[0] for pos in self.channel_positions.values()],
            [pos[1] for pos in self.channel_positions.values()],
            s=100, c='red', alpha=0.6, picker=True
        )
        
        # Update the electrode colors based on whether they are selected or not
        scatter.set_facecolor([self.color_map[name] if name in self.active_electrodes else 'red' for name in self.channel_names])

        # Connect click event
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.draw()

    def on_pick(self, event):
        if isinstance(event.artist, PathCollection):
            ind = event.ind[0]
            channel_name = self.channel_names[ind]

            # Toggle electrode selection: If it's selected, remove it; otherwise, add it
            if channel_name in self.active_electrodes:
                self.active_electrodes.remove(channel_name)
                if channel_name in self.selected_electrodes_data:
                    del self.selected_electrodes_data[channel_name]  # Remove from data
            else:
                self.active_electrodes.add(channel_name)

            # Redraw the plot with updated electrode colors
            self.update_plot()

            # Get the channel data
            channel_idx = self.evoked.info['ch_names'].index(channel_name)
            channel_data = self.epochs.get_data()[:, channel_idx, :]
            times = self.epochs.times

            # Add the channel's data to the selected electrodes' data if not deselected
            if channel_name not in self.selected_electrodes_data and channel_name in self.active_electrodes:
                self.selected_electrodes_data[channel_name] = (times, channel_data)

            # Show the updated graph with all selected channels
            self.show_channel_popup()

    def show_channel_popup(self):
        # Create the popup window if it doesn't exist
        if not hasattr(self, 'popup') or not self.popup.isVisible():
            self.popup = QDialog(self)
            self.popup.setWindowTitle(f"Selected Channels")
            self.popup.setGeometry(100, 100, 800, 400)

            layout = QVBoxLayout(self.popup)

            # Create matplotlib figure for the popup
            self.popup_fig = Figure(figsize=(8, 4))
            self.popup_canvas = FigureCanvas(self.popup_fig)
            layout.addWidget(self.popup_canvas)

            self.popup_ax = self.popup_fig.add_subplot(111)
            self.popup_ax.set_xlabel('Time (s)')
            self.popup_ax.set_ylabel('Amplitude (µV)')
            self.popup_ax.grid()

            # Add buttons to close the popup
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(self.popup.accept)
            layout.addWidget(buttons)

            self.popup.show()

        # Clear the current plot
        self.popup_ax.clear()

        # Plot the data for each selected channel in a different color
        for channel_name, (times, channel_data) in self.selected_electrodes_data.items():
            # Assign each channel a distinct color
            color = self.color_map.get(channel_name, 'blue')
            self.popup_ax.plot(times, channel_data.mean(axis=0), label=channel_name, color=color)

            # Plot ±1 SD range
            self.popup_ax.fill_between(
                times,
                channel_data.mean(axis=0) - channel_data.std(axis=0),
                channel_data.mean(axis=0) + channel_data.std(axis=0),
                alpha=0.2, color=color
            )

        self.popup_ax.legend()
        self.popup_canvas.draw()


from mne_bids import BIDSPath, read_raw_bids
import mne

def load_eeg_data():
    # Define the BIDS root directory and BIDSPath
    bids_path = BIDSPath(
        root='',
        subject='02',  # subject ID
        session='01',
        task='letters',  # task name
        run='01',
        datatype='eeg', 
        suffix='eeg'  # file suffix, based on your filename
    )
    
    # Load the data using the BIDSPath
    raw = read_raw_bids(bids_path)

    raw.load_data()

    # Define a list of channels you are interested in analyzing

    channels_of_interest = ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

    # 'channels_of_interest' is a list of specific electrode names that you want to retain for analysis.

    # These typically correspond to electrodes positioned at various locations on the scalp (frontal, central, parietal, etc.).

    # Select only the channels of interest from the raw EEG data

    raw.pick_channels(channels_of_interest)

    # Apply bandpass filter
    raw.filter(1, 40, fir_design='firwin')

    # Extract events from annotations
    events, event_id = mne.events_from_annotations(raw)

    # Create epochs based on the events
    epochs = mne.Epochs(
        raw, events, event_id={'Stimulus/S 11': 11}, tmin=-0.2, tmax=0.5, baseline=(None, 0), preload=True
    )

    # Compute the average evoked response
    evoked = epochs.average()

    # Extract channel positions
    channel_positions = {
        ch['ch_name']: ch['loc'][:2] for ch in evoked.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH
    }

    return evoked, epochs, channel_positions


if __name__ == "__main__":
    evoked, epochs, channel_positions = load_eeg_data()
    app = QApplication(sys.argv)
    window = EEGVisualizerWindow(evoked, epochs, channel_positions)
    window.show()
    sys.exit(app.exec_())
