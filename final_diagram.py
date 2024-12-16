import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mne
from matplotlib.collections import PathCollection

class EEGVisualizerWindow(QMainWindow):
    def __init__(self, evoked, epochs, channel_positions, electrode_descriptions):
        super().__init__()
        self.setGeometry(100, 100, 1600, 800)

        self.evoked = evoked
        self.channel_positions = channel_positions
        self.channel_names = list(channel_positions.keys())
        self.electrode_descriptions = electrode_descriptions

        # Initialize EEG data and active electrodes
        self.selected_electrodes_data = {}
        self.active_electrodes = set()

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)  # Use a single column layout

        # Create a label for the title and center it in the layout
        title_label = QLabel("EEG Visualization For a Subject Reading Braille")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        main_layout.addWidget(title_label)

        # Create a horizontal layout to hold the graphs and the table
        graph_table_layout = QHBoxLayout()
        main_layout.addLayout(graph_table_layout)

        # Create left vertical layout for the topomap and controls
        left_layout = QVBoxLayout()
        graph_table_layout.addLayout(left_layout)

        # Create matplotlib figure for the topomap
        self.topomap_fig = Figure(figsize=(6, 6))
        self.topomap_canvas = FigureCanvas(self.topomap_fig)
        left_layout.addWidget(self.topomap_canvas)

        # Create time slider
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(len(evoked.times) - 1)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self.update_plot)
        left_layout.addWidget(self.time_slider)

        # Create label for time display
        self.time_label = QLabel()
        left_layout.addWidget(self.time_label)

        # Create middle layout for the graph
        middle_layout = QVBoxLayout()
        graph_table_layout.addLayout(middle_layout)

        # Create matplotlib figure for the graph
        self.graph_fig = Figure(figsize=(12, 6))
        self.graph_canvas = FigureCanvas(self.graph_fig)
        middle_layout.addWidget(self.graph_canvas)

        self.graph_ax = self.graph_fig.add_subplot(111)
        self.graph_ax.grid()

        # Create right layout for electrode description table
        right_layout = QVBoxLayout()
        graph_table_layout.addLayout(right_layout)

        self.description_table = QTableWidget()
        self.description_table.setColumnCount(2)
        self.description_table.setHorizontalHeaderLabels(["Electrode", "Description"])
        self.description_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.description_table)

        # Initialize EEG data and plots
        self.initialize_plot()

        # Track the color for each channel
        self.channel_colors = {}

    def initialize_plot(self):
        self.topomap_ax = self.topomap_fig.add_subplot(111)
        self.plotted_channels = {}
        self.color_map = self.create_color_map()
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

        # Update topomap
        self.topomap_ax.clear()
        mne.viz.plot_topomap(
            self.evoked.data[:, time_index],
            self.evoked.info,
            axes=self.topomap_ax,
            show=False,
            cmap='RdBu_r'
        )

        # Add a title to the topomap figure (subtitle explaining click functionality)
        self.topomap_ax.set_title("Click electrodes to view and compare responsibilities\nof that brain region and the voltage")

        # Overlay clickable electrodes
        scatter = self.topomap_ax.scatter(
            [pos[0] for pos in self.channel_positions.values()],
            [pos[1] for pos in self.channel_positions.values()],
            s=250, c='red', alpha=0.6, picker=True
        )
        
        scatter.set_facecolor([self.color_map[name] if name in self.active_electrodes else 'red' for name in self.channel_names])

        # Connect click event
        self.topomap_fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.topomap_canvas.draw()

        # Update the graph with selected channels
        self.update_graph()

    def update_graph(self):
        self.graph_ax.clear()

        if not self.selected_electrodes_data:
            self.graph_ax.text(0.5, 0.5, "No electrode selected.", transform=self.graph_ax.transAxes,
                               ha='center', va='center', fontsize=12, color='gray')

        for channel_name, (times, channel_data) in self.selected_electrodes_data.items():
            color = self.color_map.get(channel_name, 'blue')
            self.graph_ax.plot(times, channel_data, label=channel_name, color=color)

        # Add pre-stimulus and post-stimulus shading
        self.graph_ax.axvspan(self.evoked.times[0], 0, color='lightblue', alpha=0.3, label='Pre-Stimulus')
        self.graph_ax.axvspan(0, self.evoked.times[-1], color='lightgreen', alpha=0.2, label='Post-Stimulus')

        # Plot red vertical line to indicate current time
        current_time = self.evoked.times[self.time_slider.value()]
        self.graph_ax.axvline(x=current_time, color='gray', linestyle='--', lw=2)

        # Add a vertical line at time = 0 to indicate the stimulus event
        self.graph_ax.axvline(0, color='red', linestyle='--', label='Stimulus')

        # Add axis labels
        self.graph_ax.set_xlabel('Time (s)')
        self.graph_ax.set_ylabel('Amplitude (µV)')
        self.graph_ax.grid()

        self.graph_ax.legend()
        self.graph_canvas.draw()

    def on_pick(self, event):
        if isinstance(event.artist, PathCollection):
            ind = event.ind[0]
            channel_name = self.channel_names[ind]

            if channel_name in self.active_electrodes:
                self.active_electrodes.remove(channel_name)
                if channel_name in self.selected_electrodes_data:
                    del self.selected_electrodes_data[channel_name]  # Remove from data

                # Remove row from table
                rows_to_remove = []
                for row in range(self.description_table.rowCount()):
                    if self.description_table.item(row, 0).text() == channel_name:
                        rows_to_remove.append(row)
                for row in sorted(rows_to_remove, reverse=True):
                    self.description_table.removeRow(row)

            else:
                self.active_electrodes.add(channel_name)

                # Fetch description for the electrode
                description = self.electrode_descriptions.get(channel_name, ["No description available."])

                # Format description as a bulleted list
                description_bullets = "\n".join([f"\u2022 {line}" for line in description])

                # Add row to table
                row_position = self.description_table.rowCount()
                self.description_table.insertRow(row_position)
                self.description_table.setItem(row_position, 0, QTableWidgetItem(channel_name))

                description_item = QTableWidgetItem(description_bullets)
                description_item.setTextAlignment(Qt.AlignTop | Qt.AlignLeft)
                description_item.setFlags(description_item.flags() ^ Qt.ItemIsEditable)  # Make it non-editable

                self.description_table.setItem(row_position, 1, description_item)

                self.description_table.setWordWrap(True)
                self.description_table.resizeRowsToContents()

                channel_idx = self.evoked.info['ch_names'].index(channel_name)
                channel_data = self.evoked.data[channel_idx, :]
                times = self.evoked.times

                if channel_name not in self.selected_electrodes_data and channel_name in self.active_electrodes:
                    self.selected_electrodes_data[channel_name] = (times, channel_data)

            self.update_plot()

from mne_bids import BIDSPath, read_raw_bids

def load_eeg_data():
    bids_path = BIDSPath(
        root='',
        subject='02',
        session='01',
        task='letters',
        run='01',
        datatype='eeg', 
        suffix='eeg'
    )
    
    raw = read_raw_bids(bids_path)
    raw.load_data()

    channels_of_interest = ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    raw.pick_channels(channels_of_interest)
    raw.filter(1, 40, fir_design='firwin')

    events, event_id = mne.events_from_annotations(raw)

    epochs = mne.Epochs(
        raw, events, event_id={'Stimulus/S 11': 11}, tmin=-0.2, tmax=0.5, baseline=(None, 0), preload=True
    )

    evoked = epochs.average()

    channel_positions = {
        ch['ch_name']: ch['loc'][:2] for ch in evoked.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH
    }

    # Example electrode descriptions
    electrode_descriptions = {
        'Fp1': ["Thinking and problem-solving", "Working memory", "Understanding what’s happening around you"],
        'Fp2': ["Understanding feelings and emotions", "Thinking about social situations"],
        'F7': ["Speaking and forming words", "Remembering things you’ve heard", "Handling emotions"],
        'F3': ["A bit more specialized for language, logic, and reasoning", "Making plans and solving problems", "Learning and practicing movements", "Controlling where you look and what you pay attention to for your right eye"],
        'Fz': ["Staying focused", "Controlling thoughts and actions", "Planning and coordinating movements"],
        'F4': ["Right specific: regulating feelings and being aware of the surroundings", "Making plans and solving problems", "Learning and practicing movements", "Controlling where you look and what you pay attention to for your left eye"],
        'F8': ["Deciding risks and rewards", "Managing social and emotional behavior", "Handling emotions"],
        'T3': ["Listening and understanding language", "Storing new memories", "Managing emotions", "Processes sounds from the right ear"],
        'T4': ["Listening and understanding language", "Storing new memories", "Managing emotions", "Processes sounds from the left ear"],
        'T5': ["Understanding other people’s thoughts and feelings", "Recognizing social signals like gestures and facial expressions", "Combining what you see and hear"],
        'T6': ["Understanding social situations", "Processing language", "Remembering things", "Handling emotions"],
        'C3': ["Moving the right side of your body (especially your hand, arm, and face)", "Feeling touch, pressure, or temperature on the right side of your body"],
        'Cz': ["Planning and coordinating movements"],
        'C4': ["Moving the left side of your body (especially your hand, arm, and face)", "Feeling touch, pressure, or temperature on the left side of your body"],
        'P3': ["Understanding where your body is in space", "Paying attention to certain areas", "Solving math problems and reading"],
        'Pz': ["Self-reflection", "Visualizing mental images"],
        'P4': ["Understanding where things are around you", "Paying attention to the environment", "Solving math problems and reading"],
        'O1': ["Visual processing (right visual field)"],
        'O2': ["Visual processing (right visual field)"],
        'T7': ["Processing auditory information from the left ear", "Understanding language and speech on the left side", "Memory processing and recognition", "Visual-spatial processing in the left hemisphere"],
        'T8': ["Processing auditory information from the right ear", "Understanding language and speech on the right side", "Memory processing and recognition", "Visual-spatial processing in the right hemisphere"],
        'P7': ["Sensory processing on the left side of the body", "Spatial awareness and coordination on the left side", "Attention and focus", "Integration of sensory input from different sources on the left side"],
        'P8': ["Sensory processing on the right side of the body", "Spatial awareness and coordination on the right side", "Attention and focus", "Integration of sensory input from different sources on the right side"]
    }

    return evoked, epochs, channel_positions, electrode_descriptions

if __name__ == "__main__":
    app = QApplication(sys.argv)

    evoked, epochs, channel_positions, electrode_descriptions = load_eeg_data()
    window = EEGVisualizerWindow(evoked, epochs, channel_positions, electrode_descriptions)
    window.setGeometry(100, 100, 1920, 1080)
    window.show()

    sys.exit(app.exec_())
