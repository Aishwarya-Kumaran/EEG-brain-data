import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.collections as collections

# Load BIDS data
bids_path = BIDSPath(
    root='Parkinsons mini/',
    subject='001',  # subject ID
    task='PassiveViewing',  # task name
    datatype='eeg', 
    suffix='eeg'  # file suffix, based on your filename
)

raw = read_raw_bids(bids_path, verbose=False)
raw.load_data()
raw.filter(1, 40, fir_design='firwin')

# Extract events and create epochs
events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id={'Stimulus/S 11': 11}, tmin=-0.2, tmax=0.5, baseline=(None, 0), preload=True)
evoked = epochs.average()

# Topomap plotting with clickable electrodes
times = [0.267, 0.5]
fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1, 0.05]})
ax1, ax2, colorbar_ax = axes
evoked.plot_topomap(times=times, ch_type='eeg', time_unit='s', axes=[ax1, ax2, colorbar_ax], colorbar=True, show=False)

# Extract channel positions and names
channel_pos = np.array([ch['loc'][:2] for ch in evoked.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH])
channel_names = [ch['ch_name'] for ch in evoked.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH]

# Create a vibrant color map for electrodes
vibrant_colors = [
    "blue", "green", "cyan", "magenta", "orange", "purple", "yellow", "lime", "pink", "teal",
    "gold", "red", "navy", "violet", "brown", "orchid", "turquoise", "crimson"
]
color_map = {name: vibrant_colors[i % len(vibrant_colors)] for i, name in enumerate(evoked.info['ch_names'])}

# Track which electrodes are active
active_electrodes = set()

# Persistent reference for the overlay plot and channel visibility tracking
overlay_fig, overlay_ax = None, None
plotted_channels = {}  # Dictionary to track lines for each channel

# Overlay scatter plots for clickable electrodes
scatters = []
for ax in [ax1, ax2]:
    scatter = ax.scatter(channel_pos[:, 0], channel_pos[:, 1], s=100, c='red', alpha=0.6, picker=True)
    scatters.append(scatter)

# Define the click event handler function
def on_pick(event):
    global overlay_fig, overlay_ax, plotted_channels, active_electrodes  # Use the persistent figure, axes, and tracking dictionary

    # Check if the picked object is one of the scatter points
    if isinstance(event.artist, collections.PathCollection):
        # Get the index of the clicked electrode from event
        ind = event.ind[0]
        channel_name = channel_names[ind]

        # Get the data for the clicked channel
        channel_idx = evoked.info['ch_names'].index(channel_name)
        channel_data = epochs.get_data()[:, channel_idx, :]  # Extract epochs data for this channel
        times = epochs.times

        # Initialize the overlay figure if it doesn't exist
        if overlay_fig is None or overlay_ax is None:
            overlay_fig, overlay_ax = plt.subplots(figsize=(8, 4))
            overlay_ax.set_title("Channel Data Comparison")
            overlay_ax.set_xlabel('Time (s)')
            overlay_ax.set_ylabel('Amplitude (µV)')
            overlay_ax.grid()

        # Toggle the channel's line on the graph
        if channel_name in plotted_channels:
            # Remove the channel's line and shaded area if already plotted
            line, shaded_area = plotted_channels.pop(channel_name)
            line.remove()
            shaded_area.remove()
            # Remove highlight from the topomap
            active_electrodes.remove(channel_name)
        else:
            # Plot the channel's data and add it to the tracking dictionary
            color = color_map.get(channel_name, 'gray')  # Default to gray if channel not in color_map
            line, = overlay_ax.plot(times, channel_data.mean(axis=0), label=f'{channel_name} (Avg)', color=color)
            shaded_area = overlay_ax.fill_between(
                times,
                channel_data.mean(axis=0) - channel_data.std(axis=0),
                channel_data.mean(axis=0) + channel_data.std(axis=0),
                alpha=0.2, color=color, label=f'{channel_name} (±1 SD)'
            )
            plotted_channels[channel_name] = (line, shaded_area)
            # Highlight the topomap electrode
            active_electrodes.add(channel_name)

        # Update scatter colors on topomap
        for scatter in scatters:
            colors = ['red' if name not in active_electrodes else color_map[name] for name in channel_names]
            scatter.set_facecolor(colors)

        # Update the legend and refresh the plot
        overlay_ax.legend()
        overlay_fig.canvas.draw()
        fig.canvas.draw_idle()

        # Show the overlay figure if it's the first time
        overlay_fig.show()

# Connect the click event handler
fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()
