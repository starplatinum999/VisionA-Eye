import sys
import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend instead of Qt5Agg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import Counter, defaultdict
import random

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QFrame, QGridLayout,
    QTabWidget, QScrollArea, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QStackedWidget,
    QRadioButton, QButtonGroup, QCheckBox, QDateEdit,
    QTimeEdit, QDialog, QSlider, QToolTip, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QSize, Signal, QTimer, QDateTime, QDate, QTime
from PySide6.QtGui import QFont, QColor, QPalette, QPixmap, QIcon, QCursor, QBrush, QLinearGradient, QPainter, QPainterPath

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding charts in the Qt application."""
    def __init__(self, width=8, height=5, dpi=100):
        # Create figure with light background
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#F5F7FA')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#FFFFFF')
        
        # Set text colors to dark
        self.axes.tick_params(colors='#1F2937', labelsize=9)
        self.axes.xaxis.label.set_color('#1F2937')
        self.axes.yaxis.label.set_color('#1F2937')
        self.axes.title.set_color('#1F2937')
        
        # Set spines (border) colors
        for spine in self.axes.spines.values():
            spine.set_color('#D1D5DB')
        
        super(MplCanvas, self).__init__(self.fig)
        self.setStyleSheet("background-color: #F5F7FA;")
        
        # Set minimum size for better visibility
        self.setMinimumHeight(400)

class EventBarChart(MplCanvas):
    """Bar chart showing event counts by type."""
    def __init__(self, *args, **kwargs):
        super(EventBarChart, self).__init__(*args, **kwargs)
        # Set larger figure size
        self.fig.set_size_inches(9, 6)
        self.setMinimumHeight(450)
    
    def update_chart(self, events, title="Events by Type"):
        """Update the chart with the provided events data."""
        self.axes.clear()
        
        if not events:
            self.axes.set_title("No Events to Display")
            self.draw()
            return
        
        # Count events by type
        event_types = [event['type'] for event in events]
        event_counts = Counter(event_types)
        
        # Sort by count (descending)
        sorted_events = dict(sorted(event_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Create gradient colors
        color_map = {
            'ROI Transition': '#0EA5E9',  # Sky blue
            'ROI Exit': '#3B82F6',        # Blue
            'Person at Shelf': '#8B5CF6',  # Purple
            'Item Pickup': '#10B981',     # Green
            'Potential Theft': '#EF4444',  # Red
            'Long Dwell Time': '#F59E0B'  # Amber
        }
        
        # Assign colors to event types, using fallbacks when needed
        colors = []
        for event_type in sorted_events.keys():
            if event_type in color_map:
                colors.append(color_map[event_type])
            else:
                # Generate a random color if not in our map
                colors.append(f"#{random.randint(0, 0xFFFFFF):06x}")
        
        # Create bars with gradient effect
        bars = self.axes.bar(
            sorted_events.keys(), 
            sorted_events.values(),
            color=colors,
            width=0.6,
            edgecolor='white',
            linewidth=1,
            alpha=0.8
        )
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.axes.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'{int(height)}',
                ha='center', 
                va='bottom',
                color='#1F2937',
                fontsize=9,
                fontweight='bold'
            )
        
        # Add labels and title
        self.axes.set_title(title, fontsize=12, fontweight='bold', color='#1F2937')
        self.axes.set_xlabel("Event Type", color='#1F2937', fontsize=10)
        self.axes.set_ylabel("Count", color='#1F2937', fontsize=10)
        
        # Rotate x labels for better readability
        plt.setp(self.axes.get_xticklabels(), rotation=30, ha='right', fontsize=9)
        
        # Add grid lines for better readability
        self.axes.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove top and right spines
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        
        # Adjust layout
        self.fig.tight_layout()
        self.draw()

class TimelineChart(MplCanvas):
    """Timeline chart showing events over time."""
    def __init__(self, *args, **kwargs):
        super(TimelineChart, self).__init__(*args, **kwargs)
        # Set a larger figure size for better visibility
        self.fig.set_size_inches(9, 6)
        # Ensure minimum height
        self.setMinimumHeight(450)
    
    def update_chart(self, events, title="Event Timeline"):
        """Update the timeline chart with event data."""
        self.axes.clear()
        
        if not events:
            self.axes.set_title("No Events to Display")
            self.draw()
            return
        
        # Group events by time
        events_by_time = defaultdict(list)
        for event in events:
            timestamp = event.get('timestamp', '')
            if timestamp:
                events_by_time[timestamp].append(event)
        
        # Prepare data
        timestamps = list(events_by_time.keys())
        
        # Sort timestamps
        timestamps.sort()
        
        # Setup plot
        # Define colors for different event types with improved palette
        event_colors = {
            'ROI Transition': '#0EA5E9',  # Sky blue
            'ROI Exit': '#3B82F6',        # Blue
            'Person at Shelf': '#8B5CF6',  # Purple
            'Item Pickup': '#10B981',     # Green
            'Potential Theft': '#EF4444',  # Red
            'Long Dwell Time': '#F59E0B'  # Amber
        }
        
        # Count events by type for each timestamp
        data_by_type = {}
        for timestamp, events_list in events_by_time.items():
            for event in events_list:
                event_type = event['type']
                if event_type not in data_by_type:
                    data_by_type[event_type] = {ts: 0 for ts in timestamps}
                data_by_type[event_type][timestamp] += 1
        
        # Create stacked bar chart
        bottom = np.zeros(len(timestamps))
        for event_type, data in data_by_type.items():
            values = [data[ts] for ts in timestamps]
            self.axes.bar(
                timestamps, 
                values, 
                bottom=bottom,
                label=event_type,
                color=event_colors.get(event_type, '#555555'),
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5,
                width=0.7
            )
            bottom += np.array(values)
        
        # Add labels and title
        self.axes.set_title(title, fontsize=12, fontweight='bold', color='#1F2937')
        self.axes.set_xlabel("Time", color='#1F2937', fontsize=10)
        self.axes.set_ylabel("Number of Events", color='#1F2937', fontsize=10)
        
        # Rotate x labels
        plt.setp(self.axes.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        
        # Add grid lines
        self.axes.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add legend with better styling
        legend = self.axes.legend(
            loc='upper right', 
            facecolor='#FFFFFF', 
            edgecolor='#D1D5DB',
            fontsize=8,
            framealpha=0.9,
            fancybox=True
        )
        
        # Remove top and right spines
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        
        # Adjust layout
        self.fig.tight_layout()
        self.draw()

class RoiActivityChart(MplCanvas):
    """Chart showing activity in each ROI."""
    def __init__(self, *args, **kwargs):
        super(RoiActivityChart, self).__init__(*args, **kwargs)
        # Set larger figure size
        self.fig.set_size_inches(9, 6)
        self.setMinimumHeight(450)
    
    def update_chart(self, events, title="ROI Activity"):
        """Update the ROI activity chart with event data."""
        self.axes.clear()
        
        if not events:
            self.axes.set_title("No Events to Display")
            self.draw()
            return
        
        # Count events by ROI
        roi_counts = defaultdict(int)
        
        for event in events:
            # Look for 'to_roi' first (for transitions)
            if 'to_roi' in event and event['to_roi']:
                roi_counts[event['to_roi']] += 1
            # Then 'roi_name' (for exits and other ROI-specific events)
            elif 'roi_name' in event and event['roi_name']:
                roi_counts[event['roi_name']] += 1
        
        # Handle empty data
        if not roi_counts:
            self.axes.set_title("No ROI Activity Found")
            self.draw()
            return
        
        # Sort ROIs by count (descending)
        sorted_rois = dict(sorted(roi_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Create pie chart with improved styling
        wedges, texts, autotexts = self.axes.pie(
            sorted_rois.values(), 
            labels=sorted_rois.keys(),
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
            textprops={'fontsize': 9, 'color': '#1F2937'},
            colors=plt.cm.tab10.colors[:len(sorted_rois)]
        )
        
        # Style the percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # Equal aspect ratio ensures the pie chart is circular
        self.axes.set_aspect('equal')
        
        # Add title
        self.axes.set_title(title, fontsize=12, fontweight='bold', color='#1F2937')
        
        # Add legend with counts
        legend_labels = [f"{roi} ({count})" for roi, count in sorted_rois.items()]
        self.axes.legend(
            wedges, 
            legend_labels, 
            title="ROIs", 
            loc="center left", 
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
            title_fontsize=9,
            frameon=True,
            fancybox=True,
            facecolor='#FFFFFF',
            edgecolor='#D1D5DB'
        )
        
        # Adjust layout
        self.fig.tight_layout()
        self.draw()

class EventTypeByRoiChart(MplCanvas):
    """Chart showing distribution of event types by ROI."""
    def __init__(self, *args, **kwargs):
        super(EventTypeByRoiChart, self).__init__(*args, **kwargs)
        # Set larger figure size
        self.fig.set_size_inches(9, 6)
        self.setMinimumHeight(450)
    
    def update_chart(self, events, title="Event Types by ROI"):
        """Update the chart showing event types by ROI."""
        self.axes.clear()
        
        if not events:
            self.axes.set_title("No Events to Display")
            self.draw()
            return
        
        # Build data structure to count event types by ROI
        roi_event_counts = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            event_type = event['type']
            roi_name = None
            
            # Extract ROI information from event
            if 'to_roi' in event and event['to_roi']:
                roi_name = event['to_roi']
            elif 'roi_name' in event and event['roi_name']:
                roi_name = event['roi_name']
            
            if roi_name:
                roi_event_counts[roi_name][event_type] += 1
        
        # Get unique ROIs and event types
        rois = list(roi_event_counts.keys())
        event_types = set()
        for roi_events in roi_event_counts.values():
            event_types.update(roi_events.keys())
        event_types = list(event_types)
        
        # Create data matrix
        data = np.zeros((len(rois), len(event_types)))
        for i, roi in enumerate(rois):
            for j, event_type in enumerate(event_types):
                data[i, j] = roi_event_counts[roi].get(event_type, 0)
        
        # Define visually appealing colors for different event types
        event_colors = {
            'ROI Transition': '#0EA5E9',  # Sky blue
            'ROI Exit': '#3B82F6',        # Blue
            'Person at Shelf': '#8B5CF6',  # Purple
            'Item Pickup': '#10B981',     # Green
            'Potential Theft': '#EF4444',  # Red
            'Long Dwell Time': '#F59E0B'  # Amber
        }
        
        # Create colors list for each event type, using fallbacks if needed
        colors = []
        for event_type in event_types:
            if event_type in event_colors:
                colors.append(event_colors[event_type])
            else:
                # Generate a color if not in our map
                colors.append(f"#{random.randint(0, 0xFFFFFF):06x}")
        
        # Create stacked bar chart with a bit of spacing between bars
        bottom = np.zeros(len(rois))
        for j, event_type in enumerate(event_types):
            self.axes.bar(
                rois, 
                data[:, j], 
                bottom=bottom,
                label=event_type,
                color=colors[j],
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5,
                width=0.75
            )
            bottom += data[:, j]
        
        # Add labels and title
        self.axes.set_title(title, fontsize=12, fontweight='bold', color='#1F2937')
        self.axes.set_xlabel("ROI Name", color='#1F2937', fontsize=10)
        self.axes.set_ylabel("Number of Events", color='#1F2937', fontsize=10)
        
        # Rotate x labels for better readability
        plt.setp(self.axes.get_xticklabels(), rotation=30, ha='right', fontsize=9)
        
        # Add grid lines for better readability
        self.axes.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add legend with better styling
        self.axes.legend(
            loc='upper right',
            fontsize=8,
            frameon=True,
            fancybox=True,
            facecolor='#FFFFFF',
            edgecolor='#D1D5DB',
            ncol=2
        )
        
        # Remove top and right spines
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        
        # Adjust layout
        self.fig.tight_layout()
        self.draw()

class HeatmapChart(MplCanvas):
    """Heatmap showing ROI activity over time periods."""
    def __init__(self, *args, **kwargs):
        super(HeatmapChart, self).__init__(*args, **kwargs)
        # Set larger figure size
        self.fig.set_size_inches(9, 6)
        self.setMinimumHeight(450)
    
    def update_chart(self, events, title="ROI Activity Heatmap"):
        """Create a heatmap showing ROI activity over time."""
        self.axes.clear()
        
        if not events:
            self.axes.set_title("No Events to Display")
            self.draw()
            return
        
        # Extract times from events and convert to hour
        time_roi_counts = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            timestamp = event.get('timestamp', '')
            if not timestamp:
                continue
                
            # Extract hour from timestamp (format: HH:MM:SS)
            try:
                hour = int(timestamp.split(':')[0])
                # Group hours into 2-hour blocks for better visualization
                hour_block = f"{hour//2 * 2:02d}-{(hour//2 * 2) + 2:02d}"
                
                # Get ROI name
                roi_name = None
                if 'to_roi' in event and event['to_roi']:
                    roi_name = event['to_roi']
                elif 'roi_name' in event and event['roi_name']:
                    roi_name = event['roi_name']
                
                if roi_name:
                    time_roi_counts[hour_block][roi_name] += 1
            except (ValueError, IndexError):
                continue
        
        # Handle empty data
        if not time_roi_counts:
            self.axes.set_title("No Time-based ROI Activity Found")
            self.draw()
            return
        
        # Create the heatmap data
        time_blocks = sorted(time_roi_counts.keys())
        unique_rois = set()
        for roi_counts in time_roi_counts.values():
            unique_rois.update(roi_counts.keys())
        rois = sorted(unique_rois)
        
        # Create matrix for heatmap
        data = np.zeros((len(rois), len(time_blocks)))
        for i, roi in enumerate(rois):
            for j, time_block in enumerate(time_blocks):
                data[i, j] = time_roi_counts[time_block].get(roi, 0)
        
        # Create heatmap with custom colormap
        cmap = LinearSegmentedColormap.from_list('BlueGreen', ['#EBF5FF', '#3B82F6', '#10B981'])
        heatmap = self.axes.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Add colorbar
        cbar = self.fig.colorbar(heatmap, ax=self.axes, shrink=0.8, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
        
        # Set labels and ticks
        self.axes.set_yticks(np.arange(len(rois)))
        self.axes.set_yticklabels(rois, fontsize=9)
        
        self.axes.set_xticks(np.arange(len(time_blocks)))
        self.axes.set_xticklabels(time_blocks, rotation=45, ha='right', fontsize=9)
        
        # Add grid
        self.axes.grid(False)
        
        # Add title
        self.axes.set_title(title, fontsize=12, fontweight='bold', color='#1F2937')
        self.axes.set_xlabel("Time Period (Hours)", fontsize=10, color='#1F2937')
        self.axes.set_ylabel("ROI Name", fontsize=10, color='#1F2937')
        
        # Add annotations
        for i in range(len(rois)):
            for j in range(len(time_blocks)):
                value = int(data[i, j])
                if value > 0:
                    text_color = 'white' if value > np.max(data) / 2 else 'black'
                    self.axes.text(j, i, str(value), ha="center", va="center", 
                                 color=text_color, fontsize=8, fontweight='bold')
        
        # Adjust layout
        self.fig.tight_layout()
        self.draw()

class DwellTimeChart(MplCanvas):
    """Chart showing dwell time distribution across ROIs."""
    def __init__(self, *args, **kwargs):
        super(DwellTimeChart, self).__init__(*args, **kwargs)
        # Set larger figure size
        self.fig.set_size_inches(9, 6)
        self.setMinimumHeight(450)
    
    def update_chart(self, events, title="Dwell Time by ROI"):
        """Update chart showing average dwell time by ROI."""
        self.axes.clear()
        
        if not events:
            self.axes.set_title("No Events to Display")
            self.draw()
            return
        
        # Calculate dwell time for each ROI
        roi_dwell_times = defaultdict(list)
        
        # Group events by track_id
        events_by_track = defaultdict(list)
        for event in events:
            track_id = event.get('track_id')
            if track_id is not None:
                events_by_track[track_id].append(event)
        
        # Calculate dwell time for each ROI visit
        for track_id, track_events in events_by_track.items():
            # Sort events by frame_idx
            track_events.sort(key=lambda e: e.get('frame_idx', 0))
            
            # Track ROI visits
            current_roi = None
            enter_frame = None
            
            for event in track_events:
                if event['type'] == 'ROI Transition' and 'to_roi' in event:
                    if current_roi is not None and enter_frame is not None:
                        # Calculate dwell time for previous ROI
                        exit_frame = event.get('frame_idx', 0)
                        if exit_frame > enter_frame:
                            dwell_frames = exit_frame - enter_frame
                            roi_dwell_times[current_roi].append(dwell_frames)
                    
                    # Start new ROI visit
                    current_roi = event['to_roi']
                    enter_frame = event.get('frame_idx', 0)
                
                elif event['type'] == 'ROI Exit' and 'roi_name' in event:
                    if current_roi == event['roi_name'] and enter_frame is not None:
                        # Calculate dwell time
                        exit_frame = event.get('frame_idx', 0)
                        if exit_frame > enter_frame:
                            dwell_frames = exit_frame - enter_frame
                            roi_dwell_times[current_roi].append(dwell_frames)
                        
                        # Reset tracking
                        current_roi = None
                        enter_frame = None
        
        # Calculate average dwell time for each ROI
        avg_dwell_times = {}
        for roi, dwell_times in roi_dwell_times.items():
            if dwell_times:
                avg_dwell_times[roi] = sum(dwell_times) / len(dwell_times)
        
        # If no dwell times calculated
        if not avg_dwell_times:
            self.axes.set_title("No Dwell Time Data Available")
            self.draw()
            return
        
        # Sort by average dwell time
        sorted_roi_dwell = dict(sorted(avg_dwell_times.items(), key=lambda x: x[1], reverse=True))
        
        # Improved horizontal bar chart
        bars = self.axes.barh(
            list(sorted_roi_dwell.keys()),
            list(sorted_roi_dwell.values()),
            color=sns.color_palette("viridis", len(sorted_roi_dwell)),
            alpha=0.8,
            edgecolor='white',
            linewidth=1,
            height=0.7
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            self.axes.text(
                width + max(sorted_roi_dwell.values()) * 0.02,
                bar.get_y() + bar.get_height()/2,
                f"{width:.1f}",
                va='center',
                fontsize=9,
                color='#1F2937'
            )
        
        # Add labels and title
        self.axes.set_title(title, fontsize=12, fontweight='bold', color='#1F2937')
        self.axes.set_xlabel("Average Frames", fontsize=10, color='#1F2937')
        self.axes.set_ylabel("ROI", fontsize=10, color='#1F2937')
        
        # Customize y-axis labels
        self.axes.tick_params(axis='y', labelsize=9)
        
        # Add grid lines for better readability
        self.axes.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Remove top and right spines
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        
        # Adjust layout
        self.fig.tight_layout()
        self.draw()

class FootfallAnalysisChart(MplCanvas):
    """Chart showing footfall (number of people) over time periods."""
    def __init__(self, *args, **kwargs):
        super(FootfallAnalysisChart, self).__init__(*args, **kwargs)
        # Set a larger figure size to increase height
        self.fig.set_size_inches(10, 6)
        
        # Add a secondary axis for annotations
        self.ax2 = self.axes.twinx()
        self.ax2.set_visible(False)  # Initially hidden, only used for peak markers
    
    def update_chart(self, events, time_range="Hourly", title="Footfall Analysis"):
        """
        Update chart showing people count over time.
        
        Args:
            events: List of event dictionaries
            time_range: Hourly, Daily, Weekly, or Monthly aggregation
            title: Chart title
        """
        self.axes.clear()
        self.ax2.clear()
        
        if not events:
            self.axes.set_title("No Events to Display")
            self.draw()
            return
        
        # Filter to only include person-related events
        person_events = []
        for event in events:
            if (event.get('type') == 'Person at Shelf' or 
                'person' in event.get('description', '').lower() or
                (event.get('type') == 'ROI Transition' and event.get('class_id') == 0)):
                person_events.append(event)
        
        if not person_events:
            self.axes.set_title("No Person Data Available")
            self.draw()
            return
        
        # Group events by time according to selected range
        footfall_by_time = defaultdict(int)
        
        for event in person_events:
            timestamp = event.get('timestamp', '')
            if not timestamp:
                continue
                
            # Extract time components for grouping
            try:
                # Basic timestamp format: "HH:MM:SS"
                hour, minute, second = map(int, timestamp.split(':'))
                
                # For testing - in real implementation these should be generated from actual timestamps
                # Would use frame_idx and fps to calculate relative times, or use real datetime objects
                # This is simplified for visualization purposes
                
                if time_range == "Hourly":
                    # Group by hour
                    time_key = f"{hour:02d}:00"
                elif time_range == "Daily":
                    # In real implementation, would use actual date
                    # For demo, create artificial days from hours
                    day = hour % 24  # Simulate different days
                    time_key = f"Day {day+1}"
                elif time_range == "Weekly":
                    # Simulate weeks from hours
                    week = (hour % 24) // 7
                    time_key = f"Week {week+1}"
                elif time_range == "Monthly":
                    # Simulate months from hours
                    month = (hour % 24) // 8
                    time_key = f"Month {month+1}"
                else:
                    # Default hourly
                    time_key = f"{hour:02d}:00"
                
                footfall_by_time[time_key] += 1
            
            except (ValueError, IndexError):
                continue
        
        # Sort time keys appropriately
        time_keys = sorted(footfall_by_time.keys())
        footfall_counts = [footfall_by_time[k] for k in time_keys]
        
        # Create a more visually appealing bar chart
        bars = self.axes.bar(
            time_keys,
            footfall_counts,
            width=0.7,
            color='#3B82F6',  # Blue
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        
        # Find peak times (local maxima)
        peak_indices = []
        for i in range(1, len(footfall_counts)-1):
            if footfall_counts[i] > footfall_counts[i-1] and footfall_counts[i] > footfall_counts[i+1]:
                peak_indices.append(i)
        
        # Additionally, check first and last elements
        if len(footfall_counts) > 1:
            if footfall_counts[0] > footfall_counts[1]:
                peak_indices.append(0)
            if footfall_counts[-1] > footfall_counts[-2]:
                peak_indices.append(len(footfall_counts)-1)
        
        # Highlight peak times 
        for idx in peak_indices:
            bar = bars[idx]
            bar.set_color('#10B981')  # Green for peak times
            bar.set_edgecolor('white')
            bar.set_linewidth(1.5)
            
            # Add a "PEAK" annotation above the bar
            height = bar.get_height()
            self.axes.annotate('PEAK',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 12),  # 12 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10,
                color='#10B981',
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='#10B981', alpha=0.8))
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add labels to bars with values
                self.axes.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + max(footfall_counts) * 0.02,
                    f'{int(height)}',
                    ha='center', 
                    va='bottom',
                    color='#1F2937',
                    fontsize=10,
                    fontweight='bold'
                )
        
        # Add labels and title
        self.axes.set_title(f"{title} ({time_range})", fontsize=14, fontweight='bold', color='#1F2937')
        self.axes.set_xlabel("Time Period", color='#1F2937', fontsize=12)
        self.axes.set_ylabel("Number of People", color='#1F2937', fontsize=12)
        
        # Rotate x labels if there are more than 6 time periods
        if len(time_keys) > 6:
            plt.setp(self.axes.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        else:
            plt.setp(self.axes.get_xticklabels(), fontsize=10)
        
        plt.setp(self.axes.get_yticklabels(), fontsize=10)
        
        # Add grid lines for better readability
        self.axes.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove top and right spines
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        
        # Add summary statistics
        if footfall_counts:
            avg_footfall = sum(footfall_counts) / len(footfall_counts)
            max_footfall = max(footfall_counts)
            max_time = time_keys[footfall_counts.index(max_footfall)]
            
            stats_text = f"Peak Time: {max_time} ({max_footfall} people)\nAverage: {avg_footfall:.1f} people per period"
            self.axes.text(
                0.02, 0.97, stats_text,
                transform=self.axes.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#E5E7EB')
            )
        
        # Adjust layout
        self.fig.tight_layout()
        self.draw()

class AnalyticsDashboard(QWidget):
    """Analytics dashboard widget showing statistics and visualizations for events."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F7FA;
                color: #1F2937;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
        """)
        
        # Data
        self.events = []
        
        # Set up the UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(24, 24, 24, 24)
        self.layout.setSpacing(24)
        self.setLayout(self.layout)
        
        # Header section
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Dashboard title
        title_label = QLabel("Analytics Dashboard")
        title_label.setFont(QFont("-apple-system", 22, QFont.DemiBold))
        title_label.setStyleSheet("color: #111827; padding: 0; margin-bottom: 12px; letter-spacing: -0.5px;")
        header_layout.addWidget(title_label, 1)
        
        # Filter options
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(16)
        
        time_label = QLabel("Time Range:")
        time_label.setStyleSheet("color: #4B5563; font-size: 14px; font-weight: 500;")
        filter_layout.addWidget(time_label)
        
        self.time_filter = QComboBox()
        self.time_filter.addItems(["All Time", "Last Hour", "Last 12 Hours", "Last 24 Hours"])
        self.time_filter.setStyleSheet("""
            QComboBox {
                background-color: #FFFFFF;
                color: #1F2937;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 140px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox:hover {
                border-color: #9CA3AF;
            }
            QComboBox:focus {
                border-color: #3B82F6;
                border-width: 2px;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #1F2937;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                selection-background-color: #EFF6FF;
                selection-color: #2563EB;
            }
        """)
        self.time_filter.currentIndexChanged.connect(self.update_dashboard)
        filter_layout.addWidget(self.time_filter)
        
        event_label = QLabel("Event Type:")
        event_label.setStyleSheet("color: #4B5563; font-size: 14px; font-weight: 500;")
        filter_layout.addWidget(event_label)
        
        self.event_filter = QComboBox()
        self.event_filter.addItem("All Events")
        self.event_filter.setStyleSheet("""
            QComboBox {
                background-color: #FFFFFF;
                color: #1F2937;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 140px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox:hover {
                border-color: #9CA3AF;
            }
            QComboBox:focus {
                border-color: #3B82F6;
                border-width: 2px;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #1F2937;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                selection-background-color: #EFF6FF;
                selection-color: #2563EB;
            }
        """)
        self.event_filter.currentIndexChanged.connect(self.update_dashboard)
        filter_layout.addWidget(self.event_filter)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #4F86F7, stop:1 #3B72D9);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 14px;
                min-width: 90px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #5D93FF, stop:1 #4B82E9);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #3A67BA, stop:1 #2B57A9);
            }
        """)
        self.refresh_button.clicked.connect(self.update_dashboard)
        filter_layout.addWidget(self.refresh_button)
        
        # Export button
        self.export_button = QPushButton("Export Data")
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        filter_layout.addWidget(self.export_button)
        
        header_layout.addLayout(filter_layout)
        self.layout.addLayout(header_layout)
        
        # Stats summary section
        stats_section = QWidget()
        stats_section.setStyleSheet("background-color: transparent;")
        stats_layout = QHBoxLayout(stats_section)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(16)
        
        # Create stat cards
        self.stat_cards = [
            {"id": "total_events", "title": "Total Events", "value": "0", "icon": "📊", "subtitle": "Total events detected", "color": "#3B82F6"},
            {"id": "unique_tracks", "title": "Unique Objects", "value": "0", "icon": "🔍", "subtitle": "Distinct objects tracked", "color": "#8B5CF6"},
            {"id": "person_count", "title": "Person Count", "value": "0", "icon": "👤", "subtitle": "People detected in frame", "color": "#10B981"},
            {"id": "potential_issues", "title": "Potential Issues", "value": "0", "icon": "⚠️", "subtitle": "Suspicious activities", "color": "#EF4444"}
        ]
        
        for card in self.stat_cards:
            card_widget = QFrame()
            # Create shadow effect
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(15)
            shadow.setColor(QColor(0, 0, 0, 25))
            shadow.setOffset(0, 2)
            card_widget.setGraphicsEffect(shadow)
            
            card_widget.setStyleSheet(f"""
                background-color: #FFFFFF;
                border-radius: 12px;
                border-left: 4px solid {card["color"]};
            """)
            
            card_layout = QVBoxLayout()
            card_layout.setContentsMargins(20, 16, 20, 16)
            card_layout.setSpacing(6)
            
            # Title row
            title_row = QHBoxLayout()
            title_label = QLabel(card["title"])
            title_label.setStyleSheet("color: #4B5563; font-size: 14px; font-weight: 500;")
            title_row.addWidget(title_label)
            title_row.addStretch()
            
            icon_label = QLabel(card["icon"])
            icon_label.setStyleSheet(f"font-size: 20px; color: {card['color']};")
            title_row.addWidget(icon_label)
            
            card_layout.addLayout(title_row)
            
            # Value
            value_label = QLabel(card["value"])
            value_label.setObjectName(f"{card['id']}_value")
            value_label.setStyleSheet("""
                font-size: 28px;
                font-weight: 700;
                color: #111827;
                margin-top: 5px;
            """)
            card_layout.addWidget(value_label)
            
            # Subtitle
            subtitle_label = QLabel(card["subtitle"])
            subtitle_label.setStyleSheet("color: #6B7280; font-size: 13px;")
            card_layout.addWidget(subtitle_label)
            
            card_widget.setLayout(card_layout)
            stats_layout.addWidget(card_widget)
        
        self.layout.addWidget(stats_section)
        
        # Main dashboard content using tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #E4E7EB;
                background-color: #FFFFFF;
                border-radius: 8px;
                padding: 10px;
            }
            QTabBar::tab {
                background-color: #F5F7FA;
                color: #4B5563;
                border: 1px solid #E4E7EB;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 8px 16px;
                margin-right: 2px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                color: #3B82F6;
                border-bottom: 2px solid #3B82F6;
            }
            QTabBar::tab:hover:!selected {
                background-color: #EFF6FF;
                color: #2563EB;
            }
        """)
        
        # Overview tab
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setContentsMargins(10, 15, 10, 10)
        overview_layout.setSpacing(20)
        
        # Create grid layout for charts in overview tab
        charts_grid = QGridLayout()
        charts_grid.setSpacing(20)
        
        # Event type chart - top left
        event_type_frame = QFrame()
        event_type_frame.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E4E7EB;
            padding: 10px;
        """)
        event_type_layout = QVBoxLayout(event_type_frame)
        event_type_layout.setContentsMargins(10, 10, 10, 10)
        
        event_type_title = QLabel("Events by Type")
        event_type_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #111827;")
        event_type_layout.addWidget(event_type_title)
        
        self.event_type_chart = EventBarChart()
        event_type_layout.addWidget(self.event_type_chart)
        
        charts_grid.addWidget(event_type_frame, 0, 0)
        
        # ROI activity chart - top right
        roi_activity_frame = QFrame()
        roi_activity_frame.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E4E7EB;
            padding: 10px;
        """)
        roi_activity_layout = QVBoxLayout(roi_activity_frame)
        roi_activity_layout.setContentsMargins(10, 10, 10, 10)
        
        roi_activity_title = QLabel("ROI Activity Distribution")
        roi_activity_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #111827;")
        roi_activity_layout.addWidget(roi_activity_title)
        
        self.roi_activity_chart = RoiActivityChart()
        roi_activity_layout.addWidget(self.roi_activity_chart)
        
        charts_grid.addWidget(roi_activity_frame, 0, 1)
        
        # Timeline chart - bottom left
        timeline_frame = QFrame()
        timeline_frame.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E4E7EB;
            padding: 10px;
        """)
        timeline_layout = QVBoxLayout(timeline_frame)
        timeline_layout.setContentsMargins(10, 10, 10, 10)
        
        timeline_title = QLabel("Event Timeline")
        timeline_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #111827;")
        timeline_layout.addWidget(timeline_title)
        
        self.timeline_chart = TimelineChart()
        timeline_layout.addWidget(self.timeline_chart)
        
        charts_grid.addWidget(timeline_frame, 1, 0)
        
        # Event type by ROI chart - bottom right
        event_roi_frame = QFrame()
        event_roi_frame.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E4E7EB;
            padding: 10px;
        """)
        event_roi_layout = QVBoxLayout(event_roi_frame)
        event_roi_layout.setContentsMargins(10, 10, 10, 10)
        
        event_roi_title = QLabel("Event Types by ROI")
        event_roi_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #111827;")
        event_roi_layout.addWidget(event_roi_title)
        
        self.event_roi_chart = EventTypeByRoiChart()
        event_roi_layout.addWidget(self.event_roi_chart)
        
        charts_grid.addWidget(event_roi_frame, 1, 1)
        
        overview_layout.addLayout(charts_grid)
        
        # Add overview tab to tab widget
        self.tab_widget.addTab(overview_tab, "Overview")
        
        # Advanced Analytics tab with additional charts
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        advanced_layout.setContentsMargins(10, 15, 10, 10)
        advanced_layout.setSpacing(20)
        
        # Create grid layout for charts in advanced tab
        advanced_grid = QGridLayout()
        advanced_grid.setSpacing(20)
        
        # ROI Heatmap - top left
        heatmap_frame = QFrame()
        heatmap_frame.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E4E7EB;
            padding: 10px;
        """)
        heatmap_layout = QVBoxLayout(heatmap_frame)
        heatmap_layout.setContentsMargins(10, 10, 10, 10)
        
        heatmap_title = QLabel("ROI Activity Heatmap")
        heatmap_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #111827;")
        heatmap_layout.addWidget(heatmap_title)
        
        self.heatmap_chart = HeatmapChart()
        heatmap_layout.addWidget(self.heatmap_chart)
        
        advanced_grid.addWidget(heatmap_frame, 0, 0)
        
        # Dwell time chart - top right
        dwell_frame = QFrame()
        dwell_frame.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E4E7EB;
            padding: 10px;
        """)
        dwell_layout = QVBoxLayout(dwell_frame)
        dwell_layout.setContentsMargins(10, 10, 10, 10)
        
        dwell_title = QLabel("Dwell Time by ROI")
        dwell_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #111827;")
        dwell_layout.addWidget(dwell_title)
        
        self.dwell_chart = DwellTimeChart()
        dwell_layout.addWidget(self.dwell_chart)
        
        advanced_grid.addWidget(dwell_frame, 0, 1)
        
        # Add grid layout to advanced tab
        advanced_layout.addLayout(advanced_grid)
        
        # Add advanced tab to tab widget
        self.tab_widget.addTab(advanced_tab, "Advanced Analytics")
        
        # Footfall Analysis tab
        footfall_tab = QWidget()
        footfall_layout = QVBoxLayout(footfall_tab)
        footfall_layout.setContentsMargins(15, 15, 15, 15)
        footfall_layout.setSpacing(20)
        
        # Controls section for footfall analysis
        controls_frame = QFrame()
        controls_frame.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E4E7EB;
            padding: 10px;
        """)
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(15, 10, 15, 10)
        
        # Time range selector
        time_range_label = QLabel("Time Aggregation:")
        time_range_label.setStyleSheet("font-weight: 500; color: #4B5563; min-width: 120px;")
        controls_layout.addWidget(time_range_label)
        
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["Hourly", "Daily", "Weekly", "Monthly"])
        self.time_range_combo.setCurrentIndex(0)
        self.time_range_combo.setStyleSheet("""
            QComboBox {
                background-color: #FFFFFF;
                color: #1F2937;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 120px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox:hover {
                border-color: #9CA3AF;
            }
        """)
        self.time_range_combo.currentIndexChanged.connect(self.update_footfall_chart)
        controls_layout.addWidget(self.time_range_combo)
        
        # Date range selector for future implementation
        controls_layout.addStretch(1)
        
        date_range_label = QLabel("Data Source:")
        date_range_label.setStyleSheet("font-weight: 500; color: #4B5563; margin-left: 20px; min-width: 100px;")
        controls_layout.addWidget(date_range_label)
        
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["All Data", "Current Session", "Selected Timeframe"])
        self.data_source_combo.setStyleSheet("""
            QComboBox {
                background-color: #FFFFFF;
                color: #1F2937;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 150px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox:hover {
                border-color: #9CA3AF;
            }
        """)
        self.data_source_combo.currentIndexChanged.connect(self.update_footfall_chart)
        controls_layout.addWidget(self.data_source_combo)
        
        # Apply button
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #4F86F7, stop:1 #3B72D9);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #5D93FF, stop:1 #4B82E9);
            }
        """)
        self.apply_btn.clicked.connect(self.update_footfall_chart)
        controls_layout.addWidget(self.apply_btn)
        
        footfall_layout.addWidget(controls_frame)
        
        # Footfall chart with increased height
        footfall_chart_frame = QFrame()
        footfall_chart_frame.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E4E7EB;
            padding: 15px;
        """)
        footfall_chart_layout = QVBoxLayout(footfall_chart_frame)
        footfall_chart_layout.setContentsMargins(10, 15, 10, 10)
        
        # Title
        footfall_title = QLabel("People Traffic Analysis")
        footfall_title.setStyleSheet("font-size: 18px; font-weight: 600; color: #111827; margin-bottom: 10px;")
        footfall_chart_layout.addWidget(footfall_title)
        
        # Description
        footfall_desc = QLabel("Analyze when your location experiences the highest visitor traffic. Peaks are highlighted in green.")
        footfall_desc.setStyleSheet("color: #6B7280; font-size: 14px; margin-bottom: 15px;")
        footfall_desc.setWordWrap(True)
        footfall_chart_layout.addWidget(footfall_desc)
        
        # The chart
        self.footfall_chart = FootfallAnalysisChart()
        footfall_chart_layout.addWidget(self.footfall_chart)
        
        footfall_layout.addWidget(footfall_chart_frame, 1)  # Give it a stretch factor of 1
        
        # Add insights section
        insights_frame = QFrame()
        insights_frame.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E4E7EB;
            padding: 15px;
        """)
        insights_layout = QVBoxLayout(insights_frame)
        
        insights_title = QLabel("Traffic Insights")
        insights_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #111827;")
        insights_layout.addWidget(insights_title)
        
        self.insights_content = QLabel(
            "• Analyze when your foot traffic is highest to optimize staffing and operations\n"
            "• Green bars indicate peak times with the highest visitor counts\n"
            "• Compare different time periods to identify patterns and trends\n"
            "• Use these insights for scheduling, marketing campaigns, and resource allocation"
        )
        self.insights_content.setStyleSheet("color: #4B5563; font-size: 14px; line-height: 1.6;")
        self.insights_content.setWordWrap(True)
        insights_layout.addWidget(self.insights_content)
        
        footfall_layout.addWidget(insights_frame)
        
        # Add tab to widget
        self.tab_widget.addTab(footfall_tab, "Footfall Analysis")
        
        # Event Log tab
        events_tab = QWidget()
        events_layout = QVBoxLayout(events_tab)
        events_layout.setContentsMargins(10, 15, 10, 10)
        
        # Event table
        self.event_table = QTableWidget()
        self.event_table.setColumnCount(5)
        self.event_table.setHorizontalHeaderLabels(["Time", "Event Type", "Description", "Location", "Needs Attention"])
        self.event_table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: #F9FAFB;
                color: #111827;
                padding: 8px;
                border: 1px solid #E4E7EB;
                font-weight: 600;
            }
        """)
        self.event_table.setStyleSheet("""
            QTableWidget {
                background-color: #FFFFFF;
                border-radius: 8px;
                border: 1px solid #E4E7EB;
                gridline-color: #F3F4F6;
                outline: none;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #F3F4F6;
            }
            QTableWidget::item:selected {
                background-color: #EFF6FF;
                color: #2563EB;
            }
        """)
        
        # Set column widths
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.event_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        
        events_layout.addWidget(self.event_table)
        
        self.tab_widget.addTab(events_tab, "Event Log")
        
        # Add tab widget to main layout
        self.layout.addWidget(self.tab_widget, 1)  # stretch factor of 1
        
        # Connect tab changed signal
        self.tab_widget.currentChanged.connect(self.switch_tab)
        
        # Update the dashboard initially
        QTimer.singleShot(100, self.update_dashboard)
    
    def switch_tab(self, index):
        """Switch between chart tabs."""
        self.tab_widget.setCurrentIndex(index)
    
    def set_events(self, events):
        """Set the events data and update the dashboard."""
        self.events = events
        
        # Update event filter options
        self.update_event_filter_options()
        
        # Update the dashboard with the new data
        self.update_dashboard()
        
        # Print debug info about events
        num_events = len(events) if events else 0
        print(f"Analytics Dashboard: Received {num_events} events")
        if events and num_events > 0:
            event_types = [e.get('type', 'Unknown') for e in events]
            type_counts = Counter(event_types)
            print(f"Event types: {dict(type_counts)}")
    
    def update_event_filter_options(self):
        """Update the event filter dropdown with available event types."""
        # Store current selection
        current_text = self.event_filter.currentText()
        
        # Clear and re-add "All Events"
        self.event_filter.clear()
        self.event_filter.addItem("All Events")
        
        # Get unique event types
        event_types = set()
        for event in self.events:
            if 'type' in event:
                event_types.add(event['type'])
        
        # Add event types to filter
        for event_type in sorted(event_types):
            self.event_filter.addItem(event_type)
        
        # Try to restore previous selection
        index = self.event_filter.findText(current_text)
        if index >= 0:
            self.event_filter.setCurrentIndex(index)
        else:
            # Default to "All Events" if previous selection not available
            self.event_filter.setCurrentIndex(0)
    
    def update_dashboard(self):
        """Update the entire dashboard with current data and filters."""
        # Get filtered events based on current selections
        filtered_events = self.get_filtered_events()
        
        # Update statistics
        self.update_statistics(filtered_events)
        
        # Update charts in Overview tab
        self.event_type_chart.update_chart(filtered_events)
        self.timeline_chart.update_chart(filtered_events)
        self.roi_activity_chart.update_chart(filtered_events)
        self.event_roi_chart.update_chart(filtered_events)
        
        # Update advanced charts
        self.heatmap_chart.update_chart(filtered_events)
        self.dwell_chart.update_chart(filtered_events)
        
        # Update event table
        self.update_event_table(filtered_events)
        
        # Update footfall chart
        self.update_footfall_chart()
    
    def get_filtered_events(self):
        """Filter events based on selected time range and event type."""
        # Filter by event type
        event_type_filter = self.event_filter.currentText()
        if event_type_filter == "All Events":
            type_filtered_events = self.events
        else:
            type_filtered_events = [
                event for event in self.events 
                if event.get('type', '') == event_type_filter
            ]
        
        # Filter by time range
        time_filter = self.time_filter.currentText()
        if time_filter == "All Time":
            return type_filtered_events
        
        # Set time threshold based on filter
        now = datetime.now()
        if time_filter == "Last Hour":
            threshold = now - timedelta(hours=1)
        elif time_filter == "Last 12 Hours":
            threshold = now - timedelta(hours=12)
        elif time_filter == "Last 24 Hours":
            threshold = now - timedelta(hours=24)
        else:
            return type_filtered_events
        
        # Convert threshold to string for comparison (HH:MM:SS format)
        threshold_str = threshold.strftime("%H:%M:%S")
        
        # Filter events by time
        time_filtered_events = []
        for event in type_filtered_events:
            timestamp = event.get('timestamp', '')
            if not timestamp:
                continue
            
            # Simple string comparison works for HH:MM:SS format
            # For more complex cases, would need to parse the timestamp
            if timestamp >= threshold_str:
                time_filtered_events.append(event)
        
        return time_filtered_events
    
    def update_statistics(self, events):
        """Update the statistics cards with event data."""
        # Set event count
        total_events = len(events)
        total_events_value = self.findChild(QLabel, "total_events_value")
        if total_events_value:
            total_events_value.setText(str(total_events))
        
        # Calculate unique tracks
        unique_tracks = set()
        person_count = 0
        potential_issues = 0
        
        for event in events:
            # Count unique track IDs
            if 'track_id' in event:
                unique_tracks.add(event['track_id'])
            
            # Look for person count (this is a simplification, actual implementation would depend on your data structure)
            if event.get('type') == 'Person at Shelf' or (
                'description' in event and 'person' in event['description'].lower()):
                person_count += 1
            
            # Count potential issues/alerts
            if event.get('type') in ['Potential Theft', 'Long Dwell Time'] or event.get('needs_reasoning', False):
                potential_issues += 1
        
        # Update unique tracks count
        unique_tracks_value = self.findChild(QLabel, "unique_tracks_value") 
        if unique_tracks_value:
            unique_tracks_value.setText(str(len(unique_tracks)))
        
        # Update person count
        person_count_value = self.findChild(QLabel, "person_count_value")
        if person_count_value:
            person_count_value.setText(str(person_count))
        
        # Update potential issues count
        potential_issues_value = self.findChild(QLabel, "potential_issues_value")
        if potential_issues_value:
            potential_issues_value.setText(str(potential_issues))
    
    def update_event_table(self, events):
        """Update the event table with filtered events."""
        self.event_table.setRowCount(0)
        
        if not events:
            return
        
        # Sort events by timestamp (recent first)
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', ''), reverse=True)
        
        self.event_table.setRowCount(len(sorted_events))
        
        # Define colors for different event types
        event_colors = {
            'ROI Transition': QColor('#EFF6FF'),  # Light blue
            'ROI Exit': QColor('#F5F3FF'),       # Light purple
            'Person at Shelf': QColor('#ECFDF5'), # Light green 
            'Item Pickup': QColor('#F0FDF4'),     # Light green
            'Potential Theft': QColor('#FEF2F2'), # Light red
            'Long Dwell Time': QColor('#FFFBEB')  # Light yellow
        }
        
        # Fill table data
        for i, event in enumerate(sorted_events):
            # Time
            time_item = QTableWidgetItem(event.get('timestamp', ''))
            self.event_table.setItem(i, 0, time_item)
            
            # Event Type
            type_item = QTableWidgetItem(event.get('type', ''))
            self.event_table.setItem(i, 1, type_item)
            
            # Description
            desc_item = QTableWidgetItem(event.get('description', ''))
            self.event_table.setItem(i, 2, desc_item)
            
            # Location (ROI)
            roi_name = ''
            if 'to_roi' in event:
                roi_name = event['to_roi']
            elif 'roi_name' in event:
                roi_name = event['roi_name']
            elif 'from_roi' in event and 'to_roi' in event:
                roi_name = f"{event['from_roi']} → {event['to_roi']}"
            
            location_item = QTableWidgetItem(roi_name)
            self.event_table.setItem(i, 3, location_item)
            
            # Needs Attention
            needs_attention = event.get('needs_reasoning', False) or event.get('type') == 'Potential Theft'
            attention_item = QTableWidgetItem("⚠️" if needs_attention else "")
            attention_item.setTextAlignment(Qt.AlignCenter)
            self.event_table.setItem(i, 4, attention_item)
            
            # Set background color based on event type
            if event.get('type') in event_colors:
                color = event_colors[event.get('type')]
                for col in range(5):
                    item = self.event_table.item(i, col)
                    if item:
                        item.setBackground(color)
                        
                        # Make important events bold
                        if needs_attention:
                            font = item.font()
                            font.setBold(True)
                            item.setFont(font)

    def update_footfall_chart(self):
        """Update the footfall analysis chart."""
        # Get selected time range
        time_range = self.time_range_combo.currentText()
        
        # Update the footfall chart with the new data
        self.footfall_chart.update_chart(self.events, time_range)
        
        # Update insights based on selected time range and data
        if self.events:
            self.insights_content.setText(
                f"• Peak traffic detected during {time_range.lower()} analysis\n"
                f"• Consider scheduling staff rotations based on traffic patterns\n"
                f"• {len(self.events)} total events analyzed in this dataset\n"
                f"• Use these insights for scheduling, marketing campaigns, and resource allocation"
            )

    def update_dashboard(self):
        """Update the entire dashboard with current data and filters."""
        # Get filtered events based on current selections
        filtered_events = self.get_filtered_events()
        
        # Update statistics
        self.update_statistics(filtered_events)
        
        # Update charts in Overview tab
        self.event_type_chart.update_chart(filtered_events)
        self.timeline_chart.update_chart(filtered_events)
        self.roi_activity_chart.update_chart(filtered_events)
        self.event_roi_chart.update_chart(filtered_events)
        
        # Update advanced charts
        self.heatmap_chart.update_chart(filtered_events)
        self.dwell_chart.update_chart(filtered_events)
        
        # Update event table
        self.update_event_table(filtered_events)
        
        # Update footfall chart
        self.update_footfall_chart() 