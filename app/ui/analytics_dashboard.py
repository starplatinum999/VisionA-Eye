import sys
import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend instead of Qt5Agg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QFrame, QGridLayout,
    QTabWidget, QScrollArea, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QStackedWidget
)
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QFont, QColor, QPalette

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding charts in the Qt application."""
    def __init__(self, width=5, height=4, dpi=100):
        # Create figure with light background
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#F5F7FA')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#FFFFFF')
        
        # Set text colors to dark
        self.axes.tick_params(colors='#1F2937')
        self.axes.xaxis.label.set_color('#1F2937')
        self.axes.yaxis.label.set_color('#1F2937')
        self.axes.title.set_color('#1F2937')
        
        # Set spines (border) colors
        for spine in self.axes.spines.values():
            spine.set_color('#D1D5DB')
        
        super(MplCanvas, self).__init__(self.fig)
        self.setStyleSheet("background-color: #F5F7FA;")

class EventBarChart(MplCanvas):
    """Bar chart showing event counts by type."""
    def __init__(self, *args, **kwargs):
        super(EventBarChart, self).__init__(*args, **kwargs)
    
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
        
        # Create bars with custom colors
        colors = ['#3B82F6', '#93C5FD', '#22C55E', '#FACC15', '#EF4444', '#D1D5DB']
        bars = self.axes.bar(
            sorted_events.keys(), 
            sorted_events.values(),
            color=colors[:len(sorted_events)]
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
                color='#1F2937'
            )
        
        # Add labels and title
        self.axes.set_title(title, fontsize=12, color='#1F2937')
        self.axes.set_xlabel("Event Type", color='#1F2937')
        self.axes.set_ylabel("Count", color='#1F2937')
        
        # Rotate x labels for better readability
        plt.setp(self.axes.get_xticklabels(), rotation=30, ha='right')
        
        # Adjust layout
        self.fig.tight_layout()
        self.draw()

class TimelineChart(MplCanvas):
    """Timeline chart showing events over time."""
    def __init__(self, *args, **kwargs):
        super(TimelineChart, self).__init__(*args, **kwargs)
    
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
        counts = [len(events) for events in events_by_time.values()]
        
        # Setup plot
        # Define colors for different event types
        event_colors = {
            'ROI Transition': '#E4E7EB',
            'ROI Exit': '#D1D5DB',
            'Person at Shelf': '#3B82F6',
            'Item Pickup': '#93C5FD',
            'Potential Theft': '#EF4444',
            'Long Dwell Time': '#FACC15',
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
                color=event_colors.get(event_type, '#555555')
            )
            bottom += np.array(values)
        
        # Add labels and title
        self.axes.set_title(title, fontsize=12, color='#1F2937')
        self.axes.set_xlabel("Time", color='#1F2937')
        self.axes.set_ylabel("Number of Events", color='#1F2937')
        
        # Rotate x labels
        plt.setp(self.axes.get_xticklabels(), rotation=45, ha='right')
        
        # Add legend
        self.axes.legend(loc='upper right', facecolor='#FFFFFF', edgecolor='#D1D5DB')
        
        # Adjust layout
        self.fig.tight_layout()
        self.draw()

class RoiActivityChart(MplCanvas):
    """Chart showing activity in each ROI."""
    def __init__(self, *args, **kwargs):
        super(RoiActivityChart, self).__init__(*args, **kwargs)
    
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
        
        # Define colormap
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(sorted_rois))]
        
        # Create horizontal bar chart
        bars = self.axes.barh(
            list(sorted_rois.keys()),
            list(sorted_rois.values()),
            color=colors
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            self.axes.text(
                width + 0.1,
                bar.get_y() + bar.get_height()/2.,
                f'{int(width)}',
                va='center',
                color='#1F2937'
            )
        
        # Add labels and title
        self.axes.set_title(title, fontsize=12, color='#1F2937')
        self.axes.set_xlabel("Number of Events", color='#1F2937')
        self.axes.set_ylabel("ROI Name", color='#1F2937')
        
        # Adjust layout
        self.fig.tight_layout()
        self.draw()

class EventTypeByRoiChart(MplCanvas):
    """Chart showing the distribution of event types across ROIs."""
    def __init__(self, *args, **kwargs):
        super(EventTypeByRoiChart, self).__init__(*args, **kwargs)
    
    def update_chart(self, events, title="Event Types by ROI"):
        """Update the chart with event data."""
        self.axes.clear()
        
        if not events:
            self.axes.set_title("No Events to Display")
            self.draw()
            return
        
        # Build data structure for events by ROI and type
        roi_event_types = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            event_type = event['type']
            
            # Determine which ROI this event is associated with
            roi_name = None
            if 'to_roi' in event and event['to_roi']:
                roi_name = event['to_roi']
            elif 'roi_name' in event and event['roi_name']:
                roi_name = event['roi_name']
            
            if roi_name:
                roi_event_types[roi_name][event_type] += 1
        
        # Handle empty data
        if not roi_event_types:
            self.axes.set_title("No ROI-Event Data Found")
            self.draw()
            return
        
        # Prepare data for stacked bar chart
        rois = list(roi_event_types.keys())
        event_types = set()
        for roi_data in roi_event_types.values():
            event_types.update(roi_data.keys())
        event_types = list(event_types)
        
        # Colors for event types
        event_colors = {
            'ROI Transition': '#3a506b',
            'ROI Exit': '#343a40',
            'Person at Shelf': '#2e6f95',
            'Item Pickup': '#6b705c',
            'Potential Theft': '#9e2a2b',
            'Long Dwell Time': '#774936',
        }
        
        # Create a matrix for the data
        data = np.zeros((len(rois), len(event_types)))
        for i, roi in enumerate(rois):
            for j, event_type in enumerate(event_types):
                data[i, j] = roi_event_types[roi][event_type]
        
        # Create stacked bar chart
        bottom = np.zeros(len(rois))
        for j, event_type in enumerate(event_types):
            self.axes.bar(
                rois,
                data[:, j],
                bottom=bottom,
                label=event_type,
                color=event_colors.get(event_type, '#555555')
            )
            bottom += data[:, j]
        
        # Add labels and title
        self.axes.set_title(title, fontsize=12, color='#1F2937')
        self.axes.set_xlabel("ROI Name", color='#1F2937')
        self.axes.set_ylabel("Number of Events", color='#1F2937')
        
        # Rotate x labels for better readability
        plt.setp(self.axes.get_xticklabels(), rotation=30, ha='right')
        
        # Add legend
        self.axes.legend(loc='upper right', facecolor='#FFFFFF', edgecolor='#D1D5DB')
        
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
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 14px;
                min-width: 90px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        self.refresh_button.clicked.connect(self.update_dashboard)
        filter_layout.addWidget(self.refresh_button)
        
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
            {"id": "total_events", "title": "Total Events", "value": "0", "icon": "📊", "subtitle": "No events detected in the selected time range"},
            {"id": "unique_tracks", "title": "Unique Objects", "value": "0", "icon": "🔍", "subtitle": "No objects detected in the selected time range"},
            {"id": "person_count", "title": "Person Count", "value": "0", "icon": "👤", "subtitle": "No people detected in the selected time range"},
            {"id": "potential_issues", "title": "Potential Issues", "value": "0", "icon": "⚠️", "subtitle": "No issues detected in the selected time range"}
        ]
        
        for card in self.stat_cards:
            card_widget = QFrame()
            card_widget.setStyleSheet("""
                background-color: #FFFFFF;
                border-radius: 8px;
                border: 1px solid #E4E7EB;
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
            
            card_layout.addLayout(title_row)
            
            # Value
            value_label = QLabel(card["value"])
            value_label.setFont(QFont("-apple-system", 28, QFont.Bold))
            value_label.setStyleSheet("color: #111827; letter-spacing: -0.5px;")
            card["value_label"] = value_label
            card_layout.addWidget(value_label)
            
            # Subtitle
            subtitle_label = QLabel(card["subtitle"])
            subtitle_label.setStyleSheet("color: #6B7280; font-size: 13px;")
            subtitle_label.setWordWrap(True)
            card["subtitle_label"] = subtitle_label
            card_layout.addWidget(subtitle_label)
            
            card_widget.setLayout(card_layout)
            stats_layout.addWidget(card_widget)
        
        self.layout.addWidget(stats_section)
        
        # Charts section
        charts_section = QWidget()
        charts_layout = QVBoxLayout(charts_section)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(0)
        
        # Tab controls for charts
        tab_bar = QWidget()
        tab_bar.setStyleSheet("background-color: transparent; border-bottom: 1px solid #E5E7EB;")
        tab_layout = QHBoxLayout(tab_bar)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)
        
        self.tab_buttons = []
        tab_styles = """
            QPushButton {
                background-color: transparent;
                color: #6B7280;
                border: none;
                padding: 12px 20px;
                font-weight: 500;
                font-size: 15px;
                text-align: left;
                border-bottom: 2px solid transparent;
            }
            QPushButton:checked {
                color: #3B82F6;
                border-bottom: 2px solid #3B82F6;
                font-weight: 600;
            }
            QPushButton:hover:!checked {
                color: #4B5563;
                background-color: #F9FAFB;
            }
        """
        
        # Events by Type tab
        events_by_type_btn = QPushButton("Events by Type")
        events_by_type_btn.setCheckable(True)
        events_by_type_btn.setChecked(True)
        events_by_type_btn.setStyleSheet(tab_styles)
        events_by_type_btn.clicked.connect(lambda: self.switch_tab(0))
        tab_layout.addWidget(events_by_type_btn)
        self.tab_buttons.append(events_by_type_btn)
        
        # Event Timeline tab
        timeline_btn = QPushButton("Event Timeline")
        timeline_btn.setCheckable(True)
        timeline_btn.setStyleSheet(tab_styles)
        timeline_btn.clicked.connect(lambda: self.switch_tab(1))
        tab_layout.addWidget(timeline_btn)
        self.tab_buttons.append(timeline_btn)
        
        # ROI Activity tab
        roi_activity_btn = QPushButton("ROI Activity")
        roi_activity_btn.setCheckable(True)
        roi_activity_btn.setStyleSheet(tab_styles)
        roi_activity_btn.clicked.connect(lambda: self.switch_tab(2))
        tab_layout.addWidget(roi_activity_btn)
        self.tab_buttons.append(roi_activity_btn)
        
        # Events by ROI tab
        events_by_roi_btn = QPushButton("Events by ROI")
        events_by_roi_btn.setCheckable(True)
        events_by_roi_btn.setStyleSheet(tab_styles)
        events_by_roi_btn.clicked.connect(lambda: self.switch_tab(3))
        tab_layout.addWidget(events_by_roi_btn)
        self.tab_buttons.append(events_by_roi_btn)
        
        tab_layout.addStretch()
        charts_layout.addWidget(tab_bar)
        
        # Chart stack
        self.chart_stack = QStackedWidget()
        self.chart_stack.setStyleSheet("""
            background-color: #FFFFFF; 
            border: 1px solid #E4E7EB; 
            border-top: none; 
            border-radius: 0 0 8px 8px;
        """)
        
        # Chart panels
        chart_panel_style = "padding: 20px; background-color: #FFFFFF;"
        
        # Events by Type chart
        events_type_panel = QWidget()
        events_type_layout = QVBoxLayout(events_type_panel)
        events_type_layout.setContentsMargins(20, 20, 20, 20)
        self.event_type_chart = EventBarChart(width=6, height=4)
        events_type_layout.addWidget(self.event_type_chart)
        
        # Add "No data available" message for empty state
        self.event_type_empty = QLabel("No data available for the selected time range")
        self.event_type_empty.setAlignment(Qt.AlignCenter)
        self.event_type_empty.setStyleSheet("color: #6B7280; font-size: 15px; padding: 20px; background: transparent;")
        self.event_type_empty.setVisible(False)
        events_type_layout.addWidget(self.event_type_empty)
        
        self.chart_stack.addWidget(events_type_panel)
        
        # Timeline chart
        timeline_panel = QWidget()
        timeline_layout = QVBoxLayout(timeline_panel)
        timeline_layout.setContentsMargins(20, 20, 20, 20)
        self.timeline_chart = TimelineChart(width=6, height=4)
        timeline_layout.addWidget(self.timeline_chart)
        
        # Add "No data available" message for empty state
        self.timeline_empty = QLabel("No data available for the selected time range")
        self.timeline_empty.setAlignment(Qt.AlignCenter)
        self.timeline_empty.setStyleSheet("color: #6B7280; font-size: 15px; padding: 20px; background: transparent;")
        self.timeline_empty.setVisible(False)
        timeline_layout.addWidget(self.timeline_empty)
        
        self.chart_stack.addWidget(timeline_panel)
        
        # ROI activity chart
        roi_panel = QWidget()
        roi_layout = QVBoxLayout(roi_panel)
        roi_layout.setContentsMargins(20, 20, 20, 20)
        self.roi_chart = RoiActivityChart(width=6, height=4)
        roi_layout.addWidget(self.roi_chart)
        
        # Add "No data available" message for empty state
        self.roi_empty = QLabel("No data available for the selected time range")
        self.roi_empty.setAlignment(Qt.AlignCenter)
        self.roi_empty.setStyleSheet("color: #6B7280; font-size: 15px; padding: 20px; background: transparent;")
        self.roi_empty.setVisible(False)
        roi_layout.addWidget(self.roi_empty)
        
        self.chart_stack.addWidget(roi_panel)
        
        # Event types by ROI chart
        event_roi_panel = QWidget()
        event_roi_layout = QVBoxLayout(event_roi_panel)
        event_roi_layout.setContentsMargins(20, 20, 20, 20)
        self.event_roi_chart = EventTypeByRoiChart(width=6, height=4)
        event_roi_layout.addWidget(self.event_roi_chart)
        
        # Add "No data available" message for empty state
        self.event_roi_empty = QLabel("No data available for the selected time range")
        self.event_roi_empty.setAlignment(Qt.AlignCenter)
        self.event_roi_empty.setStyleSheet("color: #6B7280; font-size: 15px; padding: 20px; background: transparent;")
        self.event_roi_empty.setVisible(False)
        event_roi_layout.addWidget(self.event_roi_empty)
        
        self.chart_stack.addWidget(event_roi_panel)
        
        charts_layout.addWidget(self.chart_stack)
        self.layout.addWidget(charts_section)
        
        # Detailed event data section
        data_section = QWidget()
        data_layout = QVBoxLayout(data_section)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(12)
        
        # Section header
        data_header = QWidget()
        data_header_layout = QHBoxLayout(data_header)
        data_header_layout.setContentsMargins(0, 8, 0, 12)
        
        data_title = QLabel("Detailed Event Data")
        data_title.setFont(QFont("-apple-system", 18, QFont.DemiBold))
        data_title.setStyleSheet("color: #111827; letter-spacing: -0.5px;")
        data_header_layout.addWidget(data_title)
        
        data_subtitle = QLabel("Comprehensive list of all detected events")
        data_subtitle.setStyleSheet("color: #6B7280; font-size: 14px; margin-left: 12px; padding-top: 4px;")
        data_header_layout.addWidget(data_subtitle)
        
        data_header_layout.addStretch()
        data_layout.addWidget(data_header)
        
        # Event table
        self.event_table = QTableWidget()
        self.event_table.setStyleSheet("""
            QTableWidget {
                background-color: #FFFFFF;
                color: #1F2937;
                gridline-color: #E5E7EB;
                border: 1px solid #E4E7EB;
                border-radius: 8px;
                font-size: 14px;
            }
            QHeaderView::section {
                background-color: #F8FAFC;
                color: #4B5563;
                font-weight: 600;
                border: none;
                border-bottom: 1px solid #E5E7EB;
                padding: 12px;
                font-size: 13px;
            }
            QTableWidget::item {
                border: none;
                border-bottom: 1px solid #E5E7EB;
                padding: 8px 4px;
            }
            QTableWidget::item:selected {
                background-color: #EFF6FF;
                color: #1E40AF;
            }
        """)
        self.event_table.setColumnCount(5)
        self.event_table.setHorizontalHeaderLabels(["Type", "Time", "Description", "Frame", "ID"])
        self.event_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.setAlternatingRowColors(True)
        self.event_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        data_layout.addWidget(self.event_table)
        
        # Pagination controls
        pagination = QWidget()
        pagination_layout = QHBoxLayout(pagination)
        pagination_layout.setContentsMargins(0, 5, 0, 0)
        
        self.events_count_label = QLabel("Showing 0 of 0 events")
        self.events_count_label.setStyleSheet("color: #6B7280; font-size: 14px;")
        pagination_layout.addWidget(self.events_count_label)
        
        pagination_layout.addStretch()
        
        prev_btn = QPushButton("Previous")
        prev_btn.setStyleSheet("""
            QPushButton {
                background-color: #F9FAFB;
                color: #4B5563;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover:!disabled {
                background-color: #F3F4F6;
                border-color: #9CA3AF;
            }
            QPushButton:pressed {
                background-color: #E5E7EB;
            }
            QPushButton:disabled {
                color: #9CA3AF;
                border-color: #E5E7EB;
                background-color: #F9FAFB;
            }
        """)
        prev_btn.setEnabled(False)
        pagination_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next")
        next_btn.setStyleSheet("""
            QPushButton {
                background-color: #F9FAFB;
                color: #4B5563;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 14px;
                font-weight: 500;
                margin-left: 8px;
            }
            QPushButton:hover:!disabled {
                background-color: #F3F4F6;
                border-color: #9CA3AF;
            }
            QPushButton:pressed {
                background-color: #E5E7EB;
            }
            QPushButton:disabled {
                color: #9CA3AF;
                border-color: #E5E7EB;
                background-color: #F9FAFB;
            }
        """)
        next_btn.setEnabled(False)
        pagination_layout.addWidget(next_btn)
        
        data_layout.addWidget(pagination)
        self.layout.addWidget(data_section)
    
    def switch_tab(self, index):
        """Switch between chart tabs."""
        self.chart_stack.setCurrentIndex(index)
        for i, btn in enumerate(self.tab_buttons):
            btn.setChecked(i == index)
    
    def set_events(self, events):
        """Set the events data and update the dashboard."""
        self.events = events
        
        # Update event type filter options
        self.event_filter.clear()
        self.event_filter.addItem("All Events")
        
        # Find unique event types
        event_types = set()
        for event in events:
            event_types.add(event['type'])
        
        # Add event types to filter
        for event_type in sorted(event_types):
            self.event_filter.addItem(event_type)
        
        # Update the dashboard with the new data
        self.update_dashboard()
    
    def update_dashboard(self):
        """Update all charts and stats with filtered data."""
        # Apply filters to get filtered events
        filtered_events = self.get_filtered_events()
        
        # Update statistics
        self.update_statistics(filtered_events)
        
        # Update charts
        self.event_type_chart.update_chart(filtered_events)
        self.timeline_chart.update_chart(filtered_events)
        self.roi_chart.update_chart(filtered_events)
        self.event_roi_chart.update_chart(filtered_events)
        
        # Update empty state visibility
        has_events = len(filtered_events) > 0
        self.event_type_empty.setVisible(not has_events)
        self.timeline_empty.setVisible(not has_events)
        self.roi_empty.setVisible(not has_events)
        self.event_roi_empty.setVisible(not has_events)
        
        # Update table
        self.update_event_table(filtered_events)
        
        # Update pagination
        self.events_count_label.setText(f"Showing {len(filtered_events)} of {len(filtered_events)} events")
    
    def get_filtered_events(self):
        """Apply selected filters to the events."""
        if not self.events:
            return []
        
        # Start with all events
        filtered_events = self.events.copy()
        
        # Apply time filter
        time_filter = self.time_filter.currentText()
        if time_filter != "All Time":
            current_time = datetime.now()
            
            if time_filter == "Last Hour":
                cutoff_time = current_time - timedelta(hours=1)
            elif time_filter == "Last 12 Hours":
                cutoff_time = current_time - timedelta(hours=12)
            elif time_filter == "Last 24 Hours":
                cutoff_time = current_time - timedelta(hours=24)
            
            # Filter by timestamp
            # Note: This is a simple filter that assumes timestamps are in HH:MM:SS format
            # In a real application, you would want to store and compare actual datetime objects
            cutoff_str = cutoff_time.strftime('%H:%M:%S')
            filtered_events = [e for e in filtered_events if e.get('timestamp', '') >= cutoff_str]
        
        # Apply event type filter
        event_type = self.event_filter.currentText()
        if event_type != "All Events":
            filtered_events = [e for e in filtered_events if e.get('type', '') == event_type]
        
        return filtered_events
    
    def update_statistics(self, events):
        """Update the statistics cards with the filtered event data."""
        # Total events
        self.stat_cards[0]["value_label"].setText(str(len(events)))
        if len(events) > 0:
            self.stat_cards[0]["subtitle_label"].setText(f"{len(events)} events in the selected time range")
        else:
            self.stat_cards[0]["subtitle_label"].setText("No events detected in the selected time range")
        
        # Unique objects tracked
        track_ids = set()
        for event in events:
            if 'track_id' in event:
                track_ids.add(event['track_id'])
        
        self.stat_cards[1]["value_label"].setText(str(len(track_ids)))
        if len(track_ids) > 0:
            self.stat_cards[1]["subtitle_label"].setText(f"{len(track_ids)} unique objects tracked")
        else:
            self.stat_cards[1]["subtitle_label"].setText("No objects detected in the selected time range")
        
        # Person count (assuming person events have specific identifiers)
        person_events = [e for e in events if 'Person' in e.get('type', '') or 'person' in e.get('description', '').lower()]
        person_ids = set()
        for event in person_events:
            if 'track_id' in event:
                person_ids.add(event['track_id'])
        
        self.stat_cards[2]["value_label"].setText(str(len(person_ids)))
        if len(person_ids) > 0:
            self.stat_cards[2]["subtitle_label"].setText(f"{len(person_ids)} people detected")
        else:
            self.stat_cards[2]["subtitle_label"].setText("No people detected in the selected time range")
        
        # Potential issues (theft, long dwell)
        issue_events = [e for e in events if 
                        'Potential Theft' in e.get('type', '') or 
                        'Long Dwell Time' in e.get('type', '')]
        
        self.stat_cards[3]["value_label"].setText(str(len(issue_events)))
        if len(issue_events) > 0:
            self.stat_cards[3]["subtitle_label"].setText(f"{len(issue_events)} potential issues detected")
        else:
            self.stat_cards[3]["subtitle_label"].setText("No issues detected in the selected time range")
    
    def update_event_table(self, events):
        """Update the table with detailed event data."""
        # Clear existing table data
        self.event_table.setRowCount(0)
        
        if not events:
            # Add a single row showing "No events to display"
            self.event_table.setRowCount(1)
            no_data_item = QTableWidgetItem("No events to display")
            no_data_item.setTextAlignment(Qt.AlignCenter)
            self.event_table.setSpan(0, 0, 1, 5)
            self.event_table.setItem(0, 0, no_data_item)
            return
        
        # Add events to table
        for i, event in enumerate(events):
            self.event_table.insertRow(i)
            
            # Event type
            type_item = QTableWidgetItem(event.get('type', ''))
            type_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.event_table.setItem(i, 0, type_item)
            
            # Event timestamp
            time_item = QTableWidgetItem(event.get('timestamp', ''))
            time_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.event_table.setItem(i, 1, time_item)
            
            # Event description
            desc_item = QTableWidgetItem(event.get('description', ''))
            desc_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.event_table.setItem(i, 2, desc_item)
            
            # Frame index
            frame_item = QTableWidgetItem(str(event.get('frame_idx', '')))
            frame_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.event_table.setItem(i, 3, frame_item)
            
            # Track ID
            id_item = QTableWidgetItem(str(event.get('track_id', '')))
            id_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.event_table.setItem(i, 4, id_item)
            
            # Style rows for critical events
            if 'Potential Theft' in event.get('type', ''):
                for col in range(5):
                    self.event_table.item(i, col).setBackground(QColor('#FEE2E2'))  # Light red background
                    self.event_table.item(i, col).setForeground(QColor('#B91C1C'))  # Dark red text
            elif 'Long Dwell Time' in event.get('type', ''):
                for col in range(5):
                    self.event_table.item(i, col).setBackground(QColor('#FEF3C7'))  # Light yellow background
                    self.event_table.item(i, col).setForeground(QColor('#92400E'))  # Dark yellow/orange text
        
        # Resize columns to content
        self.event_table.resizeColumnsToContents() 