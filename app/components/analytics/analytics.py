import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
import os
import json
import tempfile
from collections import Counter

class AnalyticsManager:
    """
    Analytics manager for generating visualizations and reports
    """
    def __init__(self, events, roi_areas, reasoner=None):
        """
        Initialize analytics manager
        
        Args:
            events: List of events from video processing
            roi_areas: Dictionary of ROI areas {name: [x1, y1, x2, y2]}
            reasoner: SymbolicReasoner instance (optional, for generating insights)
        """
        self.events = events
        self.roi_areas = roi_areas
        self.reasoner = reasoner
        
        # Process events into DataFrame for easier analysis
        self.events_df = self._process_events()
        
        # Apply reasoning to events that need it (if reasoner is available)
        if reasoner:
            self._apply_delayed_reasoning()
    
    def _apply_delayed_reasoning(self):
        """
        Apply LLM reasoning to events that were marked for delayed reasoning
        """
        # Check if reasoner is available
        if not self.reasoner:
            return
            
        # Look for events that need reasoning
        reasoning_data_path = os.path.join(tempfile.gettempdir(), "events_needing_reasoning.json")
        
        try:
            # Check if we have stored indices of events needing reasoning
            if os.path.exists(reasoning_data_path):
                with open(reasoning_data_path, 'r') as f:
                    indices = json.load(f)
                
                # Display progress
                st.info("Generating AI insights for events...")
                progress = st.progress(0)
                
                # Process each event
                for i, idx in enumerate(indices):
                    if idx < len(self.events):
                        event = self.events[idx]
                        
                        # Make sure event needs reasoning and doesn't already have it
                        if event.get('needs_reasoning', False) and 'reasoning' not in event:
                            # Find track data
                            track_id = event.get('track_id')
                            track_data = None
                            
                            # Try to reconstruct track data
                            if track_id is not None:
                                # Find all events for this track
                                track_events = [e for e in self.events if e.get('track_id') == track_id]
                                
                                # Construct minimal track data
                                track_data = {
                                    'id': track_id,
                                    'events': track_events,
                                    'class_id': 0  # Assume person
                                }
                            
                            # Apply reasoning
                            try:
                                if event['type'] == 'Potential Theft':
                                    event['reasoning'] = self.reasoner.analyze_event(event, track_data or {})
                                elif event['type'] == 'Long Dwell Time':
                                    event['reasoning'] = self.reasoner.analyze_dwell_time(event, track_data or {})
                                else:
                                    event['reasoning'] = self.reasoner.analyze_event(event, track_data or {})
                            except Exception as e:
                                event['reasoning'] = f"Error in reasoning: {e}"
                    
                    # Update progress
                    progress.progress((i + 1) / len(indices))
                
                # Clean up the file after processing
                try:
                    os.remove(reasoning_data_path)
                except:
                    pass
                    
                st.success("AI insights generation complete!")
            
            # Alternatively, just check all events with needs_reasoning flag
            else:
                reasoning_needed = [i for i, event in enumerate(self.events) 
                                     if event.get('needs_reasoning', False) and 'reasoning' not in event]
                
                if reasoning_needed:
                    # Display progress
                    st.info("Generating AI insights for events...")
                    progress = st.progress(0)
                    
                    for i, idx in enumerate(reasoning_needed):
                        event = self.events[idx]
                        
                        # Find track data (minimal reconstruction)
                        track_id = event.get('track_id')
                        track_data = {'id': track_id, 'class_id': 0}  # Assume person
                        
                        # Apply reasoning
                        try:
                            if event['type'] == 'Potential Theft':
                                event['reasoning'] = self.reasoner.analyze_event(event, track_data)
                            elif event['type'] == 'Long Dwell Time':
                                event['reasoning'] = self.reasoner.analyze_dwell_time(event, track_data)
                            else:
                                event['reasoning'] = self.reasoner.analyze_event(event, track_data)
                        except Exception as e:
                            event['reasoning'] = f"Error in reasoning: {e}"
                        
                        # Update progress
                        progress.progress((i + 1) / len(reasoning_needed))
                    
                    st.success("AI insights generation complete!")
        except Exception as e:
            st.warning(f"Error applying reasoning to events: {e}")
            
    def _process_events(self):
        """
        Process events into a DataFrame
        
        Returns:
            pandas.DataFrame: Processed events
        """
        if not self.events:
            return pd.DataFrame()
        
        # Extract relevant data from events
        processed_data = []
        
        for event in self.events:
            # Convert timestamp to datetime
            timestamp = event.get('timestamp', '00:00:00')
            
            # Create a base datetime for today and add the time
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            event_time = datetime.strptime(timestamp, '%H:%M:%S').time()
            event_datetime = datetime.combine(base_date.date(), event_time)
            
            # Extract event data
            event_data = {
                'timestamp': event_datetime,
                'type': event.get('type', 'Unknown'),
                'description': event.get('description', ''),
                'from_roi': event.get('from_roi', None),
                'to_roi': event.get('to_roi', None),
                'track_id': event.get('track_id', None),
                'person_id': event.get('person_id', None),
                'item_id': event.get('item_id', None),
                'roi_name': event.get('roi_name', None),
                'dwell_time': event.get('dwell_time', None),
                'frame_idx': event.get('frame_idx', 0),
                'has_reasoning': 'reasoning' in event
            }
            
            processed_data.append(event_data)
        
        # Create DataFrame
        return pd.DataFrame(processed_data)
    
    def display_dashboard(self):
        """
        Display analytics dashboard
        """
        if self.events_df.empty:
            st.warning("No events data available for analysis.")
            return
        
        # Create dashboard layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Event Timeline
            st.subheader("Event Timeline")
            self._display_event_timeline()
            
            # ROI Analysis
            st.subheader("ROI Analysis")
            self._display_roi_analysis()
        
        with col2:
            # Event Type Distribution
            st.subheader("Event Distribution")
            self._display_event_distribution()
            
            # Person Tracking Analysis
            st.subheader("Person Tracking")
            self._display_person_tracking()
        
        # Footfall Heatmap
        st.subheader("Footfall Heatmap")
        self._display_footfall_heatmap()
        
        # Potential Threats Section
        st.subheader("Potential Security Threats")
        self._display_potential_threats()
    
    def _display_event_timeline(self):
        """
        Display event timeline visualization
        """
        # Create a timeline of events
        fig = px.scatter(
            self.events_df,
            x='timestamp',
            y='type',
            color='type',
            hover_name='description',
            size_max=10,
            height=300
        )
        
        # Add connecting lines for same track_id
        if 'track_id' in self.events_df.columns and not self.events_df['track_id'].isna().all():
            # Get unique track IDs
            track_ids = self.events_df['track_id'].dropna().unique()
            
            # Add connecting lines for each track
            for track_id in track_ids:
                track_events = self.events_df[self.events_df['track_id'] == track_id].sort_values('timestamp')
                
                if len(track_events) > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=track_events['timestamp'],
                            y=track_events['type'],
                            mode='lines',
                            line=dict(width=1, dash='dot'),
                            showlegend=False,
                            hoverinfo='none'
                        )
                    )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Event Type",
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_event_distribution(self):
        """
        Display event type distribution
        """
        # Count events by type
        event_counts = self.events_df['type'].value_counts().reset_index()
        event_counts.columns = ['Event Type', 'Count']
        
        # Create bar chart
        fig = px.bar(
            event_counts,
            x='Count',
            y='Event Type',
            orientation='h',
            color='Event Type',
            height=300
        )
        
        # Update layout
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_roi_analysis(self):
        """
        Display ROI analysis
        """
        # Check if we have ROI-related events
        if 'to_roi' not in self.events_df.columns or self.events_df['to_roi'].isna().all():
            st.write("No ROI transition data available.")
            return
        
        # Get ROI transitions
        roi_transitions = self.events_df[~self.events_df['to_roi'].isna()][['from_roi', 'to_roi']]
        
        # Count transitions between ROIs
        roi_counts = {}
        for _, row in roi_transitions.iterrows():
            from_roi = row['from_roi'] if not pd.isna(row['from_roi']) else 'Entry'
            to_roi = row['to_roi']
            
            if (from_roi, to_roi) in roi_counts:
                roi_counts[(from_roi, to_roi)] += 1
            else:
                roi_counts[(from_roi, to_roi)] = 1
        
        if not roi_counts:
            st.write("No ROI transition data available.")
            return
        
        # Create Sankey diagram data
        source = []
        target = []
        value = []
        
        # Get unique ROIs
        unique_rois = set()
        for from_roi, to_roi in roi_counts.keys():
            if from_roi:
                unique_rois.add(from_roi)
            unique_rois.add(to_roi)
        
        # Map ROIs to indices
        roi_to_idx = {roi: i for i, roi in enumerate(unique_rois)}
        
        # Create Sankey data
        for (from_roi, to_roi), count in roi_counts.items():
            if from_roi:
                source.append(roi_to_idx[from_roi])
                target.append(roi_to_idx[to_roi])
                value.append(count)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(unique_rois)
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        # Update layout
        fig.update_layout(
            title="ROI Transitions",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display dwell time analysis if available
        dwell_events = self.events_df[self.events_df['type'] == 'Long Dwell Time']
        
        if not dwell_events.empty:
            # Create dwell time chart
            with st.expander("Dwell Time Analysis"):
                dwell_data = dwell_events[['roi_name', 'dwell_time']].dropna()
                
                if not dwell_data.empty:
                    # Group by ROI and calculate average dwell time
                    avg_dwell = dwell_data.groupby('roi_name')['dwell_time'].mean().reset_index()
                    avg_dwell.columns = ['ROI', 'Average Dwell Time (s)']
                    
                    # Create bar chart
                    fig = px.bar(
                        avg_dwell,
                        x='ROI',
                        y='Average Dwell Time (s)',
                        color='ROI',
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No dwell time data available.")
    
    def _display_person_tracking(self):
        """
        Display person tracking analysis
        """
        # Filter events related to persons
        person_events = self.events_df[
            (self.events_df['type'] == 'Person at Shelf') | 
            (self.events_df['type'] == 'Potential Theft') |
            (~self.events_df['person_id'].isna())
        ]
        
        if person_events.empty:
            st.write("No person tracking data available.")
            return
        
        # Get unique person IDs
        person_ids = set()
        
        if 'track_id' in person_events.columns:
            person_ids.update(person_events[person_events['type'].isin(['Person at Shelf', 'Potential Theft'])]['track_id'].dropna())
        
        if 'person_id' in person_events.columns:
            person_ids.update(person_events['person_id'].dropna())
        
        person_ids = list(person_ids)
        
        # Create metrics
        num_persons = len(person_ids)
        num_potential_thefts = len(person_events[person_events['type'] == 'Potential Theft'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("People Tracked", num_persons)
        
        with col2:
            st.metric("Potential Theft Events", num_potential_thefts)
        
        # Person activity table
        with st.expander("Person Activity Details"):
            if person_ids:
                person_data = []
                
                for person_id in person_ids:
                    # Find events for this person
                    person_track_events = self.events_df[
                        (self.events_df['track_id'] == person_id) | 
                        (self.events_df['person_id'] == person_id)
                    ]
                    
                    # Calculate activity metrics
                    num_events = len(person_track_events)
                    visited_rois = set()
                    
                    if 'to_roi' in person_track_events.columns:
                        visited_rois.update(person_track_events['to_roi'].dropna())
                    
                    if 'from_roi' in person_track_events.columns:
                        visited_rois.update(person_track_events['from_roi'].dropna())
                    
                    if 'roi_name' in person_track_events.columns:
                        visited_rois.update(person_track_events['roi_name'].dropna())
                    
                    suspicious_events = len(person_track_events[
                        (person_track_events['type'] == 'Potential Theft') | 
                        (person_track_events['type'] == 'Long Dwell Time')
                    ])
                    
                    # Add to person data
                    person_data.append({
                        'Person ID': person_id,
                        'Events': num_events,
                        'ROIs Visited': len(visited_rois),
                        'Suspicious Events': suspicious_events
                    })
                
                # Create DataFrame and display
                person_df = pd.DataFrame(person_data)
                st.dataframe(person_df)
            else:
                st.write("No person activity data available.")
    
    def _display_footfall_heatmap(self):
        """
        Display footfall heatmap visualization
        """
        # Check if we have ROI data and ROI transitions
        if not self.roi_areas or 'to_roi' not in self.events_df.columns or self.events_df['to_roi'].isna().all():
            st.write("No footfall data available.")
            return
        
        # Count events by ROI
        roi_events = []
        
        # Include 'to_roi' events
        if 'to_roi' in self.events_df.columns:
            roi_to_events = self.events_df[~self.events_df['to_roi'].isna()]['to_roi']
            roi_events.extend(roi_to_events.tolist())
        
        # Include 'from_roi' events
        if 'from_roi' in self.events_df.columns:
            roi_from_events = self.events_df[~self.events_df['from_roi'].isna()]['from_roi']
            roi_events.extend(roi_from_events.tolist())
        
        # Include 'roi_name' events
        if 'roi_name' in self.events_df.columns:
            roi_name_events = self.events_df[~self.events_df['roi_name'].isna()]['roi_name']
            roi_events.extend(roi_name_events.tolist())
        
        if not roi_events:
            st.write("No footfall data available.")
            return
        
        # Count ROI visits
        roi_counts = Counter(roi_events)
        
        # Create blank heatmap image
        # Use the first frame size as reference (we'll make it black)
        height, width = 500, 800  # Default size if ROI coordinates don't give clear dimensions
        
        # Try to determine image size from ROI coordinates
        if self.roi_areas:
            max_x = max(coord[2] for coord in self.roi_areas.values())
            max_y = max(coord[3] for coord in self.roi_areas.values())
            
            if max_x > 0 and max_y > 0:
                width, height = max_x, max_y
        
        # Create blank image
        heatmap_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw ROIs with color intensity based on visit count
        max_count = max(roi_counts.values()) if roi_counts else 1
        
        for roi_name, roi_coords in self.roi_areas.items():
            # Skip if no count data for this ROI
            if roi_name not in roi_counts:
                continue
            
            # Calculate color intensity based on count
            intensity = int(255 * roi_counts[roi_name] / max_count)
            
            # Create heatmap color (hot colormap: black -> red -> yellow -> white)
            if intensity < 85:
                color = (0, 0, intensity * 3)  # Blue to purple
            elif intensity < 170:
                color = (intensity - 85, 0, 255)  # Purple to magenta
            else:
                color = (255, intensity - 170, 255)  # Magenta to white
            
            # Draw filled rectangle for ROI
            x1, y1, x2, y2 = roi_coords
            cv2.rectangle(heatmap_img, (x1, y1), (x2, y2), color, -1)
            
            # Draw ROI name
            cv2.putText(heatmap_img, f"{roi_name}: {roi_counts[roi_name]}", 
                       (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display heatmap image
        st.image(heatmap_img, caption="Footfall Heatmap (brighter = more visits)", use_column_width=True)
        
        # Show legend
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Low Traffic")
        with col2:
            st.write("Medium Traffic")
        with col3:
            st.write("High Traffic")
        
        # Create legend colors
        legend_img = np.zeros((50, width, 3), dtype=np.uint8)
        for i in range(width):
            intensity = int(255 * i / width)
            if intensity < 85:
                color = (0, 0, intensity * 3)  # Blue to purple
            elif intensity < 170:
                color = (intensity - 85, 0, 255)  # Purple to magenta
            else:
                color = (255, intensity - 170, 255)  # Magenta to white
            
            cv2.line(legend_img, (i, 0), (i, 50), color, 1)
        
        st.image(legend_img, use_column_width=True)
    
    def _display_potential_threats(self):
        """
        Display potential security threats
        """
        # Filter suspicious events
        suspicious_events = self.events_df[
            (self.events_df['type'] == 'Potential Theft') | 
            (self.events_df['type'] == 'Long Dwell Time')
        ]
        
        if suspicious_events.empty:
            st.info("No potential security threats detected.")
            return
        
        # Display suspicious events
        for _, event in suspicious_events.iterrows():
            with st.expander(f"{event['type']} at {event['timestamp'].strftime('%H:%M:%S')}"):
                st.write(f"**Description:** {event['description']}")
                
                if 'reasoning' in self.events[event.name]:
                    st.write(f"**AI Analysis:** {self.events[event.name]['reasoning']}")
                
                if 'thumbnail' in self.events[event.name] and self.events[event.name]['thumbnail'] is not None:
                    st.image(self.events[event.name]['thumbnail'], caption="Event Thumbnail")
                
                # Show additional event details as key-value pairs
                details = {}
                
                if not pd.isna(event['track_id']):
                    details['Track ID'] = event['track_id']
                
                if not pd.isna(event['from_roi']):
                    details['From ROI'] = event['from_roi']
                
                if not pd.isna(event['to_roi']):
                    details['To ROI'] = event['to_roi']
                
                if not pd.isna(event['roi_name']):
                    details['ROI'] = event['roi_name']
                
                if not pd.isna(event['dwell_time']):
                    details['Dwell Time'] = f"{event['dwell_time']:.1f} seconds"
                
                if details:
                    st.json(details)
    
    def generate_report(self):
        """
        Generate a summary report
        
        Returns:
            str: Report text
        """
        # This would generate a detailed report of the analysis
        # For now, we'll just return a simple summary
        
        if self.events_df.empty:
            return "No events data available for report generation."
        
        # Count events by type
        event_counts = self.events_df['type'].value_counts()
        
        # Count suspicious events
        suspicious_count = len(self.events_df[
            (self.events_df['type'] == 'Potential Theft') | 
            (self.events_df['type'] == 'Long Dwell Time')
        ])
        
        # Generate report
        report = "# Vision AI Surveillance Report\n\n"
        report += f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Event Summary\n\n"
        report += f"* Total Events: {len(self.events_df)}\n"
        report += f"* Suspicious Events: {suspicious_count}\n\n"
        
        report += "## Event Breakdown\n\n"
        for event_type, count in event_counts.items():
            report += f"* {event_type}: {count}\n"
        
        return report 