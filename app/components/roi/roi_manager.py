import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import random
from PIL import Image

class ROIManager:
    """
    Region of Interest (ROI) manager for defining and tracking zones in a video
    """
    def __init__(self):
        """
        Initialize ROI manager
        """
        self.roi_areas = {}
        self.roi_colors = {}
        self.predefined_roi_types = [
            "Entry", "Exit", "Cashier", 
            "Shelf1", "Shelf2", "Shelf3", 
            "Electronics", "Jewelry", "Clothing",
            "Restricted"
        ]
    
    def define_roi(self, frame, auto_detect=False):
        """
        Define Regions of Interest (ROIs) on a frame
        
        Args:
            frame: First frame from video
            auto_detect: Whether to attempt automatic ROI detection
            
        Returns:
            Dictionary of ROI areas {name: [x1, y1, x2, y2]}
        """
        if auto_detect:
            # Auto-detect ROIs if requested
            self.auto_detect_roi(frame)
        
        # Display the frame for manual ROI definition
        st.subheader("Define Regions of Interest (ROIs)")
        st.write("Draw rectangles on the image to define important areas for monitoring.")
        
        # Create tabs for different ROI methods
        tab1, tab2 = st.tabs(["Manual Drawing", "Quick Presets"])
        
        with tab1:
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Convert frame to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert numpy array to PIL Image for st_canvas
            pil_image = Image.fromarray(rgb_frame)
            
            # Create a drawable canvas for defining ROIs
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # orange with some transparency
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=pil_image,
                height=height,
                width=width,
                drawing_mode="rect",
                key="roi_canvas"
            )
            
            # Process drawn rectangles
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                # Create columns for ROI naming
                roi_names = []
                
                # Form for ROI naming
                with st.form("roi_naming_form"):
                    st.write("Name your ROIs:")
                    
                    # Create input fields for each drawn rectangle
                    for i, obj in enumerate(canvas_result.json_data["objects"]):
                        # Default name based on count
                        default_name = f"ROI{i+1}"
                        
                        # Create text input for naming
                        roi_name = st.text_input(f"ROI {i+1}", value=default_name, key=f"roi_name_{i}")
                        roi_names.append(roi_name)
                    
                    # Submit button
                    submit_button = st.form_submit_button("Save ROIs")
                    
                    if submit_button:
                        # Process ROIs
                        self.roi_areas = {}
                        
                        for i, obj in enumerate(canvas_result.json_data["objects"]):
                            # Extract rectangle coordinates
                            left = obj["left"]
                            top = obj["top"]
                            right = left + obj["width"]
                            bottom = top + obj["height"]
                            
                            # Store ROI with name
                            roi_name = roi_names[i]
                            self.roi_areas[roi_name] = [int(left), int(top), int(right), int(bottom)]
                            
                            # Assign a random color to this ROI
                            self.roi_colors[roi_name] = (
                                random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255)
                            )
                        
                        # Save to session state
                        st.session_state.roi_areas = self.roi_areas.copy() 
                        st.session_state.roi_colors = self.roi_colors.copy()
                        st.session_state.roi_defined = True
                        
                        st.success(f"Saved {len(self.roi_areas)} Regions of Interest")
        
        with tab2:
            st.write("Select predefined ROI types:")
            
            # Create a form for predefined ROIs
            with st.form("predefined_roi_form"):
                # Create multiselect for predefined ROI types
                selected_rois = st.multiselect(
                    "Select ROI types to define:",
                    self.predefined_roi_types,
                    default=["Entry", "Exit", "Cashier", "Shelf1"]
                )
                
                # Show preview of selected ROIs
                if selected_rois:
                    preview_frame = rgb_frame.copy()
                    height, width = preview_frame.shape[:2]
                    
                    # Simple automatic placement based on selection
                    if "Entry" in selected_rois:
                        entry_roi = [0, height//2, width//5, height]
                        cv2.rectangle(preview_frame, (entry_roi[0], entry_roi[1]), (entry_roi[2], entry_roi[3]), (0, 255, 0), 2)
                        cv2.putText(preview_frame, "Entry", (entry_roi[0], entry_roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if "Exit" in selected_rois:
                        exit_roi = [width*4//5, height//2, width, height]
                        cv2.rectangle(preview_frame, (exit_roi[0], exit_roi[1]), (exit_roi[2], exit_roi[3]), (0, 0, 255), 2)
                        cv2.putText(preview_frame, "Exit", (exit_roi[0], exit_roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    if "Cashier" in selected_rois:
                        cashier_roi = [width*2//3, 0, width, height//3]
                        cv2.rectangle(preview_frame, (cashier_roi[0], cashier_roi[1]), (cashier_roi[2], cashier_roi[3]), (255, 0, 0), 2)
                        cv2.putText(preview_frame, "Cashier", (cashier_roi[0], cashier_roi[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Add shelves if selected
                    shelf_count = sum(1 for roi in selected_rois if "Shelf" in roi)
                    if shelf_count > 0:
                        shelf_width = width // (shelf_count + 1)
                        shelf_height = height // 3
                        
                        shelf_idx = 0
                        for roi in selected_rois:
                            if "Shelf" in roi:
                                shelf_x = shelf_width * (shelf_idx + 1)
                                shelf_roi = [shelf_x - shelf_width//2, height//3, shelf_x + shelf_width//2, height*2//3]
                                cv2.rectangle(preview_frame, (shelf_roi[0], shelf_roi[1]), (shelf_roi[2], shelf_roi[3]), (255, 255, 0), 2)
                                cv2.putText(preview_frame, roi, (shelf_roi[0], shelf_roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                                shelf_idx += 1
                    
                    # Add special areas
                    special_areas = [roi for roi in selected_rois if roi in ["Electronics", "Jewelry", "Clothing", "Restricted"]]
                    for i, area in enumerate(special_areas):
                        area_x = width // 4
                        area_y = height // 4 + (i * height // 8)
                        area_roi = [area_x, area_y, area_x + width//4, area_y + height//8]
                        cv2.rectangle(preview_frame, (area_roi[0], area_roi[1]), (area_roi[2], area_roi[3]), (0, 165, 255), 2)
                        cv2.putText(preview_frame, area, (area_roi[0], area_roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    
                    # Display preview
                    st.image(preview_frame, caption="ROI Preview", use_column_width=True)
                
                # Submit button
                preset_submit = st.form_submit_button("Use Presets")
                
                if preset_submit and selected_rois:
                    # Create preset ROIs
                    self.roi_areas = {}
                    height, width = frame.shape[:2]
                    
                    if "Entry" in selected_rois:
                        self.roi_areas["Entry"] = [0, height//2, width//5, height]
                        self.roi_colors["Entry"] = (0, 255, 0)
                    
                    if "Exit" in selected_rois:
                        self.roi_areas["Exit"] = [width*4//5, height//2, width, height]
                        self.roi_colors["Exit"] = (0, 0, 255)
                    
                    if "Cashier" in selected_rois:
                        self.roi_areas["Cashier"] = [width*2//3, 0, width, height//3]
                        self.roi_colors["Cashier"] = (255, 0, 0)
                    
                    # Add shelves if selected
                    shelf_count = sum(1 for roi in selected_rois if "Shelf" in roi)
                    if shelf_count > 0:
                        shelf_width = width // (shelf_count + 1)
                        shelf_height = height // 3
                        
                        shelf_idx = 0
                        for roi in selected_rois:
                            if "Shelf" in roi:
                                shelf_x = shelf_width * (shelf_idx + 1)
                                self.roi_areas[roi] = [
                                    shelf_x - shelf_width//2, 
                                    height//3, 
                                    shelf_x + shelf_width//2, 
                                    height*2//3
                                ]
                                self.roi_colors[roi] = (255, 255, 0)
                                shelf_idx += 1
                    
                    # Add special areas
                    special_areas = [roi for roi in selected_rois if roi in ["Electronics", "Jewelry", "Clothing", "Restricted"]]
                    for i, area in enumerate(special_areas):
                        area_x = width // 4
                        area_y = height // 4 + (i * height // 8)
                        self.roi_areas[area] = [
                            area_x, 
                            area_y, 
                            area_x + width//4, 
                            area_y + height//8
                        ]
                        self.roi_colors[area] = (0, 165, 255)
                    
                    # Save to session state
                    st.session_state.roi_areas = self.roi_areas.copy()
                    st.session_state.roi_colors = self.roi_colors.copy()
                    st.session_state.roi_defined = True
                    
                    st.success(f"Created {len(self.roi_areas)} predefined Regions of Interest")
                    st.session_state.roi_defined = True
                    st.session_state.roi_areas = self.roi_areas
        
        # Display the defined ROIs on the frame
        if self.roi_areas:
            display_frame = rgb_frame.copy()
            
            for roi_name, roi_coords in self.roi_areas.items():
                x1, y1, x2, y2 = roi_coords
                color = self.roi_colors.get(roi_name, (0, 255, 0))
                
                # Draw rectangle and label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, roi_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            st.image(display_frame, caption="Defined ROIs", use_column_width=True)
            
            # Show ROI information
            with st.expander("ROI Details"):
                for roi_name, roi_coords in self.roi_areas.items():
                    st.write(f"**{roi_name}**: {roi_coords}")
        
        return self.roi_areas
    
    def auto_detect_roi(self, frame):
        """
        Automatically detect potential ROIs in a frame
        
        Args:
            frame: Frame to analyze
        """
        # This is a placeholder for auto-detection
        # In a real implementation, this would use computer vision techniques to identify
        # potential ROIs like entrances, exits, shelves, etc.
        
        # For now, create some simple default ROIs based on frame dimensions
        height, width = frame.shape[:2]
        
        # Simple ROI definitions based on frame dimensions
        self.roi_areas = {
            "Entry": [0, height//2, width//5, height],
            "Exit": [width*4//5, height//2, width, height],
            "Cashier": [width*2//3, 0, width, height//3],
            "Shelf1": [width//4, height//3, width//2, height*2//3],
            "Shelf2": [width//2, height//3, width*3//4, height*2//3]
        }
        
        # Assign colors
        self.roi_colors = {
            "Entry": (0, 255, 0),    # Green
            "Exit": (0, 0, 255),     # Red
            "Cashier": (255, 0, 0),  # Blue
            "Shelf1": (255, 255, 0), # Yellow
            "Shelf2": (255, 165, 0)  # Orange
        }
    
    def save_roi_config(self, file_path):
        """
        Save ROI configuration to a file
        
        Args:
            file_path: Path to save configuration
        """
        # Create ROI configuration dictionary
        roi_config = {
            "roi_areas": self.roi_areas,
            "roi_colors": self.roi_colors
        }
        
        # Save to JSON file
        import json
        with open(file_path, 'w') as f:
            json.dump(roi_config, f)
    
    def load_roi_config(self, file_path):
        """
        Load ROI configuration from a file
        
        Args:
            file_path: Path to load configuration from
            
        Returns:
            bool: Success status
        """
        try:
            import json
            with open(file_path, 'r') as f:
                roi_config = json.load(f)
            
            self.roi_areas = roi_config.get("roi_areas", {})
            self.roi_colors = roi_config.get("roi_colors", {})
            
            return True
        except Exception as e:
            print(f"Error loading ROI configuration: {e}")
            return False
    
    def draw_roi(self, frame):
        """
        Draw defined ROIs on a frame
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            Frame with ROIs drawn
        """
        # Make a copy of the frame
        output_frame = frame.copy()
        
        # Draw each ROI
        for roi_name, roi_coords in self.roi_areas.items():
            x1, y1, x2, y2 = roi_coords
            color = self.roi_colors.get(roi_name, (0, 255, 0))
            
            # Draw rectangle and label
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_frame, roi_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_frame 