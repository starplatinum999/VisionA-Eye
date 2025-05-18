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
        self.roi_points = {}  # For storing quadrilateral points
        self.predefined_roi_types = [
            "Entry", "Exit", "Cashier", 
            "Shelf1", "Shelf2", "Shelf3", 
            "Electronics", "Jewelry", "Clothing",
            "Restricted"
        ]
        
        # Initialize the roi_points in session state if it doesn't exist
        if 'roi_points' not in st.session_state:
            st.session_state.roi_points = {}
    
    def define_roi(self, frame, auto_detect=False):
        """
        Define Regions of Interest (ROIs) on a frame
        
        Args:
            frame: First frame from video
            auto_detect: Whether to attempt automatic ROI detection
            
        Returns:
            Dictionary of ROI areas {name: [x1, y1, x2, y2]}
        """
        # Load existing ROIs from session state if available
        if 'roi_areas' in st.session_state and st.session_state.roi_areas:
            self.roi_areas = st.session_state.roi_areas.copy()
        
        if 'roi_colors' in st.session_state and st.session_state.roi_colors:
            self.roi_colors = st.session_state.roi_colors.copy()
            
        if 'roi_points' in st.session_state and st.session_state.roi_points:
            self.roi_points = st.session_state.roi_points.copy()
        
        if auto_detect:
            # Auto-detect ROIs if requested
            self.auto_detect_roi(frame)
        
        # Display the frame for manual ROI definition
        st.subheader("Define Regions of Interest (ROIs)")
        st.write("Create a quadrilateral ROI by selecting 4 points on the image")
        
        # Step 1: Ask for ROI name and color
        col1, col2 = st.columns(2)
        
        with col1:
            roi_name = st.text_input("ROI Name", value="QuadROI", key="point_roi_name")
        
        with col2:
            # Color selection with predefined options
            color_options = {
                "Green": (0, 255, 0),
                "Red": (0, 0, 255),
                "Blue": (255, 0, 0),
                "Yellow": (0, 255, 255),
                "Purple": (255, 0, 255),
                "Orange": (0, 165, 255)
            }
            selected_color = st.selectbox("ROI Color", list(color_options.keys()), key="roi_color_select")
            roi_color = color_options[selected_color]
        
        # Convert frame to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Initialize session state for storing points if it doesn't exist
        if "quad_points" not in st.session_state:
            st.session_state.quad_points = []
        
        # Display status and instructions
        st.write(f"Select points: {len(st.session_state.quad_points)}/4 points selected")
        
        # Create a clear button
        if st.button("Clear Points"):
            st.session_state.quad_points = []
            st.experimental_rerun()
        
        # Check if points are already selected
        if len(st.session_state.quad_points) >= 4:
            # Display the frame with the quadrilateral
            preview_frame = rgb_frame.copy()
            
            # Draw the quadrilateral
            points = np.array(st.session_state.quad_points[:4], np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(preview_frame, [points], True, roi_color, 2)
            
            # Fill with semi-transparent color
            overlay = preview_frame.copy()
            cv2.fillPoly(overlay, [points], roi_color)
            alpha = 0.2  # Transparency factor
            cv2.addWeighted(overlay, alpha, preview_frame, 1 - alpha, 0, preview_frame)
            
            # Highlight the vertices
            for i, (x, y) in enumerate(st.session_state.quad_points[:4]):
                cv2.circle(preview_frame, (x, y), 5, (255, 0, 0), -1)
                cv2.putText(preview_frame, f"P{i+1}", (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Display the preview
            st.image(preview_frame, caption=f"ROI Preview: {roi_name}", use_column_width=True)
            
            # Calculate bounding box coordinates
            min_x = min(p[0] for p in st.session_state.quad_points[:4])
            min_y = min(p[1] for p in st.session_state.quad_points[:4])
            max_x = max(p[0] for p in st.session_state.quad_points[:4])
            max_y = max(p[1] for p in st.session_state.quad_points[:4])
            
            # Save button
            if st.button("Save ROI", key="save_quad_roi"):
                # Store rectangular bounding box in roi_areas for compatibility
                self.roi_areas[roi_name] = [min_x, min_y, max_x, max_y]
                
                # Also store the actual points in roi_points
                self.roi_points[roi_name] = st.session_state.quad_points[:4]
                
                # Assign the selected color
                self.roi_colors[roi_name] = roi_color
                
                # Save to session state
                st.session_state.roi_areas = self.roi_areas.copy()
                st.session_state.roi_colors = self.roi_colors.copy()
                
                # Also save roi_points to session state
                if 'roi_points' not in st.session_state:
                    st.session_state.roi_points = {}
                st.session_state.roi_points[roi_name] = self.roi_points[roi_name]
                
                st.session_state.roi_defined = True
                
                # Clear the points for the next ROI
                st.session_state.quad_points = []
                
                st.success(f"Saved quadrilateral ROI: {roi_name}")
                st.experimental_rerun()
        else:
            # Use a canvas to allow the user to select points
            # Convert numpy array to PIL Image for st_canvas
            pil_image = Image.fromarray(rgb_frame)
            
            # Create a drawable canvas for selecting points
            point_canvas = st_canvas(
                fill_color="rgba(255, 0, 0, 0.5)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=pil_image,
                height=height,
                width=width,
                drawing_mode="point",
                point_display_radius=5,
                key="point_roi_canvas"
            )
            
            # Process canvas result to get the points
            if point_canvas.json_data is not None and "objects" in point_canvas.json_data:
                # Get only the newly added point (if any)
                objects = point_canvas.json_data["objects"]
                if objects and len(objects) > len(st.session_state.quad_points):
                    new_obj = objects[-1]
                    
                    if new_obj["type"] == "circle":
                        x = int(new_obj["left"] + new_obj["radius"])
                        y = int(new_obj["top"] + new_obj["radius"])
                        
                        # Check if point is not already in the list (avoid duplicates)
                        point = (x, y)
                        if point not in st.session_state.quad_points and len(st.session_state.quad_points) < 4:
                            st.session_state.quad_points.append(point)
                            st.experimental_rerun()
            
            # If we have some points but not 4 yet, display the current progress
            if 1 <= len(st.session_state.quad_points) < 4:
                preview_frame = rgb_frame.copy()
                
                # Draw lines between the points we have so far
                for i, point in enumerate(st.session_state.quad_points):
                    # Draw the point
                    cv2.circle(preview_frame, point, 5, (255, 0, 0), -1)
                    cv2.putText(preview_frame, f"P{i+1}", (point[0]+5, point[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Draw a line to the previous point
                    if i > 0:
                        cv2.line(preview_frame, st.session_state.quad_points[i-1], point, roi_color, 2)
                
                # Draw a line from the last point to the first if we have at least 3 points
                if len(st.session_state.quad_points) >= 3:
                    cv2.line(preview_frame, st.session_state.quad_points[-1], st.session_state.quad_points[0], roi_color, 2)
                
                # Display the preview
                st.image(preview_frame, caption=f"Progress: {len(st.session_state.quad_points)}/4 points", use_column_width=True)
        
        # Display the defined ROIs on the frame
        if self.roi_areas:
            display_frame = rgb_frame.copy()
            
            for roi_name, roi_coords in self.roi_areas.items():
                x1, y1, x2, y2 = roi_coords
                color = self.roi_colors.get(roi_name, (0, 255, 0))
                
                # Check if this ROI has custom points (quadrilateral)
                if roi_name in self.roi_points:
                    # Draw polygon with the custom points
                    points = np.array(self.roi_points[roi_name], np.int32)
                    points = points.reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [points], True, color, 2)
                    
                    # Fill with semi-transparent color
                    overlay = display_frame.copy()
                    cv2.fillPoly(overlay, [points], color)
                    alpha = 0.2  # Transparency factor
                    cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
                else:
                    # Draw rectangle and label for standard rectangular ROIs
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add ROI name label
                if roi_name in self.roi_points:
                    points = self.roi_points[roi_name]
                    min_y_point = min(points, key=lambda p: p[1])
                    label_x, label_y = min_y_point
                else:
                    label_x, label_y = x1, y1
                    
                cv2.putText(display_frame, roi_name, (label_x, label_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
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
        
        # Include quadrilateral points if they exist
        if self.roi_points:
            # Convert tuple points to list for JSON serialization
            serializable_points = {}
            for roi_name, points in self.roi_points.items():
                serializable_points[roi_name] = [list(point) for point in points]
            
            roi_config["roi_points"] = serializable_points
        
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
            
            # Load quadrilateral points if they exist
            if "roi_points" in roi_config:
                self.roi_points = {}
                for roi_name, points in roi_config["roi_points"].items():
                    # Convert lists back to tuples
                    self.roi_points[roi_name] = [tuple(point) for point in points]
                
                # Also update session state
                if 'roi_points' not in st.session_state:
                    st.session_state.roi_points = {}
                
                st.session_state.roi_points = self.roi_points.copy()
            
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
            
            # Check if this ROI has custom points (quadrilateral)
            if roi_name in self.roi_points:
                # Draw polygon with the custom points
                points = np.array(self.roi_points[roi_name], np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(output_frame, [points], True, color, 2)
                
                # Fill with semi-transparent color
                overlay = output_frame.copy()
                cv2.fillPoly(overlay, [points], color)
                alpha = 0.2  # Transparency factor
                cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
            else:
                # Draw rectangle and label for standard rectangular ROIs
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add ROI name label - for quadrilateral, use the minimum y-coordinate point
            if roi_name in self.roi_points:
                points = self.roi_points[roi_name]
                min_y_point = min(points, key=lambda p: p[1])
                label_x, label_y = min_y_point
            else:
                label_x, label_y = x1, y1
                
            cv2.putText(output_frame, roi_name, (label_x, label_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_frame 