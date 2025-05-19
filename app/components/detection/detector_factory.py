from app.components.detection.detectors.theft_detector import TheftDetector
from app.components.detection.detectors.footfall_detector import FootfallDetector
from app.components.detection.detectors.shelf_monitor_detector import ShelfMonitorDetector
from app.components.detection.detectors.unauthorized_area_detector import UnauthorizedAreaDetector

class DetectorFactory:
    """Factory for creating detector instances of various types."""
    
    @staticmethod
    def create_detector(detector_type):
        """
        Create a detector instance based on detector type.
        
        Args:
            detector_type (str): Type of detector to create
            
        Returns:
            Detector instance or None if type is not recognized
        """
        if detector_type == "theft_detector":
            return TheftDetector()
        elif detector_type == "footfall_detector":
            return FootfallDetector()
        elif detector_type == "shelf_monitor_detector":
            return ShelfMonitorDetector()
        elif detector_type == "unauthorized_area_detector":
            return UnauthorizedAreaDetector()
        else:
            return None
    
    @staticmethod
    def get_detector_info(detector_type):
        """
        Get information about a detector type.
        
        Args:
            detector_type (str): Type of detector
            
        Returns:
            dict: Information about the detector type
        """
        detector_info = {
            "theft_detector": {
                "name": "Theft Detector",
                "description": "Detect potential theft and suspicious activities",
                "events": ["Potential Theft", "Suspicious Item Grab", "Unauthorized Access", "Item Taken", "Suspicious Movement"]
            },
            "footfall_detector": {
                "name": "Footfall Detector",
                "description": "Track people movement, entries, and exits",
                "events": ["Area Entry", "Area Exit", "Long Dwell Time", "Boundary Crossing"]
            },
            "shelf_monitor_detector": {
                "name": "Shelf Monitor",
                "description": "Monitor product placement and stock levels",
                "events": ["Low Stock", "Empty Shelf", "Product Placement", "Product Removal", "Shelf Stocked", "Person at Shelf"]
            },
            "unauthorized_area_detector": {
                "name": "Unauthorized Area Detector",
                "description": "Detect people in restricted areas",
                "events": ["Unauthorized Access", "Restricted Area Access", "Staff-Only Area Access", "Loitering in Restricted Area"]
            }
        }
        
        return detector_info.get(detector_type, {
            "name": "Unknown Detector",
            "description": "No description available",
            "events": []
        }) 