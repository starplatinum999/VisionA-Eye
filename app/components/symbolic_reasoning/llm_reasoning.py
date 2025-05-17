import os
import json
import time
from llama_cpp import Llama

class SymbolicReasoner:
    """
    Symbolic reasoning module that uses a local LLM to analyze events
    and provide high-level reasoning
    """
    def __init__(self, model_path=None):
        """
        Initialize the symbolic reasoner with LLM
        
        Args:
            model_path: Path to LLM model
        """
        if model_path is None:
            model_path = os.path.join("app", "models", "llm", "deepseek-coder-1.3b-instruct.Q4_K_M.gguf")
        
        self.model_path = model_path
        
        # Initialize LLM
        try:
            # Add disable_signal_handlers=True to fix "signal only works in main thread" error in Streamlit
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,                # Context window size
                n_threads=4,               # CPU threads to use
                n_gpu_layers=0,            # GPU layers (0 for CPU only)
                n_batch=512,               # Batch size for prompt processing
                disable_signal_handlers=True  # Disable signal handlers to work in Streamlit
            )
            print(f"LLM loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            self.llm = None
            
        # Create a fallback response for when LLM is not available
        self.fallback_responses = {
            "theft": "The person appears to be exiting directly after picking up an item without paying.",
            "lingering": "The person is spending an unusually long time in this area.",
            "unusual_movement": "The person's movement pattern is unusual and may indicate suspicious behavior.",
            "normal": "This appears to be normal customer behavior."
        }
    
    def analyze_event(self, event_data, track_data):
        """
        Analyze an event using the LLM
        
        Args:
            event_data: Dictionary containing event information
            track_data: Dictionary containing tracking information
            
        Returns:
            str: Reasoning about the event
        """
        if self.llm is None:
            # Fallback if LLM is not available
            if "Potential Theft" in event_data.get('type', ''):
                return self.fallback_responses["theft"]
            return self.fallback_responses["normal"]
        
        # Create prompt for the LLM
        prompt = self._create_event_prompt(event_data, track_data)
        
        # Query the LLM
        try:
            response = self.llm(
                prompt,
                max_tokens=256,
                stop=["Human:", "AI:"],
                echo=False
            )
            
            reasoning = response['choices'][0]['text'].strip()
            return reasoning
        except Exception as e:
            print(f"Error querying LLM: {e}")
            
            # Fallback reasoning based on event type
            if "Potential Theft" in event_data.get('type', ''):
                return self.fallback_responses["theft"]
            elif "ROI Transition" in event_data.get('type', ''):
                return self.fallback_responses["normal"]
            return "Unable to analyze event."
    
    def analyze_dwell_time(self, event_data, track_data):
        """
        Analyze dwell time events
        
        Args:
            event_data: Dictionary containing event information
            track_data: Dictionary containing tracking information
            
        Returns:
            str: Reasoning about the dwell time
        """
        if self.llm is None:
            # Fallback if LLM is not available
            return self.fallback_responses["lingering"]
        
        # Create prompt for the LLM
        prompt = self._create_dwell_prompt(event_data, track_data)
        
        # Query the LLM
        try:
            response = self.llm(
                prompt,
                max_tokens=256,
                stop=["Human:", "AI:"],
                echo=False
            )
            
            reasoning = response['choices'][0]['text'].strip()
            return reasoning
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return self.fallback_responses["lingering"]
    
    def analyze_trajectory(self, track_data):
        """
        Analyze movement trajectory for unusual patterns
        
        Args:
            track_data: Dictionary containing tracking information
            
        Returns:
            str: Reasoning about the trajectory
        """
        if self.llm is None:
            # Fallback if LLM is not available
            return self.fallback_responses["normal"]
        
        # Create prompt for the LLM
        prompt = self._create_trajectory_prompt(track_data)
        
        # Query the LLM
        try:
            response = self.llm(
                prompt,
                max_tokens=256,
                stop=["Human:", "AI:"],
                echo=False
            )
            
            reasoning = response['choices'][0]['text'].strip()
            return reasoning
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return "No unusual patterns detected in movement."
    
    def generate_summary(self, events, tracks):
        """
        Generate a summary of all events
        
        Args:
            events: List of events
            tracks: Dictionary of track data
            
        Returns:
            str: Summary of events
        """
        if self.llm is None:
            # Fallback if LLM is not available
            return "Summary not available without LLM."
        
        # Create prompt for the LLM
        prompt = self._create_summary_prompt(events, tracks)
        
        # Query the LLM
        try:
            response = self.llm(
                prompt,
                max_tokens=512,
                stop=["Human:", "AI:"],
                echo=False
            )
            
            summary = response['choices'][0]['text'].strip()
            return summary
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return "Unable to generate summary."
    
    def _create_event_prompt(self, event_data, track_data):
        """
        Create a prompt for event analysis
        
        Args:
            event_data: Dictionary containing event information
            track_data: Dictionary containing tracking information
            
        Returns:
            str: Prompt for the LLM
        """
        # Convert track data to a more readable format
        track_summary = f"Track ID: {track_data.get('id', 'Unknown')}\n"
        track_summary += f"Class ID: {track_data.get('class_id', 'Unknown')}\n"
        track_summary += f"ROI visits: {json.dumps(track_data.get('roi_visits', {}))}\n"
        track_summary += f"Items picked: {track_data.get('items_picked', [])}\n"
        
        # Create the prompt
        prompt = f"""Human: You are an AI security analyst for a retail store. 
Analyze the following surveillance event and provide a brief explanation of what might be happening.
Be concise and specific about whether this represents suspicious behavior and why.

Event Type: {event_data.get('type', 'Unknown')}
Description: {event_data.get('description', 'No description')}
From ROI: {event_data.get('from_roi', 'None')}
To ROI: {event_data.get('to_roi', 'None')}

Track Information:
{track_summary}

Is this behavior suspicious? Why or why not?

AI: """
        
        return prompt
    
    def _create_dwell_prompt(self, event_data, track_data):
        """
        Create a prompt for dwell time analysis
        
        Args:
            event_data: Dictionary containing event information
            track_data: Dictionary containing tracking information
            
        Returns:
            str: Prompt for the LLM
        """
        # Create the prompt
        prompt = f"""Human: You are an AI security analyst for a retail store.
Analyze the following dwell time event and provide a brief explanation of what might be happening.
Be concise and specific about whether this represents suspicious behavior and why.

Event Type: {event_data.get('type', 'Unknown')}
Description: {event_data.get('description', 'No description')}
ROI Name: {event_data.get('roi_name', 'Unknown')}
Dwell Time: {event_data.get('dwell_time', 0)} seconds

Track Information:
ROI visits: {json.dumps(track_data.get('roi_visits', {}))}
Items picked: {track_data.get('items_picked', [])}

Consider that:
- Long dwell times in electronics/jewelry sections might indicate casing for theft
- Long dwell times in normal product areas might just be shopping
- Context matters (did they pick up items? visit cashier?)

Is this behavior suspicious? Why or why not?

AI: """
        
        return prompt
    
    def _create_trajectory_prompt(self, track_data):
        """
        Create a prompt for trajectory analysis
        
        Args:
            track_data: Dictionary containing tracking information
            
        Returns:
            str: Prompt for the LLM
        """
        # Summarize trajectory
        roi_sequence = []
        for roi_name, visit_data in track_data.get('roi_visits', {}).items():
            enter_frame = visit_data.get('enter_frame', 0)
            roi_sequence.append((enter_frame, roi_name))
        
        # Sort by frame number
        roi_sequence.sort()
        roi_path = " -> ".join([roi for _, roi in roi_sequence])
        
        # Create the prompt
        prompt = f"""Human: You are an AI security analyst for a retail store.
Analyze the following customer trajectory and provide a brief explanation of what might be happening.
Be concise and specific about whether this represents normal shopping or suspicious behavior.

Track Class: {"Person" if track_data.get('class_id', -1) == 0 else "Object"}
ROI Path: {roi_path if roi_path else "No path recorded"}
Items Picked: {track_data.get('items_picked', [])}

Consider that:
- Normal shopping usually involves browsing multiple areas, then proceeding to checkout
- Suspicious behavior might include direct paths from shelf to exit
- Visiting high-value areas without picking items might indicate casing
- Multiple store visits without purchases might be suspicious

What does this trajectory suggest about the customer's behavior?

AI: """
        
        return prompt
    
    def _create_summary_prompt(self, events, tracks):
        """
        Create a prompt for generating a summary
        
        Args:
            events: List of events
            tracks: Dictionary of track data
            
        Returns:
            str: Prompt for the LLM
        """
        # Summarize events
        event_summary = ""
        for i, event in enumerate(events[:10]):  # Limit to first 10 events
            event_summary += f"{i+1}. {event.get('type', 'Unknown')}: {event.get('description', 'No description')}\n"
        
        # Create the prompt
        prompt = f"""Human: You are an AI security analyst for a retail store.
Generate a concise summary of the following surveillance events. 
Focus on identifying any suspicious behavior or potential security threats.

Events:
{event_summary}

Total People Tracked: {sum(1 for t in tracks.values() if t.get('class_id', -1) == 0)}
Total Objects Tracked: {sum(1 for t in tracks.values() if t.get('class_id', -1) != 0)}

Please provide:
1. A brief overview of the activity
2. Any suspicious behavior detected
3. Recommendations for security staff

AI: """
        
        return prompt 