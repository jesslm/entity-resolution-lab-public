"""
State Management Utilities for Modular Notebook Architecture

This module provides utilities for managing state files across the modular notebook
architecture, enabling independent execution of each pipeline stage.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class StateManager:
    """Manages state files for the modular notebook architecture."""
    
    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize the StateManager.
        
        Args:
            state_dir: Directory to store state files. Defaults to pipeline_state.
        """
        if state_dir is None:
            # Default to pipeline_state directory in project root
            current_file = Path(__file__)
            # Go up from src/entity_resolution_demo/tools/state_manager.py to project root
            project_root = current_file.parent.parent.parent.parent
            state_dir = project_root / "pipeline_state"
        
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, stage: str, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save state data for a pipeline stage.
        
        Args:
            stage: Pipeline stage name (e.g., 'entity_preparation', 'article_processing')
            data: State data to save
            config: Optional configuration data
            
        Returns:
            Path to the saved state file
        """
        timestamp = datetime.now().isoformat()
        
        state_data = {
            "stage": stage,
            "timestamp": timestamp,
            "version": "1.0",
            "config": config or {},
            "results": data,
            "metadata": {
                "execution_time": data.get("execution_time", 0),
                "success": data.get("success", True),
                "errors": data.get("errors", [])
            }
        }
        
        state_file = self.state_dir / f"{stage}_state.json"
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        return state_file
    
    def load_state(self, stage: str) -> Optional[Dict[str, Any]]:
        """
        Load state data for a pipeline stage.
        
        Args:
            stage: Pipeline stage name
            
        Returns:
            State data dictionary or None if not found
        """
        state_file = self.state_dir / f"{stage}_state.json"
        
        if not state_file.exists():
            return None
        
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load state file {state_file}: {e}")
            return None
    
    def state_exists(self, stage: str) -> bool:
        """
        Check if state file exists for a pipeline stage.
        
        Args:
            stage: Pipeline stage name
            
        Returns:
            True if state file exists, False otherwise
        """
        state_file = self.state_dir / f"{stage}_state.json"
        return state_file.exists()
    
    def get_state_info(self, stage: str) -> Optional[Dict[str, Any]]:
        """
        Get basic information about a state file without loading full data.
        
        Args:
            stage: Pipeline stage name
            
        Returns:
            Basic state information or None if not found
        """
        state_data = self.load_state(stage)
        if not state_data:
            return None
        
        return {
            "stage": state_data.get("stage"),
            "timestamp": state_data.get("timestamp"),
            "version": state_data.get("version"),
            "success": state_data.get("metadata", {}).get("success"),
            "execution_time": state_data.get("metadata", {}).get("execution_time"),
            "file_size": (self.state_dir / f"{stage}_state.json").stat().st_size
        }
    
    def list_states(self) -> List[Dict[str, Any]]:
        """
        List all available state files.
        
        Returns:
            List of state information dictionaries
        """
        states = []
        
        for state_file in self.state_dir.glob("*_state.json"):
            stage = state_file.stem.replace("_state", "")
            state_info = self.get_state_info(stage)
            if state_info:
                states.append(state_info)
        
        return sorted(states, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    def clean_state(self, stage: str) -> bool:
        """
        Remove state file for a pipeline stage.
        
        Args:
            stage: Pipeline stage name
            
        Returns:
            True if file was removed, False if it didn't exist
        """
        state_file = self.state_dir / f"{stage}_state.json"
        
        if state_file.exists():
            state_file.unlink()
            return True
        
        return False
    
    def clean_all_states(self) -> int:
        """
        Remove all state files.
        
        Returns:
            Number of files removed
        """
        removed_count = 0
        
        for state_file in self.state_dir.glob("*_state.json"):
            state_file.unlink()
            removed_count += 1
        
        return removed_count
    
    def validate_dependencies(self, required_stages: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that required pipeline stages have completed successfully.
        
        Args:
            required_stages: List of required stage names
            
        Returns:
            Tuple of (all_valid, missing_stages)
        """
        missing_stages = []
        
        for stage in required_stages:
            state_info = self.get_state_info(stage)
            if not state_info or not state_info.get("success", False):
                missing_stages.append(stage)
        
        return len(missing_stages) == 0, missing_stages


def get_state_manager(state_dir: Optional[Path] = None) -> StateManager:
    """
    Get a StateManager instance.
    
    Args:
        state_dir: Optional custom state directory
        
    Returns:
        StateManager instance
    """
    return StateManager(state_dir)


def check_pipeline_readiness(required_stages: List[str], state_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Check if the pipeline is ready to run a specific stage.
    
    Args:
        required_stages: List of required stages that must be completed
        state_dir: Optional custom state directory
        
    Returns:
        Tuple of (is_ready, message)
    """
    state_manager = get_state_manager(state_dir)
    
    all_valid, missing_stages = state_manager.validate_dependencies(required_stages)
    
    if all_valid:
        return True, "All required stages completed successfully"
    else:
        missing_str = ", ".join(missing_stages)
        return False, f"Missing or failed stages: {missing_str}"


def create_state_summary(state_dir: Optional[Path] = None) -> str:
    """
    Create a summary of all pipeline states.
    
    Args:
        state_dir: Optional custom state directory
        
    Returns:
        HTML summary string
    """
    state_manager = get_state_manager(state_dir)
    states = state_manager.list_states()
    
    if not states:
        return "<p>No pipeline states found. Run individual pipeline stages to create state files.</p>"
    
    html = ["<h3>Pipeline State Summary</h3>", "<table border='1' style='border-collapse: collapse;'>"]
    html.append("<tr><th>Stage</th><th>Status</th><th>Timestamp</th><th>Execution Time</th><th>File Size</th></tr>")
    
    for state in states:
        status = "✅ Success" if state.get("success") else "❌ Failed"
        timestamp = state.get("timestamp", "Unknown")
        exec_time = f"{state.get('execution_time', 0):.2f}s"
        file_size = f"{state.get('file_size', 0) / 1024:.1f} KB"
        
        html.append(f"<tr><td>{state.get('stage', 'Unknown')}</td><td>{status}</td><td>{timestamp}</td><td>{exec_time}</td><td>{file_size}</td></tr>")
    
    html.append("</table>")
    return "\n".join(html)


# Convenience functions for common operations
def save_entity_preparation_state(data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Path:
    """Save entity preparation state."""
    return get_state_manager().save_state("entity_preparation", data, config)


def load_entity_preparation_state() -> Optional[Dict[str, Any]]:
    """Load entity preparation state."""
    return get_state_manager().load_state("entity_preparation")


def save_article_processing_state(data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Path:
    """Save article processing state."""
    return get_state_manager().save_state("article_processing", data, config)


def load_article_processing_state() -> Optional[Dict[str, Any]]:
    """Load article processing state."""
    return get_state_manager().load_state("article_processing")


def save_entity_matching_state(data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Path:
    """Save entity matching state."""
    return get_state_manager().save_state("entity_matching", data, config)


def load_entity_matching_state() -> Optional[Dict[str, Any]]:
    """Load entity matching state."""
    return get_state_manager().load_state("entity_matching")
