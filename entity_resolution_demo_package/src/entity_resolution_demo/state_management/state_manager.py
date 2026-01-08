"""State management utilities for the Entity Resolution Pipeline.

This module provides utilities for loading and saving state.
"""

import os
import json
import shutil
from typing import Dict, Optional
from datetime import datetime


def load_state(state_file: str) -> Dict:
    """Load state from a JSON file.
    
    Args:
        state_file (str): Path to the state file
    
    Returns:
        Dict: State dictionary
    
    Raises:
        FileNotFoundError: If state_file is not found
        json.JSONDecodeError: If state_file is not valid JSON
    """
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"State file not found: {state_file}")
    
    with open(state_file, 'r') as f:
        return json.load(f)


def save_state(state_file: str, state: Dict) -> None:
    """Save state to a JSON file.
    
    Creates a backup of the existing state file if it exists.
    
    Args:
        state_file (str): Path to the state file
        state (Dict): State dictionary
    
    Raises:
        IOError: If state_file cannot be written
    """
    # Create backup if state file exists
    if os.path.exists(state_file):
        backup_file = f"{state_file}.bak"
        shutil.copy2(state_file, backup_file)
    
    # Ensure state has timestamp
    if "timestamp" not in state:
        state["timestamp"] = datetime.now().isoformat()
    
    # Ensure state has version
    if "version" not in state:
        state["version"] = "1.0"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    
    # Write state to file
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


class StateManager:
    """Manages state for the Entity Resolution Pipeline.
    
    This class provides methods for loading, saving, and validating state.
    
    Attributes:
        state_dir (str): Directory for state files
    """
    
    def __init__(self, state_dir: Optional[str] = None):
        """Initialize the state manager.
        
        Args:
            state_dir (str, optional): Directory for state files.
                If None, uses 'state' directory in current working directory.
        """
        self.state_dir = state_dir or os.path.join(os.getcwd(), "state")
        os.makedirs(self.state_dir, exist_ok=True)
    
    def get_state_file_path(self, stage: str) -> str:
        """Get the path to a state file for a specific stage.
        
        Args:
            stage (str): Pipeline stage ('entity_preparation', 'article_processing', 'entity_matching')
        
        Returns:
            str: Path to the state file
        """
        return os.path.join(self.state_dir, f"{stage}_state.json")
    
    def load_state(self, stage: str) -> Dict:
        """Load state for a specific stage.
        
        Args:
            stage (str): Pipeline stage ('entity_preparation', 'article_processing', 'entity_matching')
        
        Returns:
            Dict: State dictionary
        
        Raises:
            FileNotFoundError: If state file is not found
        """
        state_file = self.get_state_file_path(stage)
        return load_state(state_file)
    
    def save_state(self, stage: str, state: Dict) -> None:
        """Save state for a specific stage.
        
        Args:
            stage (str): Pipeline stage ('entity_preparation', 'article_processing', 'entity_matching')
            state (Dict): State dictionary
        """
        state_file = self.get_state_file_path(stage)
        save_state(state_file, state)
    
    def state_exists(self, stage: str) -> bool:
        """Check if state exists for a specific stage.
        
        Args:
            stage (str): Pipeline stage ('entity_preparation', 'article_processing', 'entity_matching')
        
        Returns:
            bool: True if state exists, False otherwise
        """
        state_file = self.get_state_file_path(stage)
        return os.path.exists(state_file)
    
    def validate_state(self, stage: str, required_version: str = "1.0") -> bool:
        """Validate state for a specific stage.
        
        Args:
            stage (str): Pipeline stage ('entity_preparation', 'article_processing', 'entity_matching')
            required_version (str, optional): Required state version
        
        Returns:
            bool: True if state is valid, False otherwise
        """
        try:
            state = self.load_state(stage)
            return state.get("version", "0.0") == required_version
        except (FileNotFoundError, json.JSONDecodeError):
            return False
