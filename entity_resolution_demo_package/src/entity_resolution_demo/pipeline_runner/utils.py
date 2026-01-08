#!/usr/bin/env python3
"""
Utilities for Entity Resolution Pipeline Demo

Common utilities and helper functions used across the pipeline modules.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

# Setup logging
logger = logging.getLogger(__name__)

# State directory
STATE_DIR = Path(__file__).parent.parent.parent.parent / "pipeline_state"
STATE_DIR.mkdir(exist_ok=True)

def save_state(stage_name: str, state_data: Dict[str, Any], state_dir: Optional[str] = None) -> bool:
    """
    Save stage state to a JSON file
    
    Args:
        stage_name: Name of the pipeline stage
        state_data: Dictionary of state data to save
        state_dir: Optional directory path for state files (defaults to STATE_DIR)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a serializable version of the state data
        serializable_data = {}
        for key, value in state_data.items():
            if hasattr(value, '__dict__'):
                # Convert objects to dictionaries
                serializable_data[key] = value.__dict__
            elif isinstance(value, list) and value and hasattr(value[0], '__dict__'):
                # Convert list of objects to list of dictionaries
                serializable_data[key] = [item.__dict__ for item in value]
            else:
                # Use as is
                serializable_data[key] = value
        
        # Save to file
        if state_dir:
            state_dir_path = Path(state_dir)
            state_dir_path.mkdir(exist_ok=True)
            state_file = state_dir_path / f"{stage_name}_state.json"
        else:
            state_file = STATE_DIR / f"{stage_name}_state.json"
        
        with open(state_file, 'w') as f:
            json.dump(serializable_data, f, default=str, indent=2)
        
        logger.info(f"Saved state for {stage_name} to {state_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving state for {stage_name}: {e}")
        return False

def load_state(stage_name: str, state_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load stage state from a JSON file
    
    Args:
        stage_name: Name of the pipeline stage
        state_dir: Optional directory path for state files (defaults to STATE_DIR)
        
    Returns:
        dict: State data or None if not found
    """
    if state_dir:
        state_dir_path = Path(state_dir)
        state_file = state_dir_path / f"{stage_name}_state.json"
    else:
        state_file = STATE_DIR / f"{stage_name}_state.json"
    try:
        if state_file.exists():
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            logger.info(f"Loaded state for {stage_name} from {state_file}")
            return state_data
        else:
            logger.warning(f"No saved state found for {stage_name}")
            return None
    except Exception as e:
        logger.error(f"Error loading state for {stage_name}: {e}")
        return None

def print_header(title: str) -> None:
    """
    Print a formatted header
    
    Args:
        title: Header title
    """
    print(f"\n{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 30} {title} {'=' * (48 - len(title))}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}\n")

def print_subheader(title: str) -> None:
    """
    Print a formatted subheader
    
    Args:
        title: Subheader title
    """
    print(f"\n{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-' * 5} {title} {'-' * (52 - len(title))}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}\n")

def print_success(message: str) -> None:
    """
    Print a success message
    
    Args:
        message: Success message
    """
    print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")

def print_warning(message: str) -> None:
    """
    Print a warning message
    
    Args:
        message: Warning message
    """
    print(f"{Fore.YELLOW}⚠️ {message}{Style.RESET_ALL}")

def print_error(message: str) -> None:
    """
    Print an error message
    
    Args:
        message: Error message
    """
    print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")

def print_info(message: str) -> None:
    """
    Print an info message
    
    Args:
        message: Info message
    """
    print(f"{Fore.BLUE}ℹ️ {message}{Style.RESET_ALL}")

def print_entity(entity: Dict[str, Any]) -> None:
    """
    Print entity information
    
    Args:
        entity: Entity dictionary
    """
    name = entity.get('name', 'Unknown')
    entity_type = entity.get('entity_type', 'unknown')
    description = entity.get('description', '')
    aliases = entity.get('aliases', [])
    
    print(f"{Fore.GREEN}Entity: {name}{Style.RESET_ALL}")
    print(f"  Type: {entity_type}")
    if description:
        print(f"  Description: {description}")
    if aliases:
        print(f"  Aliases: {', '.join(aliases)}")

def print_match(match: Dict[str, Any]) -> None:
    """
    Print match information
    
    Args:
        match: Match dictionary
    """
    extracted = match.get('extracted_entity', {})
    watched = match.get('watched_entity', {})
    confidence = match.get('confidence', 0.0)
    match_type = match.get('match_type', 'unknown')
    explanation = match.get('explanation', '')
    
    # Get names from entities
    extracted_name = extracted.get('name', 'Unknown') if isinstance(extracted, dict) else str(extracted)
    watched_name = watched.get('name', 'Unknown') if isinstance(watched, dict) else str(watched)
    
    # Color based on confidence
    color = Fore.GREEN if confidence >= 0.8 else Fore.YELLOW if confidence >= 0.5 else Fore.RED
    
    print(f"{color}Match: {extracted_name} → {watched_name}{Style.RESET_ALL}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Type: {match_type}")
    if explanation:
        print(f"  Explanation: {explanation[:100]}..." if len(explanation) > 100 else f"  Explanation: {explanation}")

def time_function(func):
    """
    Decorator to time a function
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
