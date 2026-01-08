#!/usr/bin/env python3
"""
Reset Entity Preparation

This script resets the entity preparation process by deleting both
entity_preparation_state.json and entity_watch_list.json files.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import local modules
from entity_resolution_demo.utils import (
    print_header, print_success, print_warning, print_error, print_info
)

def reset_preparation():
    """Reset entity preparation by deleting state and watch list files"""
    print_header("Resetting Entity Preparation")
    
    # Delete entity preparation state file
    state_file = Path(__file__).parent / "state" / "entity_preparation_state.json"
    if state_file.exists():
        os.remove(state_file)
        print_success(f"Deleted entity preparation state file: {state_file}")
    else:
        print_info(f"Entity preparation state file not found: {state_file}")
    
    # Delete entity watch list file from state directory
    watch_list_file = Path(__file__).parent / "state" / "entity_watch_list.json"
    if watch_list_file.exists():
        # Create a backup before deleting
        backup_file = Path(__file__).parent / "state" / "entity_watch_list.json.bak"
        os.rename(watch_list_file, backup_file)
        print_success(f"Backed up entity watch list to: {backup_file}")
    else:
        print_info(f"Entity watch list file not found: {watch_list_file}")
    
    print_success("Entity preparation reset successfully")
    print_info("You can now run entity_preparation.py to create a fresh entity watch list and state")
    
    return True

if __name__ == "__main__":
    reset_preparation()
