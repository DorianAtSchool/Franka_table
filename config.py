"""
Path configuration for franka_table project.

This module provides standardized paths for all scripts to use,
ensuring consistent file locations regardless of where scripts are run from.
"""

import os
from pathlib import Path

# Get the base directory (franka_table/)
BASE_DIR = Path(__file__).parent.absolute()

# Directory paths
SCENES_DIR = BASE_DIR / "scenes"
DOCS_DIR = BASE_DIR / "docs"
ENVIRONMENTS_DIR = BASE_DIR / "environments"
SCRIPTS_DIR = BASE_DIR / "scripts"
UTILS_DIR = BASE_DIR / "utils"
OUTPUTS_DIR = BASE_DIR / "outputs"
SHELL_DIR = BASE_DIR / "shell"

# Output subdirectories
VIDEOS_DIR = OUTPUTS_DIR / "videos"
IMAGES_DIR = OUTPUTS_DIR / "images"

# Scene files
SCENE_XML = SCENES_DIR / "scene.xml"
SCENE_4ROBOTS_XML = SCENES_DIR / "scene_4robots.xml"

# Ensure output directories exist
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def get_scene_path(scene_name: str = "scene.xml") -> str:
    """Get the full path to a scene file."""
    return str(SCENES_DIR / scene_name)

def get_output_path(filename: str, output_type: str = "images") -> str:
    """
    Get the full path for an output file.
    
    Args:
        filename: Name of the output file
        output_type: Either 'images' or 'videos'
    
    Returns:
        Full path to the output file
    """
    if output_type == "videos":
        return str(VIDEOS_DIR / filename)
    elif output_type == "images":
        return str(IMAGES_DIR / filename)
    else:
        return str(OUTPUTS_DIR / filename)

def get_doc_path(doc_name: str) -> str:
    """Get the full path to a documentation file."""
    return str(DOCS_DIR / doc_name)

# Print configuration when imported (can be disabled if needed)
if __name__ == "__main__":
    print("Franka Table Path Configuration")
    print("=" * 50)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"SCENES_DIR: {SCENES_DIR}")
    print(f"OUTPUTS_DIR: {OUTPUTS_DIR}")
    print(f"  - VIDEOS_DIR: {VIDEOS_DIR}")
    print(f"  - IMAGES_DIR: {IMAGES_DIR}")
    print(f"\nScene files:")
    print(f"  - scene.xml: {SCENE_XML}")
    print(f"  - scene_4robots.xml: {SCENE_4ROBOTS_XML}")
