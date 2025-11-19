#!/usr/bin/env python3
"""
Utility script to resize the table in scene_4robots.xml while keeping robots at corners.

Usage:
    python resize_table.py 1.8 1.6  # Sets table to 1.8m x 1.6m
"""

import sys
import re
from pathlib import Path


def resize_table(xml_path, table_length, table_width):
    """
    Resize the table and reposition robots at corners.
    
    Args:
        xml_path: Path to the scene XML file
        table_length: New table length in X direction (meters)
        table_width: New table width in Y direction (meters)
    """
    # Calculate robot positions (at corners, slightly inside table edge)
    # Robots are positioned 0.05m inside from the edge
    robot_x = (table_length / 2) - 0.25  # 0.25m from edge
    robot_y = (table_width / 2) - 0.25   # 0.25m from edge
    
    # MuJoCo uses half-sizes for box geoms
    half_length = table_length / 2
    half_width = table_width / 2
    
    print(f"Resizing table to {table_length}m x {table_width}m")
    print(f"Table half-sizes: {half_length} x {half_width}")
    print(f"Robot positions: ±{robot_x} x ±{robot_y}")
    print(f"Target position: 0, 0 (center)")
    print()
    
    # Read the XML file
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update table comment
    content = re.sub(
        r'<!-- Table: centered at origin, [0-9.]+m x [0-9.]+m x [0-9.]+m height -->',
        f'<!-- Table: centered at origin, {table_length}m x {table_width}m x 0.48m height -->',
        content
    )
    
    # Update table size
    content = re.sub(
        r'<body name="table" pos="0 0 0\.24">\s*<geom type="box" size="[0-9.]+ [0-9.]+ [0-9.]+"',
        f'<body name="table" pos="0 0 0.24">\n      <geom type="box" size="{half_length} {half_width} 0.24"',
        content
    )
    
    # Update robot positions
    # Robot 1: Top-right corner
    content = re.sub(
        r'(<body name="robot1_link0" pos=")[^"]+(" quat="[^"]+" childclass="panda">)',
        rf'\g<1>{robot_x} {robot_y} 0.48\g<2>',
        content
    )
    
    # Robot 2: Top-left corner
    content = re.sub(
        r'(<body name="robot2_link0" pos=")[^"]+(" quat="[^"]+" childclass="panda">)',
        rf'\g<1>{-robot_x} {robot_y} 0.48\g<2>',
        content
    )
    
    # Robot 3: Bottom-left corner
    content = re.sub(
        r'(<body name="robot3_link0" pos=")[^"]+(" quat="[^"]+" childclass="panda">)',
        rf'\g<1>{-robot_x} {-robot_y} 0.48\g<2>',
        content
    )
    
    # Robot 4: Bottom-right corner
    content = re.sub(
        r'(<body name="robot4_link0" pos=")[^"]+(" quat="[^"]+" childclass="panda">)',
        rf'\g<1>{robot_x} {-robot_y} 0.48\g<2>',
        content
    )
    
    # Write back to file
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated {xml_path}")
    print(f"✓ Table size: {table_length}m x {table_width}m")
    print(f"✓ Robot 1 (Red):    ({robot_x:6.2f}, {robot_y:6.2f}, 0.48)")
    print(f"✓ Robot 2 (Cyan):   ({-robot_x:6.2f}, {robot_y:6.2f}, 0.48)")
    print(f"✓ Robot 3 (Yellow): ({-robot_x:6.2f}, {-robot_y:6.2f}, 0.48)")
    print(f"✓ Robot 4 (Green):  ({robot_x:6.2f}, {-robot_y:6.2f}, 0.48)")
    print(f"✓ Target (center):  (  0.00,   0.00, 0.58)")
    
    # Calculate max reach needed
    distance_to_center = (robot_x**2 + robot_y**2)**0.5
    print(f"\n✓ Distance from robot to center: {distance_to_center:.3f}m")
    print(f"  Franka Panda max reach: ~0.855m")
    if distance_to_center < 0.855:
        print(f"  ✓ Target is reachable! ({0.855 - distance_to_center:.3f}m margin)")
    else:
        print(f"  ⚠ Target might be out of reach! ({distance_to_center - 0.855:.3f}m over)")


def main():
    if len(sys.argv) != 3:
        print("Usage: python resize_table.py <length> <width>")
        print("Example: python resize_table.py 1.8 1.6")
        print("\nCurrent table sizes:")
        print("  Original: 1.6m x 1.4m")
        print("  Recommended: 1.8m x 1.6m (slightly larger)")
        print("  Current: 2.0m x 1.8m (may be too large)")
        sys.exit(1)
    
    try:
        table_length = float(sys.argv[1])
        table_width = float(sys.argv[2])
        
        if table_length <= 0 or table_width <= 0:
            print("Error: Table dimensions must be positive")
            sys.exit(1)
        
        if table_length < 1.0 or table_width < 1.0:
            print("Warning: Table might be too small for 4 robots")
        
        if table_length > 3.0 or table_width > 3.0:
            print("Warning: Table might be too large, target may be unreachable")
        
        xml_path = Path(__file__).parent / "scene_4robots.xml"
        
        if not xml_path.exists():
            print(f"Error: {xml_path} not found")
            sys.exit(1)
        
        resize_table(xml_path, table_length, table_width)
        
    except ValueError:
        print("Error: Invalid dimensions. Please provide numeric values.")
        sys.exit(1)


if __name__ == "__main__":
    main()
