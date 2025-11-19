#!/usr/bin/env python3
"""
Generate a scene with 4 Franka robots, one at each corner of the table.
"""

import re

# Robot positions (one at each corner, facing inward)
# Table is 1.2m x 1.0m centered at origin
# Corner positions: (+/-0.45, +/-0.35)
robot_configs = [
    {
        "prefix": "robot1_",
        "pos": "0.45 0.35 0.48",
        "quat": "0.3826834 0 0 0.9238795",  # -45 degrees (facing inward)
        "comment": "Robot 1: Top-right corner"
    },
    {
        "prefix": "robot2_",
        "pos": "-0.45 0.35 0.48",
        "quat": "0.9238795 0 0 0.3826834",  # -135 degrees (facing inward)
        "comment": "Robot 2: Top-left corner"
    },
    {
        "prefix": "robot3_",
        "pos": "-0.45 -0.35 0.48",
        "quat": "0.9238795 0 0 -0.3826834",  # 135 degrees (facing inward)
        "comment": "Robot 3: Bottom-left corner"
    },
    {
        "prefix": "robot4_",
        "pos": "0.45 -0.35 0.48",
        "quat": "0.3826834 0 0 -0.9238795",  # 45 degrees (facing inward)
        "comment": "Robot 4: Bottom-right corner"
    }
]

def add_prefix_to_names(xml_text, prefix, elements=["name=", "joint=", "joint1=", "joint2=", "tendon="]):
    """Add prefix to all name attributes in XML."""
    result = xml_text
    for elem in elements:
        # Find all instances of elem"name" and replace with elem"prefix_name"
        pattern = f'({elem}")(\\w+)(")'
        replacement = f'\\1{prefix}\\2\\3'
        result = re.sub(pattern, replacement, result)
    return result

# Read the original scene.xml
with open('scene.xml', 'r') as f:
    content = f.read()

# Extract the robot body definition (from first link0 to its closing)
robot_start = content.find('<!-- Franka Panda robot')
robot_end = content.find('</body>\n  </worldbody>', robot_start)

if robot_start == -1 or robot_end == -1:
    print("Error: Could not find robot definition in scene.xml")
    exit(1)

# Extract just the robot body (excluding comment)
robot_def_start = content.find('<body name="link0"', robot_start)
robot_def_end = robot_end + len('</body>')
original_robot = content[robot_def_start:robot_def_end]

# Extract worldbody start and the content before the first robot
worldbody_start = content.find('<worldbody>')
before_robots = content[worldbody_start:robot_start]

# Extract tendon, equality, actuator sections
tendon_start = content.find('<tendon>')
tendon_end = content.find('</tendon>') + len('</tendon>')
original_tendon = content[tendon_start:tendon_end]

equality_start = content.find('<equality>')
equality_end = content.find('</equality>') + len('</equality>')
original_equality = content[equality_start:equality_end]

actuator_start = content.find('<actuator>')
actuator_end = content.find('</actuator>') + len('</actuator>')
original_actuator = content[actuator_start:actuator_end]

# Generate new robots with prefixed names
new_robots = []
new_tendons = []
new_equalities = []
new_actuators = []

for config in robot_configs:
    prefix = config["prefix"]
    comment = config["comment"]
    pos = config["pos"]
    quat = config["quat"]
    
    # Create robot with unique names
    robot = f'\n    <!-- {comment} -->\n'
    robot_body = add_prefix_to_names(original_robot, prefix)
    # Update position and orientation
    robot_body = re.sub(r'<body name="' + prefix + 'link0" pos="[^"]*" quat="[^"]*"',
                       f'<body name="{prefix}link0" pos="{pos}" quat="{quat}"',
                       robot_body)
    robot += robot_body
    new_robots.append(robot)
    
    # Create tendon with unique names
    tendon = add_prefix_to_names(original_tendon, prefix)
    tendon = tendon.replace('<tendon>', f'\n    <!-- {comment} tendons -->').replace('</tendon>', '')
    new_tendons.append(tendon)
    
    # Create equality with unique names
    equality = add_prefix_to_names(original_equality, prefix)
    equality = equality.replace('<equality>', '').replace('</equality>', '')
    new_equalities.append(equality)
    
    # Create actuators with unique names
    actuator = add_prefix_to_names(original_actuator, prefix)
    actuator = actuator.replace('<actuator>', f'\n    <!-- {comment} actuators -->').replace('</actuator>', '')
    new_actuators.append(actuator)

# Build new scene
new_scene = content[:worldbody_start]
new_scene += '<worldbody>'
new_scene += before_robots.replace('<worldbody>', '')  # Remove duplicate worldbody tag
new_scene += ''.join(new_robots)
new_scene += '\n  </worldbody>\n'

# Add all tendons
new_scene += '\n  <tendon>'
new_scene += ''.join(new_tendons)
new_scene += '\n  </tendon>\n'

# Add all equalities
new_scene += '\n  <equality>'
new_scene += ''.join(new_equalities)
new_scene += '\n  </equality>\n'

# Add all actuators
new_scene += '\n  <actuator>'
new_scene += ''.join(new_actuators)
new_scene += '\n  </actuator>\n'

# Add keyframe and contact sections (update for all robots)
keyframe_section = '\n  <keyframe>\n'
for i, config in enumerate(robot_configs, 1):
    prefix = config["prefix"]
    # Create qpos for this robot (7 joint positions + 2 finger positions)
    robot_qpos = "0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04"
    robot_ctrl = "0 0 0 -1.57079 0 1.57079 -0.7853 255"
    keyframe_section += f'    <key name="home_{i}" qpos="{robot_qpos}" ctrl="{robot_ctrl}"/>\n'
keyframe_section += '  </keyframe>\n'
new_scene += keyframe_section

# Add contact exclusions for all robots
new_scene += '\n  <contact>\n'
for config in robot_configs:
    prefix = config["prefix"]
    new_scene += f'    <exclude body1="{prefix}link0" body2="{prefix}link1"/>\n'
new_scene += '  </contact>\n'

new_scene += '</mujoco>\n'

# Write the new scene
with open('scene_4robots.xml', 'w') as f:
    f.write(new_scene)

print("Generated scene_4robots.xml with 4 Franka robots")
print("Robot positions:")
for i, config in enumerate(robot_configs, 1):
    print(f"  Robot {i}: {config['comment']}")
    print(f"    Position: {config['pos']}")
    print(f"    Prefix: {config['prefix']}")
