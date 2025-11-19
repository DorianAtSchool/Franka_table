"""
Franka Table Environments

Gymnasium-compatible wrappers for MuJoCo simulations of Franka Panda robots.

This package focuses on the 4-robot environment.
"""

from .franka_4robots_env import FrankaTable4RobotsEnv

__all__ = ['FrankaTable4RobotsEnv']
