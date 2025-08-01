"""AODT Visualization Package

This package provides visualization capabilities for the AODT simulation.
It can be run directly as a module to visualize simulation results.

Usage:
    python -m aodt_pso.visualization
"""

from .visualizer import generate_visualization

# Allow running the package as a module
if __name__ == "__main__":
    from .visualizer import main
    main()
