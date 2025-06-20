"""
Provides 3D visualization capabilities for the AODT simulation results.

This module uses the PyVista library to render a 3D scene showing the final
positions of the Radio Units (RUs) and User Equipments (UEs).
"""
from typing import List
import numpy as np
import pyvista as pv

# Use a try-except block to handle running this file directly for testing
try:
    from ..aerial.dt import RU, UE
    from ..aerial.phy import Antenna
except (ImportError, ValueError):
    # This allows the script to be run directly for testing, assuming a certain structure
    import sys
    import os
    # Add the project root to the python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from aodt_pso.aerial.dt import RU, UE
    from aodt_pso.aerial.phy import Antenna
    from aodt_pso.simulation.signal_map import generate_ue_distribution


def plot_scene(rus: List[RU], ues: List[UE], area_bounds: tuple):
    """
    Renders a 3D scene of the simulation environment using PyVista.

    - RUs are represented as red cones.
    - UEs are represented as blue spheres.
    - The simulation area is outlined with a wireframe box.

    Args:
        rus (List[RU]): List of optimized RU objects.
        ues (List[UE]): List of UE objects in the scenario.
        area_bounds (tuple): The boundaries of the simulation area (min_x, max_x, min_y, max_y).
    """
    min_x, max_x, min_y, max_y = area_bounds
    
    # Create a PyVista plotter
    plotter = pv.Plotter(window_size=[1024, 768])
    plotter.set_background('white')

    # 1. Plot the User Equipments (UEs)
    if ues:
        ue_positions = np.array([ue.position for ue in ues])
        plotter.add_points(
            ue_positions,
            color='blue',
            render_points_as_spheres=True,
            point_size=10,
            label='User Equipments (UEs)'
        )

    # 2. Plot the Radio Units (RUs)
    if rus:
        for i, ru in enumerate(rus):
            # Represent RU as a cone pointing downwards
            cone = pv.Cone(
                center=ru.position, 
                direction=[0, 0, -1], 
                height=10, 
                radius=5
            )
            plotter.add_mesh(cone, color='red', label=f'RU-{i+1}')
            
            # Add a label with its configuration
            label_pos = ru.position + np.array([0, 0, 15]) # Position label above the cone
            freq = ru.antenna.freq
            elements = ru.antenna.elements
            plotter.add_point_labels(
                label_pos, 
                [f"RU-{i+1}\n{freq:.2f} GHz\n{elements} Elem."],
                font_size=12,
                text_color='black',
                shape=None,
                show_points=False
            )

    # 3. Plot the simulation area boundary
    bounds_box = pv.Box(bounds=[min_x, max_x, min_y, max_y, 0, 20])
    plotter.add_mesh(bounds_box, style='wireframe', color='gray', label='Simulation Area')

    # Configure plot settings
    plotter.view_isometric()
    plotter.enable_zoom_scaling()
    plotter.add_axes()
    plotter.add_floor('-z', pad=0.1)
    plotter.add_legend(bcolor=None, border=False)

    print("\n--- Displaying 3D Visualization ---")
    print("Close the PyVista window to exit the program.")
    
    # Show the plot
    plotter.show()

if __name__ == '__main__':
    # Example usage for testing
    # This block allows the script to be tested standalone
    
    # Define test parameters
    AREA_BOUNDS_TEST = (0, 500, 0, 500)
    NUM_UES_TEST = 50
    
    # Create mock RUs
    ru1_pos = np.array([100, 150, 15.0])
    ru1_ant = Antenna(freq=3.5, elements=4)
    ru1 = RU(position=ru1_pos, antenna=ru1_ant)
    
    ru2_pos = np.array([400, 350, 15.0])
    ru2_ant = Antenna(freq=5.2, elements=8)
    ru2 = RU(position=ru2_pos, antenna=ru2_ant)
    
    test_rus = [ru1, ru2]

    # Create mock UEs
    test_ues = generate_ue_distribution(num_ues=NUM_UES_TEST, area_bounds=AREA_BOUNDS_TEST)

    print("\n--- 3D Plot Test ---")
    print("This test demonstrates the plotting function with mock data.")
    
    # Plot the scene
    plot_scene(test_rus, test_ues, AREA_BOUNDS_TEST)
    
    print("\nTest finished.") 