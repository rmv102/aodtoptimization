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


def plot_scene(rus: List[RU], ues: List[UE], area_bounds: tuple, animate=False, iterations=None):
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
    
    # Create a PyVista plotter with improved settings
    plotter = pv.Plotter(window_size=[1024, 768])
    plotter.set_background('white', top='aliceblue')  # Gradient background for better depth perception
    
    # Set better camera and interaction settings
    plotter.camera_position = 'iso'  # Isometric view
    plotter.enable_trackball_style()  # Better rotation control

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
    plotter.enable_zoom_style()  # Fixed API call for newer PyVista versions
    plotter.add_axes(interactive=True)  # Interactive axes for better orientation
    plotter.add_floor('-z', pad=0.1, color='lightgrey', opacity=0.5)  # More visible floor
    plotter.add_legend(bcolor=(0.9, 0.9, 0.9, 0.3), border=True, size=(0.2, 0.2))  # More visible legend
    
    # Add orientation widget for better navigation
    plotter.add_orientation_widget()

    print("\n--- Displaying 3D Visualization ---")
    
    if animate and iterations:
        print("Animation mode: Space to Play/Pause, 'n' for Next frame, 'p' for Previous frame")
        
        # Store references to RU actors for animation updates
        ru_actors = []
        ru_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange']
        
        # Create text display for iteration and fitness
        text = plotter.add_text(f"Iteration: 0, Fitness: 0", position='upper_left', font_size=14, color='black')
        
        # Animation state variables
        current_frame = 0
        playing = True
        total_frames = len(iterations)
        
        # Create initial RU actors - we'll update these during animation
        for i, ru in enumerate(rus):
            color = ru_colors[i % len(ru_colors)]
            # Create a cone for the RU
            cone = pv.Cone(
                center=ru.position, 
                direction=[0, 0, -1], 
                height=20,  # Larger height
                radius=10   # Larger radius
            )
            # Add the mesh with a specific name for later reference
            actor = plotter.add_mesh(cone, color=color, name=f'ru_{i}')
            ru_actors.append(actor)
        
        def update_frame(frame_idx):
            """Update the visualization for a specific frame"""
            frame = iterations[frame_idx]
            particles = frame['particles']
            
            # Update text display
            plotter.remove_actor(text)
            new_text = f"Iteration: {frame['iteration']:.1f}, Fitness: {frame['fitness']:.2f}"
            new_text_actor = plotter.add_text(new_text, position='upper_left', font_size=14, color='black')
            
            # Update RU positions
            for i, ru in enumerate(particles):
                if i < len(ru_actors):
                    # Remove the old actor
                    plotter.remove_actor(ru_actors[i])
                    
                    # Create a new cone at the updated position
                    color = ru_colors[i % len(ru_colors)]
                    cone = pv.Cone(
                        center=ru.position, 
                        direction=[0, 0, -1], 
                        height=20, 
                        radius=10
                    )
                    # Add the new mesh and update our reference
                    ru_actors[i] = plotter.add_mesh(cone, color=color, name=f'ru_{i}')
            
            # Force render update
            plotter.render()
        
        def key_press_callback(key):
            nonlocal current_frame, playing
            
            if key == ' ':  # Space bar - toggle play/pause
                playing = not playing
                if playing:
                    print("Animation playing")
                else:
                    print("Animation paused")
            
            elif key == 'n':  # Next frame
                playing = False  # Pause when manually navigating
                current_frame = (current_frame + 1) % total_frames
                update_frame(current_frame)
                print(f"Frame: {current_frame}/{total_frames-1}")
            
            elif key == 'p':  # Previous frame
                playing = False  # Pause when manually navigating
                current_frame = (current_frame - 1) % total_frames
                update_frame(current_frame)
                print(f"Frame: {current_frame}/{total_frames-1}")
        
        # Register key press callback
        plotter.add_key_event(' ', key_press_callback)  # Space
        plotter.add_key_event('n', key_press_callback)  # Next
        plotter.add_key_event('p', key_press_callback)  # Previous
        
        # Animation timer callback
        def timer_callback(obj, event):
            nonlocal current_frame
            if playing:
                current_frame = (current_frame + 1) % total_frames
                update_frame(current_frame)
        
        # Create a timer for animation
        plotter.iren.create_timer(1000)  # milliseconds
        plotter.iren.add_observer('TimerEvent', timer_callback)
        
        # Show instructions
        plotter.add_text(
            "Controls:\nSpace: Play/Pause\nn: Next Frame\np: Previous Frame", 
            position=(10, 60), 
            font_size=12, 
            color='black'
        )
        
        # Show the plot with animation
        print("Close the PyVista window to exit the program.")
        plotter.show()
    else:
        # Show the static plot
        print("Close the PyVista window to exit the program.")
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