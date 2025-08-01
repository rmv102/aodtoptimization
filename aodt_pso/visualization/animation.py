#!/usr/bin/env python3
"""
PSO Animation Module

This module provides functionality to animate the PSO optimization process,
showing particles moving towards the optimal base station configuration.
"""

import numpy as np
import pyvista as pv
import time
import pandas as pd
from ..aerial.dt import RU, UE
from ..aerial.phy import Antenna


def load_pso_iterations(csv_path):
    """
    Load all PSO iterations from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing optimization results.
        
    Returns:
        list: List of iteration data, where each item contains the particles for that iteration.
    """
    try:
        df = pd.read_csv(csv_path)
        iterations = []
        
        # Group by iteration
        for iter_num, iter_group in df.groupby('iteration'):
            particles = []
            for _, row in iter_group.iterrows():
                pos = np.array([float(row['pos_x']), float(row['pos_y']), float(row['pos_z'])])
                freq = float(row['frequency_ghz'])
                elements = int(row['elements'])
                ant = Antenna(freq=freq, elements=elements)
                particles.append(RU(position=pos, antenna=ant))
            
            iterations.append({
                'iteration': iter_num,
                'particles': particles,
                'fitness': float(iter_group['gbest_fitness_score'].iloc[0])
            })
        
        return iterations
    
    except Exception as e:
        print(f"Error loading PSO iterations: {e}")
        return None


def interpolate_iterations(iterations, num_frames=10):
    """
    Interpolate between PSO iterations to create smoother animation.
    
    Args:
        iterations (list): List of iteration data.
        num_frames (int): Number of frames to interpolate between each iteration.
        
    Returns:
        list: List of interpolated frames.
    """
    if not iterations or len(iterations) < 2:
        return iterations
    
    interpolated_frames = []
    
    for i in range(len(iterations) - 1):
        start_iter = iterations[i]
        end_iter = iterations[i + 1]
        
        start_particles = start_iter['particles']
        end_particles = end_iter['particles']
        
        # Ensure we have the same number of particles in both iterations
        min_particles = min(len(start_particles), len(end_particles))
        
        for frame in range(num_frames):
            t = frame / num_frames  # Interpolation factor (0 to 1)
            frame_particles = []
            
            for p in range(min_particles):
                start_pos = start_particles[p].position
                end_pos = end_particles[p].position
                
                # Linear interpolation of position
                interp_pos = start_pos + t * (end_pos - start_pos)
                
                # Create interpolated RU
                ant = Antenna(freq=start_particles[p].antenna.freq, 
                              elements=start_particles[p].antenna.elements)
                frame_particles.append(RU(position=interp_pos, antenna=ant))
            
            # Interpolate fitness
            interp_fitness = start_iter['fitness'] + t * (end_iter['fitness'] - start_iter['fitness'])
            
            interpolated_frames.append({
                'iteration': start_iter['iteration'] + t,
                'particles': frame_particles,
                'fitness': interp_fitness
            })
    
    # Add the last iteration
    interpolated_frames.append(iterations[-1])
    
    return interpolated_frames


def animate_pso(plotter, iterations, ues, area_bounds, loop=True, frame_rate=10):
    """
    Animate the PSO optimization process.
    
    Args:
        plotter (pv.Plotter): PyVista plotter object.
        iterations (list): List of iteration data.
        ues (list): List of UE objects.
        area_bounds (tuple): Bounds of the simulation area (min_x, max_x, min_y, max_y).
        loop (bool): Whether to loop the animation.
        frame_rate (int): Number of frames per second.
    """
    print("Setting up PSO animation...")
    if not iterations:
        print("No iterations to animate")
        return
    
    min_x, max_x, min_y, max_y = area_bounds
    
    # Interpolate iterations for smoother animation
    frames = interpolate_iterations(iterations)
    
    # Create text actor for iteration and fitness display
    # Using add_text with a position tuple instead of string to avoid CornerAnnotation issues
    text = f"Iteration: 0, Fitness: 0"
    text_actor = plotter.add_text(text, position=(10, 10), font_size=14, color='black')
    
    # Add UEs to the scene (static) - make them MUCH more visible
    if ues:
        print(f"Adding {len(ues)} UEs to the scene")
        ue_positions = np.array([ue.position for ue in ues])
        
        # Method 1: Add as points (very visible)
        plotter.add_points(
            ue_positions,
            color='dodgerblue',  # Brighter blue
            render_points_as_spheres=True,
            point_size=15,  # Much larger
            label='UEs (Points)'
        )
        
        # Method 2: Add as spheres (more 3D)
        ue_points = pv.PolyData(ue_positions)
        ue_spheres = ue_points.glyph(scale=False, geom=pv.Sphere(radius=8))  # Much larger radius
        plotter.add_mesh(ue_spheres, color='deepskyblue', opacity=0.7, label='UEs (Spheres)')
    
    # Add area bounds
    bounds_points = np.array([
        [min_x, min_y, 0],
        [max_x, min_y, 0],
        [max_x, max_y, 0],
        [min_x, max_y, 0],
        [min_x, min_y, 0]
    ])
    bounds_lines = pv.lines_from_points(bounds_points)
    plotter.add_mesh(bounds_lines, color='black', line_width=2)
    
    # Create RU visualization with a simpler approach
    # Use different colors for each RU to make them more distinguishable
    ru_colors = ['crimson', 'limegreen', 'royalblue', 'gold', 'magenta', 'darkorange', 'purple']
    
    # Create initial RU meshes - we'll show/hide these during animation
    ru_meshes = []
    
    # Get the first frame's RUs to create initial meshes
    best_rus = frames[0]['particles']
    print(f"Creating {len(best_rus)} RU actors")
    
    for i, ru in enumerate(best_rus):
        color = ru_colors[i % len(ru_colors)]
        print(f"Creating RU {i} at position {ru.position}")
        
        # Create a cone for the RU with the correct position - MUCH larger and more visible
        cone = pv.Cone(center=(ru.position[0], ru.position[1], ru.position[2]), 
                      direction=(0, 0, -1), height=60, radius=30)
        
        # Add the mesh with high specular to make it stand out
        mesh = plotter.add_mesh(cone, color=color, name=f'ru_{i}', specular=1.0, specular_power=15)
        
        # Also add a sphere at the same position for extra visibility
        sphere = pv.Sphere(center=(ru.position[0], ru.position[1], ru.position[2]), radius=15)
        sphere_mesh = plotter.add_mesh(sphere, color=color, opacity=0.7, name=f'ru_sphere_{i}')
        
        # Store both meshes for this RU
        ru_meshes.append((mesh, sphere_mesh))
        
        # Add a text label for the RU
        plotter.add_point_labels(
            [ru.position], 
            [f"RU {i+1}"], 
            font_size=14, 
            point_color=color, 
            point_size=20, 
            render_points_as_spheres=True,
            always_visible=True
        )
    
    def update_frame(frame_idx):
        """Update function for animation."""
        frame_idx = frame_idx % len(frames)
        frame = frames[frame_idx]
        
        # Update text - recreate the text actor each time
        plotter.remove_actor(text_actor)
        new_text = f"Iteration: {frame['iteration']:.1f}, Fitness: {frame['fitness']:.2f}"
        new_text_actor = plotter.add_text(new_text, position=(10, 10), font_size=14, color='black')
        
        # Update RU positions by directly modifying the meshes
        for i, ru in enumerate(frame['particles']):
            if i < len(ru_meshes):
                # Get the cone and sphere meshes
                cone_mesh, sphere_mesh = ru_meshes[i]
                
                # Remove the old meshes
                plotter.remove_actor(cone_mesh)
                plotter.remove_actor(sphere_mesh)
                
                # Create new meshes at the updated position
                color = ru_colors[i % len(ru_colors)]
                
                # Create a new cone at the updated position
                cone = pv.Cone(center=(ru.position[0], ru.position[1], ru.position[2]), 
                              direction=(0, 0, -1), height=60, radius=30)
                new_cone_mesh = plotter.add_mesh(cone, color=color, name=f'ru_{i}', 
                                               specular=1.0, specular_power=15)
                
                # Create a new sphere at the updated position
                sphere = pv.Sphere(center=(ru.position[0], ru.position[1], ru.position[2]), radius=15)
                new_sphere_mesh = plotter.add_mesh(sphere, color=color, opacity=0.7, name=f'ru_sphere_{i}')
                
                # Update our reference
                ru_meshes[i] = (new_cone_mesh, new_sphere_mesh)
                
                print(f"Updated RU {i} to position {ru.position}")
        
        # Force render update
        plotter.render()
        
        return
    
    # Set up animation using a callback approach for true animation
    try:
        print("Starting animation sequence...")
        
        # Initialize with first frame
        update_frame(0)
        
        # Use a simple approach for animation that works reliably with PyVista
        print("Starting interactive animation...")
        
        # Set up the animation loop
        print("Starting animation loop...")
        
        # Set initial frame
        current_frame = [0]
        
        # Use PyVista's built-in key press callback for animation control
        # This is compatible with PyVista's interactive rendering model
        def auto_advance():
            # Function to advance one frame when called
            current_frame[0] = (current_frame[0] + 1) % len(frames)
            update_frame(current_frame[0])
            if current_frame[0] % 5 == 0:
                print(f"Frame: {current_frame[0]+1}/{len(frames)}")
            
            # Schedule the next frame if animation is running
            if animation_running[0]:
                # Use a callback that PyVista recognizes
                plotter.iren.create_timer(500, auto_advance_callback)
        
        def auto_advance_callback():
            # This is a callback that PyVista can call
            auto_advance()
            return
        
        # Create key press callbacks for manual control
        def toggle_animation():
            nonlocal animation_running
            animation_running[0] = not animation_running[0]
            if animation_running[0]:
                print("Animation resumed")
                # Start the animation
                auto_advance()
            else:
                print("Animation paused")
                # Animation will stop because animation_running[0] is False
        
        def next_frame():
            # Pause animation and advance one frame
            if animation_running[0]:
                toggle_animation()
            current_frame[0] = (current_frame[0] + 1) % len(frames)
            update_frame(current_frame[0])
            print(f"Frame: {current_frame[0]+1}/{len(frames)}")
        
        def prev_frame():
            # Pause animation and go back one frame
            if animation_running[0]:
                toggle_animation()
            current_frame[0] = (current_frame[0] - 1) % len(frames)
            update_frame(current_frame[0])
            print(f"Frame: {current_frame[0]+1}/{len(frames)}")
        
        # Register key press callbacks
        plotter.add_key_event('space', toggle_animation)
        plotter.add_key_event('n', next_frame)
        plotter.add_key_event('p', prev_frame)
        
        # Add instructions text
        plotter.add_text("Space: Play/Pause, 'n': Next frame, 'p': Previous frame", position='upper_left', font_size=12, color='black')
        
        # Set up for animation
        animation_running = [True]
        print("Starting automatic animation. Press Space to pause/resume.")
        
        # Set initial camera position to see the whole scene
        # Use a better camera position that shows depth
        min_x, max_x, min_y, max_y = area_bounds
        center = [(min_x + max_x) / 2, (min_y + max_y) / 2, 0]
        position = [(min_x + max_x) / 2, (min_y + max_y) / 2, max(max_x - min_x, max_y - min_y) * 1.2]
        plotter.camera_position = [position, center, [0, 0, 1]]
        
        # Start with first frame
        update_frame(0)
        
        # Start the animation automatically
        print("Camera positioned to show the entire scene")
        print("Animation starting automatically...")
        
        # Start the animation loop using PyVista's timer
        plotter.iren.initialize()
        auto_advance()
        
        # Start with first frame
        update_frame(0)
        
        # Set initial camera position to see the whole scene
        # Use a better camera position that shows depth
        min_x, max_x, min_y, max_y = area_bounds
        center = [(min_x + max_x) / 2, (min_y + max_y) / 2, 0]
        position = [(min_x + max_x) / 2, (min_y + max_y) / 2, max(max_x - min_x, max_y - min_y) * 1.2]
        plotter.camera_position = [position, center, [0, 0, 1]]
        
        print("Camera positioned to show the entire scene")
        
        # Show the interactive visualization with animation
        print("Animation running. Close the window to exit.")
        plotter.show()
                
    except Exception as e:
        print(f"Error during animation: {e}")
        import traceback
        traceback.print_exc()
