#!/usr/bin/env python3
"""
AODT Visualization Script

This script provides a standalone visualization tool that mimics NVIDIA AODT's
visualization behavior. It can be run directly to visualize simulation results.

Usage:
    python -m aodt_pso.visualization.visualizer
"""

import os
import sys
print("Starting AODT Visualizer...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Try importing from package structure, fall back to direct imports if run as script
try:
    from .plot_3d import plot_scene
    from ..aerial.dt import RU, UE
    from ..aerial.phy import Antenna
    from ..simulation.signal_map import generate_ue_distribution
    from ..optimizer.pso import PSO
except (ImportError, ValueError):
    # This allows the script to be run directly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  
    from aodt_pso.visualization.plot_3d import plot_scene
    from aodt_pso.aerial.dt import RU, UE
    from aodt_pso.aerial.phy import Antenna
    from aodt_pso.simulation.signal_map import generate_ue_distribution
    from aodt_pso.optimizer.pso import PSO


def load_optimization_results(csv_path):
    """
    Load optimization results from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing optimization results.
        
    Returns:
        tuple: (rus, best_fitness) where rus is a list of RU objects and
               best_fitness is the best fitness score from the optimization.
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded optimization results from {csv_path}")
        
        # Check if this is a PSO results file
        if 'iteration' not in df.columns:
            print("CSV file does not contain PSO iteration data")
            return None, None
        
        # Get the best fitness from the last iteration
        best_fitness = df['gbest_fitness_score'].iloc[-1]
        print(f"Best fitness score: {best_fitness}")
        
        # Extract the best RU configurations from the last iteration
        last_iter = df[df['iteration'] == df['iteration'].max()]
        rus = []
        
        for _, row in last_iter.iterrows():
            pos = np.array([float(row['pos_x']), float(row['pos_y']), float(row['pos_z'])])
            freq = float(row['frequency_ghz'])
            elements = int(row['elements'])
            ant = Antenna(freq=freq, elements=elements)
            rus.append(RU(position=pos, antenna=ant))
        
        print(f"Loaded {len(rus)} RUs from optimization results")
        return rus, best_fitness
        
    except Exception as e:
        print(f"Error loading optimization results: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def load_pso_iterations(csv_path):
    """
    Load all PSO iterations from a CSV file for animation.
    
    Args:
        csv_path (str): Path to the CSV file containing optimization results.
        
    Returns:
        list: List of iteration data, where each item contains the particles for that iteration.
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded PSO iterations from {csv_path}")
        
        # Check if this is a PSO results file
        if 'iteration' not in df.columns:
            print("CSV file does not contain PSO iteration data")
            return None
        
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
        
        print(f"Loaded {len(iterations)} iterations for animation")
        
        # Interpolate iterations for smoother animation
        if len(iterations) > 1:
            print("Interpolating iterations for smoother animation...")
            interpolated = interpolate_iterations(iterations)
            print(f"Created {len(interpolated)} frames after interpolation")
            return interpolated
        else:
            return iterations
        
    except Exception as e:
        print(f"Error loading PSO iterations: {e}")
        import traceback
        traceback.print_exc()
        return None


def interpolate_iterations(iterations, frames_between=5):
    """
    Interpolate between iterations to create smoother animations.
    
    Args:
        iterations (list): List of iteration data.
        frames_between (int): Number of frames to insert between each iteration.
        
    Returns:
        list: List of interpolated frames.
    """
    if len(iterations) < 2:
        return iterations
    
    interpolated_frames = []
    
    for i in range(len(iterations) - 1):
        start_iter = iterations[i]
        end_iter = iterations[i + 1]
        
        # Add the current iteration
        interpolated_frames.append(start_iter)
        
        # Skip interpolation if iterations are consecutive
        if end_iter['iteration'] - start_iter['iteration'] <= 1:
            continue
        
        # Interpolate between iterations
        for t in range(1, frames_between + 1):
            t_norm = t / (frames_between + 1)  # Normalize to [0, 1]
            
            # Interpolate particles
            frame_particles = []
            for j in range(len(start_iter['particles'])):
                start_ru = start_iter['particles'][j]
                end_ru = end_iter['particles'][j]
                
                # Interpolate position
                pos = start_ru.position + t_norm * (end_ru.position - start_ru.position)
                
                # Interpolate antenna parameters
                freq = start_ru.antenna.freq + t_norm * (end_ru.antenna.freq - start_ru.antenna.freq)
                elements = int(round(start_ru.antenna.elements + t_norm * (end_ru.antenna.elements - start_ru.antenna.elements)))
                
                # Create interpolated RU
                ant = Antenna(freq=freq, elements=elements)
                frame_particles.append(RU(position=pos, antenna=ant))
            
            # Interpolate fitness
            interp_fitness = start_iter['fitness'] + t_norm * (end_iter['fitness'] - start_iter['fitness'])
            
            # Add interpolated frame
            interpolated_frames.append({
                'iteration': start_iter['iteration'] + t_norm * (end_iter['iteration'] - start_iter['iteration']),
                'particles': frame_particles,
                'fitness': interp_fitness
            })
    
    # Add the last iteration
    interpolated_frames.append(iterations[-1])
    
    return interpolated_frames


def generate_visualization(csv_path=None, num_ues=200, area_bounds=(0, 1000, 0, 1000), ue_height=1.5, animate=False):
    """
    Generate a 3D visualization of the AODT simulation.
    
    Args:
        csv_path (str): Path to the CSV file with optimization results.
        num_ues (int): Number of UEs to generate.
        area_bounds (tuple): Simulation area bounds (min_x, max_x, min_y, max_y).
        ue_height (float): Height of the UEs in meters.
        animate (bool): Whether to animate the PSO optimization process.
    """
    print(f"\n--- AODT Visualization Parameters ---")
    print(f"CSV Path: {csv_path or 'None (will use example RUs)'}") 
    print(f"Number of UEs: {num_ues}")
    print(f"Area Bounds: {area_bounds}")
    print(f"UE Height: {ue_height} m")
    print(f"Animation: {'Enabled' if animate else 'Disabled'}")
    
    # Load RUs from optimization results
    rus, best_fitness = load_optimization_results(csv_path)
    
    if rus is None:
        print("No optimization results found. Generating example RUs...")
        # Create example RUs if no results are loaded
        ru1_pos = np.array([area_bounds[0] + 100, area_bounds[2] + 150, 15.0])
        ru1_ant = Antenna(freq=3.5, elements=4)
        ru1 = RU(position=ru1_pos, antenna=ru1_ant)
        
        ru2_pos = np.array([area_bounds[1] - 100, area_bounds[3] - 150, 15.0])
        ru2_ant = Antenna(freq=5.2, elements=8)
        ru2 = RU(position=ru2_pos, antenna=ru2_ant)
        
        rus = [ru1, ru2]
    
    # Generate UEs
    print(f"Generating {num_ues} UEs...")
    ues = generate_ue_distribution(num_ues=num_ues, area_bounds=area_bounds, ue_height=ue_height)
    
    # Visualize the scene
    print("\n--- Visualizing AODT Simulation ---")
    print("Opening PyVista window for 3D visualization...")
    
    try:
        if animate and csv_path:
            # Load all iterations for animation
            iterations = load_pso_iterations(csv_path)
            if iterations:
                print(f"Starting animation with {len(iterations)} frames...")
                plot_scene(rus=rus, ues=ues, area_bounds=area_bounds, animate=True, iterations=iterations)
                print("Animation completed successfully.")
            else:
                print("No iterations found for animation. Falling back to static visualization.")
                plot_scene(rus=rus, ues=ues, area_bounds=area_bounds)
        else:
            # Static visualization
            plot_scene(rus=rus, ues=ues, area_bounds=area_bounds)
            print("Visualization completed successfully.")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main entry point for the visualization script.
    """
    parser = argparse.ArgumentParser(description="AODT Visualization Tool")
    parser.add_argument('--csv', type=str, help='Path to the CSV file with optimization results')
    parser.add_argument('--ues', type=int, default=200, help='Number of UEs to generate')
    parser.add_argument('--area', type=float, nargs=4, default=[0, 1000, 0, 1000], 
                        help='Area bounds: min_x max_x min_y max_y')
    parser.add_argument('--ue-height', type=float, default=1.5, help='Height of the UEs')
    parser.add_argument('--animate', action='store_true', help='Animate the PSO optimization process')
    
    args = parser.parse_args()
    
    generate_visualization(
        csv_path=args.csv,
        num_ues=args.ues,
        area_bounds=tuple(args.area),
        ue_height=args.ue_height,
        animate=args.animate
    )


if __name__ == "__main__":
    main()
