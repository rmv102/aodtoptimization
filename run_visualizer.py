#!/usr/bin/env python3
"""
AODT Standalone Visualization Script

This script provides a standalone visualization tool that mimics NVIDIA AODT's
visualization behavior. It can be run directly to visualize simulation results.

Usage:
    python run_visualizer.py [options]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Starting AODT Visualizer...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Add the project root to the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from aodt_pso.visualization.plot_3d import plot_scene
from aodt_pso.visualization.animation import load_pso_iterations, animate_pso
from aodt_pso.aerial.dt import RU, UE
from aodt_pso.aerial.phy import Antenna
from aodt_pso.simulation.signal_map import generate_ue_distribution
import pyvista as pv


def load_optimization_results(csv_path):
    """
    Load optimization results from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing optimization results.
        
    Returns:
        tuple: (rus, best_fitness) where rus is a list of RU objects and
               best_fitness is the best fitness score from the optimization.
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Get the last iteration
        max_iter = df['iteration'].max()
        last_iter_data = df[df['iteration'] == max_iter]
        
        # Extract the best fitness score
        best_fitness = float(last_iter_data['gbest_fitness_score'].iloc[0])
        
        # Create RU objects
        rus = []
        for _, row in last_iter_data.iterrows():
            pos = np.array([float(row['pos_x']), float(row['pos_y']), float(row['pos_z'])])
            freq = float(row['frequency_ghz'])
            elements = int(row['elements'])
            ant = Antenna(freq=freq, elements=elements)
            rus.append(RU(position=pos, antenna=ant))
            
        print(f"Loaded optimization results from {csv_path}")
        print(f"Best fitness: {best_fitness}")
        print(f"Number of RUs: {len(rus)}")
        
        return rus, best_fitness
        
    except Exception as e:
        print(f"Error loading optimization results: {e}")
        return None, None


def generate_visualization(csv_path=None, num_ues=200, area_bounds=(0, 1000, 0, 1000), ue_height=1.5, animate=False):
    """
    Generate visualization of the AODT simulation results.
    
    Args:
        csv_path (str, optional): Path to the CSV file containing optimization results.
            If None, will look for the default file location.
        num_ues (int, optional): Number of UEs to generate if not loading from file.
        area_bounds (tuple, optional): Bounds of the simulation area (min_x, max_x, min_y, max_y).
        ue_height (float, optional): Height of the UEs.
    """
    print("\n--- AODT Visualization Starting ---")
    
    # Default CSV path if not provided
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aodt_pso', 'data', 'pso_optimization_log.csv')
        csv_path = os.path.abspath(csv_path)
        print(f"Looking for optimization results at: {csv_path}")
    
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
        if animate:
            # Load all PSO iterations for animation
            print("Loading PSO iterations for animation...")
            iterations = load_pso_iterations(csv_path)
            
            if iterations:
                print(f"Loaded {len(iterations)} iterations for animation")
                # Create plotter for animation with better settings
                plotter = pv.Plotter(window_size=[1024, 768])
                plotter.set_background('white', top='aliceblue')
                # Camera will be set by the animation function
                plotter.enable_trackball_style()
                plotter.add_axes(interactive=True)
                plotter.add_floor('-z', pad=0.1, color='lightgrey', opacity=0.5)
                
                # Add a title to the visualization
                plotter.add_text("AODT PSO Optimization Animation", position='upper_edge', font_size=16, color='black')
                
                # Add orientation axes widget (simpler approach)
                plotter.add_axes_at_origin()
                
                # Start animation
                print("Starting PSO animation...")
                animate_pso(plotter, iterations, ues, area_bounds, loop=True, frame_rate=5)
            else:
                print("No iterations found for animation, showing static visualization instead")
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
    parser.add_argument('--animate', action='store_true', help='Enable PSO animation')
    
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
