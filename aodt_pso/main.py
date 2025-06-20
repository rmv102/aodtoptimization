"""
Main entry point for the AODT-PSO simulation.

This script orchestrates the entire optimization pipeline:
1. Generates UEs.
2. Defines a vectorized objective function.
3. Initializes and runs the advanced PSO optimizer.
4. Prints results and visualizes the final deployment.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure modules are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aerial.phy import Antenna
from aerial.dt import RU, UE, Scenario
from optimizer.pso import PSO
from simulation.signal_map import generate_ue_distribution
from visualization.plot_3d import plot_scene

# --- Simulation Parameters ---
AREA_BOUNDS = (0, 1000, 0, 1000)
RU_HEIGHT = 15.0
UE_HEIGHT = 1.5
NUM_RUS = 3
NUM_UES = 200

# --- PSO Parameters ---
PSO_TOPOLOGY = 'gbest'  # 'gbest' or 'lbest'
NUM_PARTICLES = 50
MAX_ITER = 100
OUTPUT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pso_optimization_log.csv')

def parse_pso_solution(solution: np.ndarray, num_rus: int):
    """Parses a flat solution vector from PSO into a list of RU objects."""
    params_per_ru = 4  # x, y, freq, elements
    rus = []
    solution_reshaped = solution.reshape((num_rus, params_per_ru))
    
    for params in solution_reshaped:
        pos = np.array([params[0], params[1], RU_HEIGHT])
        freq = params[2]
        elements = int(round(params[3]))
        elements = np.clip(elements, 1, 8)
        ant = Antenna(freq=freq, elements=elements)
        rus.append(RU(position=pos, antenna=ant))
        
    return rus

def objective_function_vectorized(particles: np.ndarray, ues: list, num_rus: int):
    """
    Vectorized objective function for PSO.
    
    Evaluates the fitness for an entire swarm of particles simultaneously.
    Fitness is the negative of the sum of log-signal quality. Lower is better.
    """
    num_particles = particles.shape[0]
    fitness_scores = np.zeros(num_particles)

    for i in range(num_particles):
        try:
            # Create a scenario for each particle
            rus = parse_pso_solution(particles[i], num_rus)
            scenario = Scenario(rus=rus, ues=ues)
            channel_matrix = scenario.run()
            
            # For each UE, find the signal from the best RU
            max_signal_per_ue = np.max(channel_matrix, axis=1)
            
            # Sum of log-signal quality. Epsilon avoids log(0).
            total_signal_quality = np.sum(np.log10(max_signal_per_ue + 1e-12))
            fitness_scores[i] = -total_signal_quality
        except (ValueError, IndexError):
            fitness_scores[i] = 1e9 # Assign a high penalty for invalid particles
    
    return fitness_scores

def plot_convergence(log: dict):
    """Plots multiple convergence metrics from the PSO log."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot fitness
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Fitness (Lower is Better)', color='tab:blue')
    ax1.plot(log['gbest_fitness'], color='tab:blue', label='Best Fitness')
    ax1.plot(log['avg_fitness'], color='tab:cyan', linestyle='--', label='Avg. Swarm Fitness')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Plot swarm diversity on a second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Swarm Diversity', color='tab:red')
    ax2.plot(log['swarm_diversity'], color='tab:red', linestyle=':', label='Swarm Diversity')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle('PSO Performance Analysis')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("pso_performance.png")
    plt.show()

if __name__ == "__main__":
    print("--- 1. Initializing Scenario ---")
    ues = generate_ue_distribution(
        num_ues=NUM_UES,
        area_bounds=AREA_BOUNDS,
        ue_height=UE_HEIGHT
    )

    print("\n--- 2. Setting up PSO Optimizer ---")
    bounds = []
    for _ in range(NUM_RUS):
        bounds.extend([
            (AREA_BOUNDS[0], AREA_BOUNDS[1]),
            (AREA_BOUNDS[2], AREA_BOUNDS[3]),
            (2.0, 6.0),
            (1, 8)
        ])
    num_dimensions = len(bounds)

    # The objective function needs to be a lambda to pass the extra `ues` and `NUM_RUS` arguments
    objective_func = lambda p: objective_function_vectorized(p, ues=ues, num_rus=NUM_RUS)

    pso = PSO(
        objective_func=objective_func,
        num_dimensions=num_dimensions,
        bounds=bounds,
        num_particles=NUM_PARTICLES,
        max_iter=MAX_ITER,
        topology=PSO_TOPOLOGY,
        log_to_csv=OUTPUT_CSV_PATH
    )
    
    best_solution, best_fitness, pso_log = pso.run()

    print("\n--- 3. Optimization Results ---")
    print(f"Best Fitness Score: {best_fitness:.4f}")
    if pso_log['stagnation_resets'] > 0:
        print(f"Swarm was diversified {pso_log['stagnation_resets']} time(s) due to stagnation.")
    
    optimized_rus = parse_pso_solution(best_solution, NUM_RUS)
    print("\nOptimized Base Station Configurations:")
    for i, ru in enumerate(optimized_rus):
        pos, ant = ru.position, ru.antenna
        print(f"  RU-{i+1}: Pos=({pos[0]:.2f}, {pos[1]:.2f}), Freq={ant.freq:.2f} GHz, Elem={ant.elements}")

    print("\n--- 4. Plotting Performance ---")
    plot_convergence(pso_log)

    print("\n--- 5. Visualizing 3D Scene ---")
    plot_scene(rus=optimized_rus, ues=ues, area_bounds=AREA_BOUNDS)

    print("\n--- Simulation Complete ---") 