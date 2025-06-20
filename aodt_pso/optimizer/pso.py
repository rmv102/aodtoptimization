"""
Implements an advanced Particle Swarm Optimization (PSO) algorithm.

This module provides a more sophisticated PSO implementation with features like:
- G-best (global) and L-best (local ring) topologies.
- Stagnation detection and swarm diversification.
- Dynamic and adaptive inertia weight calculation.
- Vectorized fitness evaluation for efficiency.
- Comprehensive logging of optimization metrics.
"""
import numpy as np
from typing import Callable, List, Dict, Any
import os
import csv

class PSO:
    """
    An advanced class for performing Particle Swarm Optimization.
    """
    def __init__(self,
                 objective_func: Callable,
                 num_dimensions: int,
                 bounds: List[tuple],
                 num_particles: int,
                 max_iter: int,
                 topology: str = 'gbest',
                 w_range: tuple = (0.9, 0.4),
                 c1: float = 2.05,
                 c2: float = 2.05,
                 stagnation_patience: int = 15,
                 k: int = 3,
                 log_to_csv: str = None):
        """
        Initializes the PSO optimizer.

        Args:
            objective_func (Callable): A vectorized function to be minimized. It must
                accept a 2D numpy array of shape (num_particles, num_dimensions) and
                return a 1D array of fitness values of shape (num_particles,).
            num_dimensions (int): The number of parameters to optimize.
            bounds (List[tuple]): A list of (min, max) tuples for each dimension.
            num_particles (int): The number of particles in the swarm.
            max_iter (int): The maximum number of iterations.
            topology (str): The swarm topology ('gbest' or 'lbest').
            w_range (tuple): The (start, end) range for inertia weight decay.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
            stagnation_patience (int): Iterations to wait before diversification.
            k (int): The number of neighbors for the l-best topology (must be odd).
            log_to_csv (str, optional): Path to a CSV file to log results. Defaults to None.
        """
        if topology not in ['gbest', 'lbest']:
            raise ValueError("Topology must be either 'gbest' or 'lbest'.")
        if topology == 'lbest' and k % 2 == 0:
            raise ValueError("k must be an odd integer for l-best topology.")

        self.objective_func = objective_func
        self.num_dimensions = num_dimensions
        self.bounds = np.array(bounds).T
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.topology = topology
        self.w_range = w_range
        self.c1 = c1
        self.c2 = c2
        self.stagnation_patience = stagnation_patience
        self.k = k
        self.csv_log_path = log_to_csv
        
        self.params_per_ru = 4
        if self.num_dimensions % self.params_per_ru != 0:
            raise ValueError("num_dimensions must be a multiple of parameters per RU (4).")
        self.num_rus = self.num_dimensions // self.params_per_ru

        self._initialize_swarm()

        # History and logging
        self.log: Dict[str, List[Any]] = {
            "gbest_fitness": [],
            "swarm_diversity": [],
            "avg_fitness": [],
            "stagnation_resets": 0
        }

        # If logging to CSV, ensure the header is present
        if self.csv_log_path and not os.path.exists(self.csv_log_path):
             with open(self.csv_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'ru_id', 'pos_x', 'pos_y', 'frequency_ghz', 'elements', 'gbest_fitness_score'])

    def _initialize_swarm(self):
        """Initializes the swarm's positions, velocities, and bests."""
        # Positions
        self.swarm = self.bounds[0] + np.random.rand(self.num_particles, self.num_dimensions) * (self.bounds[1] - self.bounds[0])
        
        # Velocities
        v_range = self.bounds[1] - self.bounds[0]
        self.velocity = -v_range + 2 * v_range * np.random.rand(self.num_particles, self.num_dimensions)
        
        # Personal and global bests
        self.pbest_pos = self.swarm.copy()
        self.pbest_fitness = self.objective_func(self.pbest_pos)
        
        self.gbest_idx = np.argmin(self.pbest_fitness)
        self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
        self.gbest_fitness = self.pbest_fitness[self.gbest_idx]
        
        # Stagnation tracking
        self.stagnation_counter = 0
        self.last_gbest_fitness = self.gbest_fitness

    def _get_social_best(self):
        """Determines the social best position for each particle based on topology."""
        if self.topology == 'gbest':
            return self.gbest_pos
        
        # L-best (ring topology)
        social_best = np.zeros_like(self.pbest_pos)
        half_k = self.k // 2
        
        for i in range(self.num_particles):
            # Define neighbors with wrap-around
            indices = [(i + j) % self.num_particles for j in range(-half_k, half_k + 1)]
            
            # Find the best in the neighborhood
            best_neighbor_idx = indices[np.argmin(self.pbest_fitness[indices])]
            social_best[i] = self.pbest_pos[best_neighbor_idx]
            
        return social_best

    def _log_to_csv(self, iteration: int):
        """Appends the current global best solution to the CSV log file."""
        if not self.csv_log_path:
            return
            
        solution_reshaped = self.gbest_pos.reshape((self.num_rus, self.params_per_ru))
        
        try:
            with open(self.csv_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for ru_idx, params in enumerate(solution_reshaped):
                    row = [
                        iteration,
                        ru_idx + 1,
                        f"{params[0]:.2f}",
                        f"{params[1]:.2f}",
                        f"{params[2]:.2f}",
                        int(round(params[3])),
                        f"{self.gbest_fitness:.4f}"
                    ]
                    writer.writerow(row)
        except IOError as e:
            print(f"Warning: Could not write to CSV log file {self.csv_log_path}. Error: {e}")

    def _update_velocity_and_position(self, iteration: int):
        """Updates particle velocities and positions."""
        r1 = np.random.rand(self.num_particles, self.num_dimensions)
        r2 = np.random.rand(self.num_particles, self.num_dimensions)
        
        social_best = self._get_social_best()
        
        cognitive_velocity = self.c1 * r1 * (self.pbest_pos - self.swarm)
        social_velocity = self.c2 * r2 * (social_best - self.swarm)
        
        # Dynamic inertia decay
        w = self.w_range[0] - (self.w_range[0] - self.w_range[1]) * (iteration / self.max_iter)
        
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity
        self.swarm += self.velocity
        
        # Boundary clipping
        self.swarm = np.clip(self.swarm, self.bounds[0], self.bounds[1])

    def _diversify_if_stagnated(self):
        """Checks for stagnation and re-initializes part of the swarm if needed."""
        # Fitness-based stagnation check
        if np.isclose(self.gbest_fitness, self.last_gbest_fitness, rtol=1e-6):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_gbest_fitness = self.gbest_fitness

        if self.stagnation_counter >= self.stagnation_patience:
            print(f"\nStagnation detected at iteration. Diversifying swarm...")
            self.stagnation_counter = 0
            self.log["stagnation_resets"] += 1

            # Re-initialize the worst half of the swarm, keeping the global best
            fitness_ranks = np.argsort(self.pbest_fitness)
            worst_indices = fitness_ranks[self.num_particles // 2:]
            
            # Ensure gbest is not reset
            worst_indices = np.setdiff1d(worst_indices, [self.gbest_idx])

            # Generate new positions and reset velocities for these particles
            num_to_reset = len(worst_indices)
            self.swarm[worst_indices] = self.bounds[0] + np.random.rand(num_to_reset, self.num_dimensions) * (self.bounds[1] - self.bounds[0])
            self.velocity[worst_indices] = 0

    def _log_metrics(self):
        """Logs various metrics for analysis."""
        self.log["gbest_fitness"].append(self.gbest_fitness)
        self.log["avg_fitness"].append(np.mean(self.pbest_fitness))

        # Swarm diversity (average distance from swarm centroid)
        centroid = np.mean(self.swarm, axis=0)
        diversity = np.mean(np.linalg.norm(self.swarm - centroid, axis=1))
        self.log["swarm_diversity"].append(diversity)

    def run(self):
        """Runs the PSO algorithm."""
        print(f"--- Starting PSO ({self.topology.upper()}) Optimization ---")
        for i in range(self.max_iter):
            # Evaluate fitness for all particles in a single batch
            current_fitness = self.objective_func(self.swarm)
            
            # Update personal best
            update_mask = current_fitness < self.pbest_fitness
            self.pbest_pos[update_mask] = self.swarm[update_mask]
            self.pbest_fitness[update_mask] = current_fitness[update_mask]
            
            # Update global best
            current_gbest_idx = np.argmin(self.pbest_fitness)
            if self.pbest_fitness[current_gbest_idx] < self.gbest_fitness:
                self.gbest_idx = current_gbest_idx
                self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
                self.gbest_fitness = self.pbest_fitness[self.gbest_idx]

            # Log results to CSV
            self._log_to_csv(i + 1)

            # Update velocities and positions
            self._update_velocity_and_position(i)
            
            # Check for stagnation and diversify if needed
            self._diversify_if_stagnated()

            # Log metrics
            self._log_metrics()
            
            if (i + 1) % 10 == 0:
                print(f"Iter {i+1}/{self.max_iter} | G-Best Fitness: {self.gbest_fitness:.4f} | Avg Fitness: {self.log['avg_fitness'][-1]:.4f} | Diversity: {self.log['swarm_diversity'][-1]:.2f}")
        
        print("--- PSO Optimization Finished ---")
        return self.gbest_pos, self.gbest_fitness, self.log

if __name__ == '__main__':
    # Example usage with a simple mathematical function (Sphere function)
    
    def sphere_function(x):
        """A simple quadratic function to test the optimizer."""
        return np.sum(x**2)

    # Define optimizer parameters
    DIMS = 5
    BOUNDS = [(-10, 10)] * DIMS
    PARTICLES = 50
    MAX_ITER = 100

    # Create and run the optimizer
    pso = PSO(
        objective_func=sphere_function,
        num_dimensions=DIMS,
        bounds=BOUNDS,
        num_particles=PARTICLES,
        max_iter=MAX_ITER
    )
    best_position, best_fitness, fitness_history = pso.run()

    print("\n--- Test Results ---")
    print(f"Optimal Position: {best_position}")
    print(f"Optimal Fitness: {best_fitness}")
    
    # The optimal solution for the sphere function is at [0, 0, ..., 0] with fitness 0
    assert best_fitness < 1e-4, "Test failed: Optimizer did not converge to the expected minimum."
    print("\nAssertion passed: Optimizer successfully found the minimum of the test function.")
    
    # You can plot the convergence
    import matplotlib.pyplot as plt
    plt.plot(fitness_history["gbest_fitness"])
    plt.title("Convergence of PSO on Sphere Function")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.show() 