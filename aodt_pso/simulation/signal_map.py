"""
Generates a 2D distribution of User Equipments (UEs) for the simulation.
"""
from typing import List
import numpy as np

from ..aerial.dt import UE

def generate_ue_distribution(
    num_ues: int, 
    area_bounds: tuple, 
    ue_height: float = 1.5
) -> List[UE]:
    """
    Generates a list of UE objects with random positions within a defined area.

    The UEs are distributed uniformly within the x and y boundaries of the
    simulation area. All UEs are placed at the same height.

    Args:
        num_ues (int): The number of UEs to generate.
        area_bounds (tuple): A tuple containing the min and max coordinates 
                             for x and y, e.g., (min_x, max_x, min_y, max_y).
        ue_height (float): The height (z-coordinate) for all UEs. Defaults to 1.5 meters.

    Returns:
        List[UE]: A list of initialized UE objects.
    """
    min_x, max_x, min_y, max_y = area_bounds
    
    # Generate random x and y coordinates for each UE
    x_coords = np.random.uniform(min_x, max_x, num_ues)
    y_coords = np.random.uniform(min_y, max_y, num_ues)
    
    # Create a list of UE objects
    ues = []
    for i in range(num_ues):
        position = np.array([x_coords[i], y_coords[i], ue_height])
        ues.append(UE(position=position))
        
    print(f"Generated {num_ues} UEs within area [({min_x}, {min_y}) to ({max_x}, {max_y})].")
    return ues

if __name__ == '__main__':
    # Example usage for testing
    
    # Define simulation parameters
    NUM_UES_TO_GENERATE = 100
    AREA_BOUNDS = (0, 1000, 0, 1000)  # 1km x 1km area
    
    # Generate the UEs
    user_equipments = generate_ue_distribution(
        num_ues=NUM_UES_TO_GENERATE,
        area_bounds=AREA_BOUNDS
    )
    
    # Print details of the first 5 UEs to verify
    print(f"\n--- UE Generation Test ---")
    print(f"Total UEs generated: {len(user_equipments)}")
    print("First 5 UE positions (x, y, z):")
    for i in range(min(5, len(user_equipments))):
        print(f"  UE {i+1}: {user_equipments[i].position}")

    # A simple check to ensure positions are within bounds
    first_ue_pos = user_equipments[0].position
    assert AREA_BOUNDS[0] <= first_ue_pos[0] <= AREA_BOUNDS[1]
    assert AREA_BOUNDS[2] <= first_ue_pos[1] <= AREA_BOUNDS[3]
    print("\nAssertion passed: First UE is within the defined area bounds.") 