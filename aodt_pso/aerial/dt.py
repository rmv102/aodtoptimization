"""
Defines the digital twin components for the AODT simulation.

- RU (Radio Unit): Represents a base station with a position and an antenna.
- UE (User Equipment): Represents a user with a position.
- Scenario: Manages a collection of RUs and UEs and runs the simulation.
"""

from typing import List
import numpy as np

from .phy import Antenna, compute_cfr

class RU:
    """Represents a Radio Unit (base station)."""
    def __init__(self, position: np.ndarray, antenna: Antenna):
        """
        Initializes the RU.

        Args:
            position (np.ndarray): The 3D position [x, y, z] of the RU.
            antenna (Antenna): The Antenna object associated with this RU.
        """
        self.position = position
        self.antenna = antenna

class UE:
    """Represents a User Equipment (user device)."""
    def __init__(self, position: np.ndarray):
        """
        Initializes the UE.

        Args:
            position (np.ndarray): The 3D position [x, y, z] of the UE.
        """
        self.position = position

class Scenario:
    """Manages a simulation scenario with a set of RUs and UEs."""
    def __init__(self, rus: List[RU], ues: List[UE]):
        """
        Initializes the Scenario.

        Args:
            rus (List[RU]): A list of Radio Unit objects.
            ues (List[UE]): A list of User Equipment objects.
        """
        self.rus = rus
        self.ues = ues
        self.channel_matrix = None

    def run(self) -> np.ndarray:
        """
        Runs the simulation to compute the channel matrix between all RUs and UEs.

        The channel matrix stores the computed CFR (channel gain) for each RU-UE pair.
        The matrix shape will be (num_ues, num_rus).

        Returns:
            np.ndarray: A 2D numpy array containing the channel gains.
        """
        if not self.rus or not self.ues:
            print("Warning: RUs or UEs list is empty. Returning an empty matrix.")
            return np.array([[]])

        num_rus = len(self.rus)
        num_ues = len(self.ues)
        self.channel_matrix = np.zeros((num_ues, num_rus))

        for i, ue in enumerate(self.ues):
            for j, ru in enumerate(self.rus):
                # Compute the channel gain for the current UE-RU pair
                cfr = compute_cfr(ru.position, ue.position, ru.antenna)
                self.channel_matrix[i, j] = cfr
        
        return self.channel_matrix

if __name__ == '__main__':
    # Example usage for testing
    
    # Create some antennas
    ant1 = Antenna(freq=2.4, elements=2)
    ant2 = Antenna(freq=5.8, elements=8)
    
    # Create RUs (base stations)
    ru1 = RU(position=np.array([0, 0, 10]), antenna=ant1)
    ru2 = RU(position=np.array([200, 150, 10]), antenna=ant2)
    rus = [ru1, ru2]
    
    # Create UEs (users)
    ue1 = UE(position=np.array([50, 50, 1.5]))
    ue2 = UE(position=np.array([180, 130, 1.5]))
    ue3 = UE(position=np.array([100, -50, 1.5]))
    ues = [ue1, ue2, ue3]
    
    # Create and run the scenario
    scenario = Scenario(rus=rus, ues=ues)
    channel_gains = scenario.run()
    
    print("--- Scenario Test ---")
    print(f"Number of RUs: {len(rus)}")
    print(f"Number of UEs: {len(ues)}")
    print("\nChannel Gain Matrix (UEs x RUs):")
    print(channel_gains)
    
    # You can interpret the matrix:
    # channel_gains[0, 1] is the gain from RU 2 to UE 1.
    print(f"\nExample: Gain from RU 2 to UE 1 is {channel_gains[0, 1]:.4e}")
    print(f"Example: Gain from RU 1 to UE 3 is {channel_gains[2, 0]:.4e}") 