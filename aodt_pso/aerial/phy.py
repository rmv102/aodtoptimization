"""
Defines the physical layer components for the AODT simulation.

- Antenna: Represents a base station antenna with specific physical properties.
- compute_cfr: Simulates channel frequency response between a base station and a user.
"""

import numpy as np

class Antenna:
    """Represents a 5G antenna with configurable properties."""
    def __init__(self, freq: float, elements: int, gain_dbi: float = 15.0):
        """
        Initializes the Antenna object.

        Args:
            freq (float): The operating frequency in GHz (e.g., 2.4, 3.5, 5.8).
            elements (int): The number of antenna elements.
            gain_dbi (float): The antenna gain in dBi (decibels relative to isotropic).
        """
        if not (2 <= freq <= 6):
            raise ValueError("Frequency must be between 2 and 6 GHz.")
        if not (1 <= elements <= 8):
            raise ValueError("Antenna elements must be between 1 and 8.")
        
        self.freq = freq
        self.elements = elements
        self.gain_dbi = gain_dbi

def compute_cfr(bs_pos: np.ndarray, ue_pos: np.ndarray, antenna: Antenna) -> float:
    """
    Computes a simplified Channel Frequency Response (CFR) as a measure of signal strength.

    This function simulates path loss based on the Free-Space Path Loss (FSPL) model,
    adjusted for frequency and distance. It does not simulate multipath fading for simplicity.
    The returned value is a linear magnitude, not in dB.

    Args:
        bs_pos (np.ndarray): The 3D position of the base station [x, y, z].
        ue_pos (np.ndarray): The 3D position of the user equipment [x, y, z].
        antenna (Antenna): The Antenna object used by the base station.

    Returns:
        float: The channel gain (a linear magnitude value > 0).
    """
    # Speed of light in m/s
    c = 3.0e8
    
    # Distance between base station and user
    distance = np.linalg.norm(bs_pos - ue_pos)
    if distance == 0:
        return 1.0  # Avoid division by zero; max signal strength

    # Frequency in Hz
    freq_hz = antenna.freq * 1e9
    
    # Wavelength in meters
    wavelength = c / freq_hz
    
    # Calculate Free-Space Path Loss (FSPL) in dB
    # FSPL = 20 * log10(d) + 20 * log10(f) + 20 * log10(4*pi/c)
    # Simplified FSPL in linear scale: (lambda / (4 * pi * d))^2
    path_loss_linear = (wavelength / (4 * np.pi * distance)) ** 2
    
    # Convert antenna gain from dBi to a linear scale
    # P_linear = 10^(P_db/10)
    gain_linear = 10 ** (antenna.gain_dbi / 10.0)
    
    # Effective gain considering the number of elements (simple linear scaling)
    # This is a simplification; in reality, this would involve beamforming patterns.
    effective_gain = gain_linear * antenna.elements

    # The final channel gain is the product of path loss and antenna gain
    channel_gain = path_loss_linear * effective_gain
    
    return channel_gain

if __name__ == '__main__':
    # Example usage for testing
    
    # Create an antenna
    ant = Antenna(freq=3.5, elements=4)
    
    # Define positions for a base station and a user
    bs_position = np.array([0, 0, 10])  # Base station at 10m height
    ue_position = np.array([100, 50, 1.5]) # User 100m away, at 1.5m height
    
    # Compute the channel gain
    cfr_value = compute_cfr(bs_position, ue_position, ant)
    
    print(f"Antenna: {ant.freq} GHz, {ant.elements} elements")
    print(f"BS Position: {bs_position}, UE Position: {ue_position}")
    print(f"Distance: {np.linalg.norm(bs_position - ue_position):.2f} m")
    print(f"Calculated Channel Gain (linear): {cfr_value:.4e}")

    # Convert to dB for interpretation
    cfr_db = 10 * np.log10(cfr_value)
    print(f"Calculated Channel Gain (dB): {cfr_db:.2f} dB") 