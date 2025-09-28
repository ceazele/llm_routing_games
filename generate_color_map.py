import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define the size of the image
width, height = 1200, 100  # Long and skinny

def generate_coolwarm_colormap():
    """Generates and displays a coolwarm colormap with a transparent background."""
    
    # Create a gradient from 0 to 1
    gradient = np.linspace(0, 1, width).reshape(1, -1)
    gradient = np.vstack([gradient] * height)  # Stack vertically to make it taller
    
    # Create figure and axis with transparent background
    fig, ax = plt.subplots(figsize=(12, 1), dpi=100, frameon=False)
    fig.patch.set_alpha(0)  # Transparent figure background
    ax.set_facecolor("none")  # Transparent axis background
    
    # Display the colormap
    ax.imshow(gradient, aspect='auto', cmap='Blues', extent=[0, width, 0, height])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # Save as a transparent PNG
    plt.savefig("coolwarm_colormap.png", transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# Generate and display the colormap
generate_coolwarm_colormap()
