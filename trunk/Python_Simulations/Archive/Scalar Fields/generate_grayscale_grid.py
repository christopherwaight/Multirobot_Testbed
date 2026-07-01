#!/usr/bin/env python3
"""
Generate a simple grid of all 256 grayscale levels for printing
16x16 grid = 256 cells, one for each grey level from 0 (black) to 255 (white)
"""

import numpy as np
from PIL import Image

def generate_256_level_grid(cell_size=100, output_file='grayscale_grid_256.png'):
    """
    Generate 16x16 grid with all 256 grey levels

    Parameters:
    -----------
    cell_size : int
        Pixels per cell (default 100)
    output_file : str
        Output filename
    """

    grid_size = 16  # 16x16 = 256 cells
    img_size = cell_size * grid_size

    # Create image array
    grid = np.zeros((img_size, img_size), dtype=np.uint8)

    # Fill each cell with a different grey level (0-255)
    level = 0
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate pixel coordinates for this cell
            y_start = row * cell_size
            y_end = (row + 1) * cell_size
            x_start = col * cell_size
            x_end = (col + 1) * cell_size

            # Fill cell with current grey level
            grid[y_start:y_end, x_start:x_end] = level
            level += 1

    # Save image
    img = Image.fromarray(grid, mode='L')
    img.save(output_file, dpi=(300, 300))

    print(f"16×16 Grayscale Grid Generated")
    print(f"=" * 50)
    print(f"File: {output_file}")
    print(f"Grid: 16×16 cells (256 total)")
    print(f"Cell size: {cell_size}×{cell_size} pixels")
    print(f"Total size: {img_size}×{img_size} pixels")
    print(f"Grey levels: 0 (black) to 255 (white)")
    print(f"Layout: Row 0 = levels 0-15")
    print(f"        Row 1 = levels 16-31")
    print(f"        ...")
    print(f"        Row 15 = levels 240-255")
    print(f"=" * 50)

    return img


if __name__ == "__main__":
    # Generate the grid
    generate_256_level_grid(cell_size=100, output_file='grayscale_grid_256.png')

    print("\nPrinting instructions:")
    print("  1. Print at actual size (no scaling)")
    print("  2. Use matte paper for consistent reflectance")
    print("  3. Disable color management")
    print("  4. Each cell is a different grey level for sensor testing")
