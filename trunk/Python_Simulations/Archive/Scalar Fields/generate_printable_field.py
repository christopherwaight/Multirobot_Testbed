#!/usr/bin/env python3
"""
Generate printable grayscale scalar field for hardware testing
Creates 256-level (8-bit) grayscale image of bimodal Gaussian saddle field
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

def bimodal_gaussian_field(x, y):
    """Bimodal Gaussian scalar field with saddle at origin"""
    gaussian1 = -((x + 2)**2 + y**2) / 2
    gaussian2 = -((x - 2)**2 + y**2) / 2
    max_exp = np.maximum(gaussian1, gaussian2)
    return max_exp + np.log(np.exp(gaussian1 - max_exp) + np.exp(gaussian2 - max_exp))

def generate_grayscale_field(resolution=1024, domain_size=3.0, output_file='bimodal_saddle_field.png'):
    """
    Generate high-resolution grayscale field for printing

    Parameters:
    -----------
    resolution : int
        Pixels per side (default 1024 for high quality)
    domain_size : float
        Field domain is [-domain_size, domain_size] in both x and y
    output_file : str
        Output filename
    """

    # Create coordinate grid
    x = np.linspace(-domain_size, domain_size, resolution)
    y = np.linspace(-domain_size, domain_size, resolution)
    X, Y = np.meshgrid(x, y)

    # Compute field values
    Z = bimodal_gaussian_field(X, Y)

    # Normalize to [0, 1]
    Z_min = np.min(Z)
    Z_max = np.max(Z)
    Z_normalized = (Z - Z_min) / (Z_max - Z_min)

    # Convert to 8-bit grayscale (0-255)
    Z_grayscale = (Z_normalized * 255).astype(np.uint8)

    # Create PIL image
    img = Image.fromarray(Z_grayscale, mode='L')

    # Save high-quality PNG
    img.save(output_file, dpi=(300, 300))

    print(f"Grayscale field saved to: {output_file}")
    print(f"Resolution: {resolution}x{resolution} pixels")
    print(f"Domain: [{-domain_size}, {domain_size}] x [{-domain_size}, {domain_size}] m")
    print(f"Grayscale levels: 256 (8-bit)")
    print(f"Min value: {Z_min:.4f}, Max value: {Z_max:.4f}")
    print(f"Saddle point at: (0, 0)")
    print(f"Peaks at: (-2, 0) and (2, 0)")

    return img, Z, Z_normalized


def generate_visualization(Z, Z_normalized, domain_size=3.0, output_file='field_visualization.png'):
    """Create visualization showing field structure"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Coordinate grid for visualization
    x = np.linspace(-domain_size, domain_size, Z.shape[0])
    y = np.linspace(-domain_size, domain_size, Z.shape[1])
    X, Y = np.meshgrid(x, y)

    # 1. Grayscale field (as will be printed)
    ax1 = axes[0]
    im1 = ax1.imshow(Z_normalized, cmap='gray', origin='lower',
                     extent=[-domain_size, domain_size, -domain_size, domain_size])
    ax1.plot(0, 0, 'rx', markersize=15, linewidth=3, label='Saddle')
    ax1.plot([-2, 2], [0, 0], 'b^', markersize=10, label='Peaks')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Grayscale Field (As Printed)', fontweight='bold', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Normalized Intensity (0=black, 1=white)')

    # 2. Contour plot
    ax2 = axes[1]
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(0, 0, 'rx', markersize=15, linewidth=3, label='Saddle')
    ax2.plot([-2, 2], [0, 0], 'b^', markersize=10, label='Peaks')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('Field Contours', fontweight='bold', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # 3. 3D surface
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    # Downsample for 3D plot
    skip = max(1, Z.shape[0] // 100)
    surf = ax3.plot_surface(X[::skip, ::skip], Y[::skip, ::skip], Z[::skip, ::skip],
                            cmap='viridis', alpha=0.8, edgecolor='none')
    ax3.plot([0], [0], [bimodal_gaussian_field(0, 0)], 'rx', markersize=10, linewidth=3)
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    ax3.set_zlabel('Field Value', fontsize=10)
    ax3.set_title('3D Field Surface', fontweight='bold', fontsize=13)
    cbar3 = plt.colorbar(surf, ax=ax3, shrink=0.5)

    plt.suptitle('Bimodal Gaussian Saddle Field', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.show()


def generate_calibration_strip(output_file='calibration_strip.png'):
    """Generate a calibration strip with known grayscale values"""

    # Create strip with 16 levels from black (0) to white (255)
    levels = np.linspace(0, 255, 16).astype(np.uint8)

    # Create image: 16 squares, each 200x200 pixels
    square_size = 200
    strip = np.zeros((square_size, square_size * 16), dtype=np.uint8)

    for i, level in enumerate(levels):
        strip[:, i*square_size:(i+1)*square_size] = level

    # Save calibration strip
    img = Image.fromarray(strip, mode='L')
    img.save(output_file, dpi=(300, 300))

    print(f"\nCalibration strip saved to: {output_file}")
    print(f"Contains {len(levels)} levels from 0 (black) to 255 (white)")
    print(f"Each square: {square_size}x{square_size} pixels")
    print(f"Levels: {levels}")

    return img


def generate_test_pattern_24x24(output_file='test_pattern_24x24.png'):
    """Generate a 24x24 test pattern with numbered cells for alignment testing"""

    cell_size = 100  # pixels per cell
    grid_size = 24

    # Create blank image
    img_size = cell_size * grid_size
    pattern = np.ones((img_size, img_size), dtype=np.uint8) * 255  # White background

    # Add grid lines
    for i in range(grid_size + 1):
        if i * cell_size < img_size:  # Boundary check
            # Vertical lines
            pattern[:, i * cell_size] = 0
            # Horizontal lines
            pattern[i * cell_size, :] = 0

    # Save image
    img = Image.fromarray(pattern, mode='L')
    img.save(output_file, dpi=(300, 300))

    print(f"\n24x24 test pattern saved to: {output_file}")
    print(f"Grid size: {grid_size}x{grid_size} cells")
    print(f"Cell size: {cell_size}x{cell_size} pixels")
    print(f"Total size: {img_size}x{img_size} pixels")

    return img


def main():
    """Generate all printable field files"""

    print("="*70)
    print("GENERATING PRINTABLE SCALAR FIELD FOR HARDWARE TESTING")
    print("="*70)

    # 1. Generate main grayscale field (high resolution for printing)
    print("\n1. Generating grayscale saddle field...")
    img, Z, Z_norm = generate_grayscale_field(
        resolution=2048,  # High resolution for printing
        domain_size=3.0,
        output_file='bimodal_saddle_field_printable.png'
    )

    # 2. Generate visualization
    print("\n2. Generating field visualization...")
    generate_visualization(Z, Z_norm, domain_size=3.0,
                          output_file='field_visualization.png')

    # 3. Generate calibration strip
    print("\n3. Generating calibration strip...")
    generate_calibration_strip(output_file='calibration_strip.png')

    # 4. Generate 24x24 test pattern
    print("\n4. Generating 24x24 test pattern...")
    generate_test_pattern_24x24(output_file='test_pattern_24x24.png')

    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print("\nFiles created:")
    print("  1. bimodal_saddle_field_printable.png - Main field for testing")
    print("  2. field_visualization.png - Visual reference")
    print("  3. calibration_strip.png - Sensor calibration reference")
    print("  4. test_pattern_24x24.png - Alignment test grid")
    print("\nPrinting recommendations:")
    print("  - Print on matte paper for consistent reflectance")
    print("  - Use high-quality inkjet or laser printer")
    print("  - Print at actual size (no scaling)")
    print("  - Ensure color management is off (pure grayscale)")
    print("  - Test with calibration strip first")
    print("="*70)


if __name__ == "__main__":
    main()
