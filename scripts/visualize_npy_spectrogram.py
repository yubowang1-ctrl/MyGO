import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_npy(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load the .npy file
    # Expected shape: (Height/Freq, Width/Time, Channels)
    # Dtype: int8 (0-255)
    data = np.load(file_path)
    
    print(f"Loaded {file_path}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min: {data.min()}, Max: {data.max()}")
    
    data_fixed = data.astype(np.float32)
    # If values are negative, it might be the int8 overflow issue.
    if data.min() < 0:
        print("Detected negative values. Assuming int8 overflow from 0-255 range.")
        # Convert back to 0-255 range
        data_fixed = data.astype(np.uint8)
        print(f"Corrected range: Min: {data_fixed.min()}, Max: {data_fixed.max()}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Channel 0 (Left)
    im0 = axes[0].imshow(data_fixed[:, :, 0], aspect='auto', origin='lower', cmap='inferno')
    axes[0].set_title("Channel 0 (Left)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency (Mel)")
    plt.colorbar(im0, ax=axes[0])

    # Channel 1 (Right)
    if data.shape[2] > 1:
        im1 = axes[1].imshow(data_fixed[:, :, 1], aspect='auto', origin='lower', cmap='inferno')
        axes[1].set_title("Channel 1 (Right)")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Frequency (Mel)")
        plt.colorbar(im1, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, "No 2nd Channel", ha='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize .npy spectrogram")
    parser.add_argument("file_path", help="Path to the .npy file")
    args = parser.parse_args()
    
    visualize_npy(args.file_path)
