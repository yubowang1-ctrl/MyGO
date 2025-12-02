#!/usr/bin/env python3
"""
Reconstruct .stats.json from existing .m4a files in a directory.
Only recovers 'downloaded' status. 'failed' and 'unavailable' will be empty.

Usage:
  python3 scripts/reconstruct_stats.py downloads/ --output .stats.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

def reconstruct_stats(directory: str, output_file: str) -> None:
    """Scans directory for .m4a files and creates a stats.json file."""
    
    target_dir = Path(directory)
    if not target_dir.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)

    downloaded_ids = []
    
    print(f"Scanning {directory} for .m4a files...")
    
    # Iterate over all files in the directory
    count = 0
    ignored_temp = 0
    
    for entry in tqdm(target_dir.iterdir(), desc="Scanning files"):
        if entry.is_file() and entry.suffix.lower() == '.m4a':
            stem = entry.stem
            
            # Filter out temporary files created by the downloader (ending in _temp)
            if stem.endswith("_temp"):
                ignored_temp += 1
                continue
                
            # The filename stem is the video_id
            downloaded_ids.append(stem)
            count += 1

    # Sort for consistency
    downloaded_ids.sort()

    stats_data = {
        "downloaded": downloaded_ids,
        "failed": [],
        "unavailable": []
    }

    try:
        with open(output_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print("-" * 40)
        print(f"Reconstruction Complete")
        print("-" * 40)
        print(f"Found downloaded: {count}")
        if ignored_temp > 0:
            print(f"Ignored temp files: {ignored_temp}")
        print(f"Saved stats to:   {output_file}")
        
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct .stats.json from existing .m4a files."
    )
    parser.add_argument(
        "directory",
        help="Directory containing the downloaded .m4a files"
    )
    parser.add_argument(
        "--output", "-o",
        default=".stats.json",
        help="Output JSON filename (default: .stats.json inside the directory)"
    )

    args = parser.parse_args()
    
    # Determine output path
    # If output is just a filename (no path separators), save it inside the target directory
    if os.path.dirname(args.output):
        out_path = args.output
    else:
        out_path = os.path.join(args.directory, args.output)
    reconstruct_stats(args.directory, out_path)

if __name__ == "__main__":
    main()