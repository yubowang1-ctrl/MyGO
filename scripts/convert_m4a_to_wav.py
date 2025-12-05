import os
import argparse
import subprocess
from glob import glob
from tqdm import tqdm
import concurrent.futures

def convert_file(args):
    """
    Converts a single m4a file to wav using ffmpeg.
    args: tuple (file_path, input_dir, output_dir)
    """
    file_path, input_dir, output_dir = args
    
    try:
        if output_dir:
            # Preserve directory structure relative to input_dir
            rel_path = os.path.relpath(file_path, input_dir)
            wav_rel_path = rel_path.rsplit('.', 1)[0] + '.wav'
            wav_path = os.path.join(output_dir, wav_rel_path)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        else:
            # Save alongside original file
            wav_path = file_path.rsplit('.', 1)[0] + '.wav'
        
        # Skip if wav already exists
        if os.path.exists(wav_path):
            return True
            
        # Run ffmpeg command
        # -i: input file
        # -ac 2: force stereo (2 channels)
        # -ar 48000: sample rate 48kHz
        # -y: overwrite output
        # -loglevel error: suppress output unless error
        command = [
            'ffmpeg', 
            '-i', file_path,
            '-ac', '2',
            '-ar', '48000',
            '-y', 
            '-loglevel', 'error',
            wav_path
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
        
    except subprocess.CalledProcessError:
        print(f"Error converting {file_path}: ffmpeg failed.")
        return False
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert all .m4a files in a directory to .wav")
    parser.add_argument("--input_dir", required=True, help="Directory containing .m4a files")
    parser.add_argument("--output_dir", help="Optional output directory. If not specified, saves alongside input files.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--delete_original", action="store_true", help="Delete original .m4a files after successful conversion")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a directory.")
        return

    # Find all m4a files recursively
    print(f"Scanning {args.input_dir} for .m4a files...")
    # Use glob to find files. Note: recursive=True requires python 3.10+ for glob
    # For older python, we can use os.walk, but let's assume modern python or simple glob
    files = glob(os.path.join(args.input_dir, "**", "*.m4a"), recursive=True)
    
    if not files:
        print("No .m4a files found.")
        return
        
    print(f"Found {len(files)} files. Starting conversion with {args.workers} workers...")
    
    # Prepare arguments for map
    # Each task needs (file_path, input_dir, output_dir)
    task_args = [(f, args.input_dir, args.output_dir) for f in files]
    
    success_count = 0
    error_count = 0
    
    # Use ProcessPoolExecutor for parallel conversion
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Map returns results in order
        results = list(tqdm(executor.map(convert_file, task_args), total=len(files), unit="file"))
        
    for i, success in enumerate(results):
        if success:
            success_count += 1
            if args.delete_original:
                try:
                    os.remove(files[i])
                except OSError as e:
                    print(f"Error deleting {files[i]}: {e}")
        else:
            error_count += 1
            
    print(f"\nConversion complete.")
    print(f"Successfully converted: {success_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main()
