#!/usr/bin/env python3
# tesseract_true_batch.py - Process images using Tesseract's native batch processing

import os
import sys
import argparse
import time
import subprocess
from datetime import datetime
import glob
import tempfile
import json
import shutil

# Disclaimer: This isnt part of Tesseracts documentation but it worked well 

# Add parent directory to path, to be able to find ocr_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr_utils import (
    ensure_dir_exists,
    get_output_path,
    TimingTracker
)

# Default paths and configurations
DEFAULT_INPUT_DIR = "/app/input"
DEFAULT_OUTPUT_DIR = "/app/ocr_outputs/ocr_tesseract_batch"

def get_env_var(name, default=None):
    """Get environment variable with fallback to default value."""
    return os.environ.get(name, default)

def process_batch_native(image_paths, output_dir, input_dir, tesseract_config, verbose=False):
    """
    Process a batch of images using Tesseract's native batch processing.
    
    Args:
        image_paths: List of paths to input images
        output_dir: Directory for output files
        input_dir: Base input directory (for calculating relative paths)
        tesseract_config: Tesseract configuration string
        verbose: Whether to print verbose information
    
    Returns:
        Dictionary with results
    """
    # Create a temporary directory for batch processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a list file containing all image paths
        list_file_path = os.path.join(temp_dir, "batch_list.txt")
        with open(list_file_path, 'w') as list_file:
            for img_path in image_paths:
                list_file.write(f"{img_path}\n")
        
        # Create a temporary directory for output files
        temp_output_dir = os.path.join(temp_dir, "output")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Prepare the Tesseract command
        # Parse the config string into arguments
        config_parts = tesseract_config.split()
        
        # Base command
        cmd = ['tesseract']
        
        # Add the list file with input paths
        cmd.append(list_file_path)
        
        # Add output directory prefix
        cmd.append(os.path.join(temp_output_dir, "out"))
        
        # Add configuration parameters
        cmd.extend(config_parts)
        
        # Add batch processing mode
        cmd.extend(['txt', 'batch', 'batch.nochop'])
        
        if verbose:
            print(f"Running Tesseract batch command: {' '.join(cmd)}")
            print(f"Processing {len(image_paths)} images in batch")
        
        # Start timing
        start_time = time.time()
        
        # Run Tesseract in batch mode
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            success = result.returncode == 0
            error_message = result.stderr if not success else None
            
            if not success and verbose:
                print(f"Tesseract batch command failed: {error_message}")
                print(f"Command output: {result.stdout}")
                
        except Exception as e:
            success = False
            error_message = str(e)
            if verbose:
                print(f"Exception running Tesseract batch command: {error_message}")
        
        # Record timing
        total_time = time.time() - start_time
        
        if not success:
            return {
                "success": False,
                "error": error_message,
                "batch_size": len(image_paths),
                "total_time": total_time,
                "processed_files": 0
            }
        
        # Process output files and move them to their final locations
        processed_files = 0
        
        # Debug: list all files in the temp output directory
        if verbose:
            print(f"Files in temp output directory: {os.listdir(temp_output_dir)}")
        
        # Check for a single output file with multiple pages
        single_output = os.path.join(temp_output_dir, "out.txt")
        if os.path.exists(single_output):
            if verbose:
                print(f"Found single output file: {single_output}")
                
            # Read the file content
            with open(single_output, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Split by form feed character if present
            pages = content.split('\f')
            
            # If pages match number of images, distribute them
            if len(pages) == len(image_paths):
                if verbose:
                    print(f"Split single output into {len(pages)} pages")
                    
                for i, (page, image_path) in enumerate(zip(pages, image_paths)):
                    # Create output directory structure and path
                    rel_path = os.path.relpath(os.path.dirname(image_path), input_dir)
                    final_output_dir = os.path.join(output_dir, rel_path)
                    ensure_dir_exists(final_output_dir)
                    
                    final_output_path = get_output_path(image_path, final_output_dir)
                    
                    # Write the page content to the output file
                    with open(final_output_path, 'w', encoding='utf-8') as f:
                        f.write(page.strip())
                    
                    processed_files += 1
                    
                    if verbose and (i % 10 == 0 or i == len(pages) - 1):
                        print(f"Saved page {i+1}/{len(pages)} to {final_output_path}")
            else:
                # Single output but pages don't match images
                if verbose:
                    print(f"Single output has {len(pages)} pages but batch has {len(image_paths)} images")
        
        # Check for numbered output files
        output_files = sorted(glob.glob(os.path.join(temp_output_dir, "out_*.txt")))
        
        if output_files:
            if verbose:
                print(f"Found {len(output_files)} numbered output files")
                
            for i, output_file in enumerate(output_files):
                if i < len(image_paths):
                    # Get corresponding input image path
                    image_path = image_paths[i]
                    
                    # Create output directory structure and path
                    rel_path = os.path.relpath(os.path.dirname(image_path), input_dir)
                    final_output_dir = os.path.join(output_dir, rel_path)
                    ensure_dir_exists(final_output_dir)
                    
                    final_output_path = get_output_path(image_path, final_output_dir)
                    
                    # Copy the output file to its final location
                    try:
                        shutil.copy2(output_file, final_output_path)
                        processed_files += 1
                        
                        if verbose and (i % 10 == 0 or i == len(output_files) - 1):
                            print(f"Copied output file {i+1}/{len(output_files)} to {final_output_path}")
                    except Exception as e:
                        if verbose:
                            print(f"Error copying output file {output_file} to {final_output_path}: {str(e)}")
        
        # If no output files found yet, check for files named after input files
        if processed_files == 0:
            for i, img_path in enumerate(image_paths):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                possible_outputs = [
                    os.path.join(temp_output_dir, f"{base_name}.txt"),
                    os.path.join(temp_output_dir, f"out_{base_name}.txt")
                ]
                
                output_file = None
                for path in possible_outputs:
                    if os.path.exists(path):
                        output_file = path
                        break
                
                if output_file:
                    # Create output directory structure and path
                    rel_path = os.path.relpath(os.path.dirname(img_path), input_dir)
                    final_output_dir = os.path.join(output_dir, rel_path)
                    ensure_dir_exists(final_output_dir)
                    
                    final_output_path = get_output_path(img_path, final_output_dir)
                    
                    # Copy the output file to its final location
                    try:
                        shutil.copy2(output_file, final_output_path)
                        processed_files += 1
                        
                        if verbose:
                            print(f"Copied output file {output_file} to {final_output_path}")
                    except Exception as e:
                        if verbose:
                            print(f"Error copying output file {output_file} to {final_output_path}: {str(e)}")
        
        # Last resort: if tesseract produced no output files in expected format
        # but the command appeared to succeed, create empty output files
        if processed_files == 0 and success:
            if verbose:
                print("No output files found in expected formats. Creating empty output files.")
                
            for i, img_path in enumerate(image_paths):
                # Create output directory structure and path
                rel_path = os.path.relpath(os.path.dirname(img_path), input_dir)
                final_output_dir = os.path.join(output_dir, rel_path)
                ensure_dir_exists(final_output_dir)
                
                final_output_path = get_output_path(img_path, final_output_dir)
                
                # Create an empty file
                with open(final_output_path, 'w', encoding='utf-8') as f:
                    f.write("# No output produced by Tesseract batch processing")
                
                processed_files += 1
                
                if verbose and (i % 10 == 0 or i == len(image_paths) - 1):
                    print(f"Created empty output file {i+1}/{len(image_paths)}")
        
        return {
            "success": success,
            "batch_size": len(image_paths),
            "processed_files": processed_files,
            "total_time": total_time,
            "avg_time_per_image": total_time / len(image_paths) if len(image_paths) > 0 else 0
        }

def process_images_in_batches(input_dir, output_dir, batch_size, tesseract_config, verbose):
    """
    Process all images in the input directory using Tesseract's native batch processing.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save OCR output
        batch_size: Number of images to process in a batch
        tesseract_config: Tesseract configuration string
        verbose: Whether to print verbose information
    
    Returns:
        Tuple of (processed_count, total_count)
    """
    # Create metrics directory
    metrics_dir = os.path.join(output_dir, "metrics")
    ensure_dir_exists(metrics_dir)
    
    # Initialize timing tracker
    timing_tracker = TimingTracker("tesseract_native_batch", metrics_dir)
    
    # Get all image files
    image_pattern = os.path.join(input_dir, "**", "*.png")
    image_files = glob.glob(image_pattern, recursive=True)
    
    # Also check for jpg files
    jpg_pattern = os.path.join(input_dir, "**", "*.jpg")
    image_files.extend(glob.glob(jpg_pattern, recursive=True))
    
    # Check for tif/tiff files
    tif_pattern = os.path.join(input_dir, "**", "*.tif")
    image_files.extend(glob.glob(tif_pattern, recursive=True))
    tiff_pattern = os.path.join(input_dir, "**", "*.tiff")
    image_files.extend(glob.glob(tiff_pattern, recursive=True))
    
    total_images = len(image_files)
    if total_images == 0:
        print(f"No images found in {input_dir}")
        return 0, 0
    
    print(f"Found {total_images} images to process")
    
    # Divide images into batches
    batches = [image_files[i:i+batch_size] for i in range(0, total_images, batch_size)]
    print(f"Divided into {len(batches)} batches of up to {batch_size} images each")
    
    processed_count = 0
    batch_count = 0
    
    # Process each batch
    for batch_index, batch in enumerate(batches):
        print(f"Processing batch {batch_index+1}/{len(batches)} ({len(batch)} images)")
        
        # Process the batch using native Tesseract batch processing
        batch_result = process_batch_native(
            batch,
            output_dir,
            input_dir,  # Pass the input directory for relative path calculation
            tesseract_config,
            verbose
        )
        
        batch_count += 1
        processed_count += batch_result["processed_files"]
        
        # Add timing data
        for image_path in batch:
            # Extract newspaper name for metrics
            parts = os.path.normpath(image_path).split(os.sep)
            newspaper = parts[-2] if len(parts) >= 2 else "unknown"
            
            if batch_result["success"]:
                avg_time = batch_result["avg_time_per_image"]
                timing_tracker.add_timing(
                    newspaper,
                    os.path.basename(image_path),
                    0,  # No separate preprocessing time
                    avg_time,
                    "success"
                )
            else:
                timing_tracker.add_timing(
                    newspaper,
                    os.path.basename(image_path),
                    0, 0,
                    f"error: {batch_result.get('error', 'unknown')}"
                )
        
        # Print progress
        print(f"Progress: {processed_count}/{total_images} images ({processed_count/total_images*100:.1f}%)")
    
    # Get final statistics
    stats = timing_tracker.get_averages()
    
    # Save summary metrics as JSON
    metrics = {
        "total_images": total_images,
        "processed_images": processed_count,
        "success_rate": processed_count / total_images if total_images > 0 else 0,
        "avg_recognition_time": stats["avg_recognition"],
        "avg_total_time": stats["avg_total"],
        "total_batches": batch_count,
        "batch_size": batch_size,
        "timestamp": datetime.now().isoformat(),
        "tesseract_config": tesseract_config
    }
    
    metrics_file = os.path.join(metrics_dir, f"tesseract_native_batch_metrics_{int(time.time())}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Metrics saved to {metrics_file}")
    print(f"Detailed timings saved to {timing_tracker.csv_path}")
    
    return processed_count, total_images

def main():
    parser = argparse.ArgumentParser(description="Process images with Tesseract OCR using native batch processing")
    
    # Required arguments
    parser.add_argument("--batch-size", type=int, default=20,
                      help="Number of images to process in a batch (default: 20)")
    
    # Optional arguments
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                      help="Input directory containing images")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                      help="Base output directory for OCR results")
    parser.add_argument("--tesseract-config", default="--oem 3 --psm 3 -l swe",
                      help="Tesseract configuration string")
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("="*50)
    print("   Tesseract OCR Native Batch Processing")
    print("="*50)
    
    # Override with environment variables if available
    input_dir = get_env_var("INPUT_DIR", args.input_dir)
    output_dir = get_env_var("OCR_TESSERACT_DIR", args.output_dir)
    
    # Ensure output directory exists
    ensure_dir_exists(output_dir)
    
    start_time = time.time()
    
    # Process all images
    success_count, total_images = process_images_in_batches(
        input_dir,
        output_dir,
        args.batch_size,
        args.tesseract_config,
        args.verbose
    )
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print("   Tesseract OCR Native Batch Processing Summary")
    print("="*50)
    print(f"Total images processed: {success_count}/{total_images}")
    print(f"Total elapsed time: {total_time:.1f} seconds")
    if total_time > 0:
        print(f"Processing rate: {success_count/total_time:.2f} images/second")
    
    # Return exit code based on success
    return 0 if success_count == total_images else 1

if __name__ == "__main__":
    sys.exit(main())