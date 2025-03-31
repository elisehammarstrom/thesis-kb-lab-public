#!/usr/bin/env python3
# tesseract_ocr_processing.py - Process images using Tesseract OCR with multiprocessing
import os
import sys
import argparse
import time
from datetime import datetime
import pytesseract
from PIL import Image
import glob
import multiprocessing as mp
import functools
from tqdm import tqdm
import json

# Add parent directory to path, to be able to find ocr_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr_utils import (
    ensure_dir_exists,
    get_output_path,
    TimingTracker,
    convert_to_png
)

# Default paths and configurations
DEFAULT_INPUT_DIR = "/app/input"
DEFAULT_OUTPUT_DIR = "/app/ocr_outputs/ocr_tesseract_v5.5.0"

# Global variables for multiprocessing
_timing_tracker = None
_tesseract_config = None

def get_env_var(name, default=None):
    """Get environment variable with fallback to default value."""
    return os.environ.get(name, default)

def process_text_box_tesseract(image_path, output_path, verbose=False, tesseract_config=None):
    """
    Process a single text box image with Tesseract OCR.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save OCR output
        verbose: Whether to print verbose information
        tesseract_config: Optional Tesseract configuration string
    
    Returns:
        Dictionary with processing results
    """
    print(f"Processing {os.path.basename(image_path)} in process {os.getpid()}")
    try:
        # Start timing
        preprocess_start = time.time()
        
        if verbose:
            print(f"\nDetailed debugging for {image_path}:")
            print(f"1. Output path: {output_path}")
            print(f"2. Image exists: {os.path.exists(image_path)}")
        
        # Read the image
        image = Image.open(image_path)
        
        if verbose:
            print("3. Successfully opened image")
        
        preprocess_end = time.time()
        preprocess_time = preprocess_end - preprocess_start
        
        # Recognition timing
        recognition_start = time.time()
        
        # Configure Tesseract 
        if tesseract_config is None:
            tesseract_config = r'--oem 3 --psm 3 -l swe'
            
        # Perform OCR
        text = pytesseract.image_to_string(image, config=tesseract_config).strip()
        
        recognition_end = time.time()
        recognition_time = recognition_end - recognition_start
        
        if verbose:
            print(f"4. OCR completed, text length: {len(text)}")
        
        # Create directory for the output
        output_dir = os.path.dirname(output_path)
        ensure_dir_exists(output_dir)
        
        if verbose:
            print(f"5. Will save to: {output_path}")
            print(f"6. Output dir exists after mkdir: {os.path.exists(output_dir)}")
        
        # Save the OCR text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        if verbose:
            print(f"7. File written. File exists: {os.path.exists(output_path)}")
            print(f"8. File content preview: {text[:100]}")
        
        return {
            "image_path": image_path,
            "output_path": output_path,
            "success": True,
            "text_length": len(text),
            "preprocessing_time": preprocess_time,
            "recognition_time": recognition_time,
            "total_time": preprocess_time + recognition_time,
            "text": text
        }
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        print(f"Error type: {type(e)}")
        return {
            "image_path": image_path,
            "error": str(e),
            "success": False,
            "preprocessing_time": 0,
            "recognition_time": 0,
            "total_time": 0,
            "text": ""
        }

def process_with_timing_wrapper(image_info, output_base_dir, verbose=False):
    """
    Wrapper function for multiprocessing that uses global variables.
    This must be at the module level (not inside another function).
    
    Args:
        image_info: Tuple of (image_path, rel_path, filename)
        output_base_dir: Base directory to save OCR output
        verbose: Whether to print verbose information
    
    Returns:
        Boolean indicating success
    """
    global _timing_tracker, _tesseract_config
    
    image_path, rel_path, filename = image_info
    
    # Create the same directory structure in output
    # Replace .jpg/.png extension with .txt for output
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_base_dir, rel_path, f"{base_name}.txt")
    
    # Process the image
    result = process_text_box_tesseract(image_path, output_path, verbose, _tesseract_config)
    
    # Extract newspaper name from rel_path for timing tracking
    parts = rel_path.split(os.sep)
    if len(parts) >= 2:  # Should have at least old/new and newspaper name
        newspaper_dir = parts[1]  # Get newspaper name
    else:
        newspaper_dir = "unknown"
    
    # Record timing information
    if result["success"]:
        _timing_tracker.add_timing(
            newspaper_dir, 
            filename, 
            result["preprocessing_time"], 
            result["recognition_time"], 
            "success"
        )
    else:
        _timing_tracker.add_timing(
            newspaper_dir, 
            filename, 
            0, 0, 
            f"error: {result.get('error', 'unknown')}"
        )
    
    return result["success"]

def process_with_multiprocessing(images, output_dir, verbose=False, nprocs=None):
    """
    Process images in parallel using a multiprocessing pool.
    
    Args:
        images: List of image info tuples (path, rel_path, filename)
        output_dir: Output directory for the processed results
        verbose: Whether to print verbose output
        nprocs: Number of processes to use (defaults to CPU count)
        
    Returns:
        Tuple of (success_count, failure_count)
    """
    if not images:
        return 0, 0
        
    # Set default number of processes if not specified
    if nprocs is None:
        nprocs = mp.cpu_count()
    
    success_count = 0
    failure_count = 0
    
    # Create partial function with fixed arguments
    myfun = functools.partial(
        process_with_timing_wrapper, 
        output_base_dir=output_dir,
        verbose=verbose
    )
    
    # Calculate appropriate chunk size
    # Ensure we have at least 1 as chunk size
    chunksize = max(1, len(images) // nprocs // 2)
    
    if verbose:
        print(f"Processing {len(images)} images with {nprocs} processes (chunksize: {chunksize})")
    
    # Process images using multiprocessing pool
    with mp.Pool(processes=nprocs) as pool:
        for result in tqdm(
            pool.imap_unordered(
                myfun, 
                images, 
                chunksize=chunksize
            ),
            total=len(images),
            desc="Processing images"
        ):
            if result:
                success_count += 1
            else:
                failure_count += 1
    
    return success_count, failure_count

def get_all_image_files(input_dir):
    """
    Get all image files in the input directory maintaining the full relative path structure.
    Returns a list of tuples: (image_path, relative_path, filename)
    """
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                # Get the full path to the image
                image_path = os.path.join(root, file)
                
                # Create relative path (keeping old/new structure)
                rel_path = os.path.relpath(root, input_dir)
                
                # Add to our list
                image_files.append((image_path, rel_path, file))
    
    return image_files

def main():
    parser = argparse.ArgumentParser(description="Process images with Tesseract OCR using multiprocessing")
    
    # Required arguments
    parser.add_argument("--batch-size", type=int, default=100,
                      help="Number of images to process in a batch for memory management")
    
    # Optional arguments
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                      help="Input directory containing extracted text box images")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                      help="Base output directory for OCR results")
    parser.add_argument("--tesseract-config", default="--oem 3 --psm 3 -l swe",
                      help="Tesseract configuration string")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count(),
                      help="Number of parallel worker processes")
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--gpu-id", type=int, default=0,
                      help="GPU ID to use (used for compatibility with other models)")
    parser.add_argument("--cpu-only", action="store_true",
                      help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    print("="*50)
    print("   Tesseract OCR Multiprocessing")
    print("="*50)
    
    # Override output directory with environment variable if available
    output_base_dir = get_env_var("OCR_TESSERACT_DIR", args.output_dir)
    ensure_dir_exists(output_base_dir)
    
    # Initialize global variables
    global _timing_tracker, _tesseract_config
    _tesseract_config = args.tesseract_config
    
    # Create metrics directory
    metrics_dir = os.path.join(output_base_dir, "metrics")
    ensure_dir_exists(metrics_dir)
    _timing_tracker = TimingTracker("tesseract_mp", metrics_dir)
    
    start_time = time.time()
    
    # Collect all image files
    print("Collecting image files...")
    image_files = get_all_image_files(args.input_dir)
    total_images = len(image_files)
    print(f"Found {total_images} image files")
    
    if not image_files:
        print("No files to process")
        return 0
    
    success_count = 0
    failure_count = 0
    
    # Process all images or in batches for memory management
    if args.batch_size and args.batch_size < total_images:
        # Divide images into batches
        batches = [image_files[i:i+args.batch_size] for i in range(0, total_images, args.batch_size)]
        print(f"Divided into {len(batches)} batches of up to {args.batch_size} images each")
        
        for batch_index, batch in enumerate(batches):
            print(f"Processing batch {batch_index+1}/{len(batches)} ({len(batch)} images)")
            
            # Process this batch with multiprocessing
            batch_success, batch_failure = process_with_multiprocessing(
                batch,
                output_base_dir,
                args.verbose,
                args.num_workers
            )
            
            success_count += batch_success
            failure_count += batch_failure
            
            # Print progress
            print(f"Batch {batch_index+1} complete: {batch_success} succeeded, {batch_failure} failed")
            processed_so_far = success_count + failure_count
            print(f"Overall progress: {processed_so_far}/{total_images} images ({processed_so_far/total_images*100:.1f}%)")
            
            # Force garbage collection
            import gc
            gc.collect()
    else:
        # Process all at once
        success_count, failure_count = process_with_multiprocessing(
            image_files,
            output_base_dir,
            args.verbose,
            args.num_workers
        )
    
    total_processed = success_count + failure_count
    total_time = time.time() - start_time
    
    # Get final statistics
    stats = _timing_tracker.get_averages()
    
    # Save summary metrics as JSON
    metrics = {
        "total_images": total_processed,
        "successful_images": success_count,
        "success_rate": (success_count / total_processed * 100) if total_processed > 0 else 0,
        "avg_preprocessing_time": stats["avg_preprocessing"],
        "avg_recognition_time": stats["avg_recognition"],
        "avg_total_time": stats["avg_total"],
        "timestamp": datetime.now().isoformat(),
        "tesseract_config": args.tesseract_config,
        "num_workers": args.num_workers or mp.cpu_count()
    }
    
    metrics_file = os.path.join(metrics_dir, f"tesseract_mp_metrics_{int(time.time())}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Metrics saved to {metrics_file}")
    print(f"Detailed timings saved to {_timing_tracker.csv_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("   Tesseract OCR Multiprocessing Summary")
    print("="*50)
    print(f"Total images processed successfully: {success_count}/{total_processed}")
    print(f"Total elapsed time: {total_time:.1f} seconds")
    if total_time > 0:
        print(f"Processing rate: {success_count/total_time:.2f} images/second")
    
    # Return exit code based on success
    return 0 if success_count == total_processed else 1

if __name__ == "__main__":
    sys.exit(main())