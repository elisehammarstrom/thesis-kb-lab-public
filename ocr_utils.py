import os
import re
import csv
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import multiprocessing as mp
import functools

# Define dataset periods as a constant
DATASET_PERIODS = ["1818-1870", "1871-1906"]

# Process tracker for timing information
# Add these imports at the top
import csv
from datetime import datetime

# Then modify the TimingTracker class to write to a CSV file
class TimingTracker:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.timings = []
        self.total_processed = 0
        self.successful_processed = 0
        self.total_time = 0
        self.total_preprocessing_time = 0
        self.total_recognition_time = 0
        
        ensure_dir_exists(output_dir)
        
        # Create or open CSV file for appending
        self.csv_path = os.path.join(output_dir, f"{model_name}_timings.csv")
        if os.path.exists(self.csv_path):
            print(f"TimingTracker: CSV file exists at {self.csv_path}")
        else:
            print(f"TimingTracker: Creating new CSV file at {self.csv_path}")
            # Create CSV file with headers if it doesn't exist
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "newspaper", "filename", "preprocessing_time", 
                                "recognition_time", "total_time", "status"])
    
    def add_timing(self, newspaper, filename, preprocessing_time, recognition_time, status):
        total_time = preprocessing_time + recognition_time
        timestamp = datetime.now().isoformat()
        
        # Append directly to CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, newspaper, filename, preprocessing_time, 
                            recognition_time, total_time, status])
        
        # Update counters for statistics
        self.total_processed += 1
        
        if "success" in status.lower():
            self.successful_processed += 1
            self.total_preprocessing_time += preprocessing_time
            self.total_recognition_time += recognition_time
            self.total_time += total_time
        
        # Also store in memory for statistics
        self.timings.append({
            "newspaper": newspaper,
            "filename": filename,
            "preprocessing_time": preprocessing_time,
            "recognition_time": recognition_time,
            "total_time": total_time,
            "status": status
        })
    
    def get_averages(self):
        """
        Get current average processing times
        """
        if self.total_processed == 0:
            return {
                "avg_total": 0,
                "avg_preprocessing": 0,
                "avg_recognition": 0,
                "success_rate": 0,
                "total_processed": 0
            }
        
        successful = max(1, self.successful_processed)  # Avoid division by zero
        
        return {
            "avg_total": self.total_time / successful if successful > 0 else 0,
            "avg_preprocessing": self.total_preprocessing_time / successful if successful > 0 else 0,
            "avg_recognition": self.total_recognition_time / successful if successful > 0 else 0,
            "success_rate": (self.successful_processed / self.total_processed) * 100 if self.total_processed > 0 else 0,
            "total_processed": self.total_processed
        }

def get_image_files_in_period(input_dir, period):
    """
    Get all image files in the given period and their corresponding output directories.
    """
    image_files = []
    target_start, target_end = map(int, period.split('-'))
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jpg"):
                year_match = re.search(r'(\d{4})', Path(root).parts[-1])
                if not year_match:
                    continue
                    
                year = int(year_match.group(1))
                if target_start <= year <= target_end:
                    newspaper_dir = Path(root).parts[-1]
                    image_files.append((os.path.join(root, file), newspaper_dir, file))
    
    return image_files

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)
    return os.path.exists(directory)

def process_with_multiprocessing(images, process_func, output_dir, verbose=False, nprocs=None):
    """
    Process images in parallel using a multiprocessing pool.
    
    Args:
        images: List of image paths to process
        process_func: Function to process each image (must accept image_path, output_dir, verbose)
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
        process_func, 
        output_dir=output_dir,
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

def get_output_path(image_path, output_dir, extension=".txt"):
    """Get standardized output path for a given image"""
    image_filename = os.path.basename(image_path)
    base_name = os.path.splitext(image_filename)[0]
    output_filename = f"{base_name}{extension}"
    return os.path.join(output_dir, output_filename)

def convert_to_png(jpg_path, output_dir):
    """
    Convert a JPG image to PNG format for better OCR processing
    
    Args:
        jpg_path: Path to the JPEG image
        output_dir: Directory to save the PNG image
    
    Returns:
        Path to the converted PNG image
    """
    # Create output directory if it doesn't exist already
    ensure_dir_exists(output_dir)
    
    # Get base filename without file extension
    basename = os.path.basename(jpg_path)
    filename_wo_ext = os.path.splitext(basename)[0]
    
    # Create output PNG path
    png_path = os.path.join(output_dir, f"{filename_wo_ext}.png")
    
    try:
        # Check if PNG already exists (to avoid reconverting)
        if os.path.exists(png_path):
            return png_path
            
        # Open and convert the image
        with Image.open(jpg_path) as img:
            # Save as PNG
            img.save(png_path, "PNG")
            
        return png_path
    except Exception as e:
        print(f"Error converting image {jpg_path}: {e}")
        return None

def process_period(input_dir, output_base_dir, period, process_func, verbose=False, batch_size=None):
    """
    Generic function to process all files for a specific period with a progress bar.
    
    Args:
        input_dir: Directory with input images
        output_base_dir: Base directory for outputs
        period: Time period (e.g., "1818-1870")
        process_func: Function to process each image (must accept image_path, output_dir, verbose)
        verbose: Whether to print verbose information
        batch_size: Optional batch size for memory management
    """
    # Get all image files for this period
    print(f"\nCollecting image files for period {period}...")
    image_files = get_image_files_in_period(input_dir, period)
    print(f"Found {len(image_files)} image files for period {period}")
    
    if not image_files:
        print(f"No files to process for period {period}")
        return 0
    
    # Process files with progress bar
    print(f"Processing files for period {period}...")
    
    if batch_size:
        # Process in batches
        total_images = len(image_files)
        batches = [image_files[i:i + batch_size] for i in range(0, total_images, batch_size)]
        print(f"Split into {len(batches)} batches of up to {batch_size} images each")
        
        success_count = 0
        failure_count = 0
        batch_count = 1
        
        for batch in batches:
            print(f"Processing batch {batch_count}/{len(batches)}")
            
            for image_path, newspaper_dir, filename in tqdm(batch, desc=f"Batch {batch_count}"):
                output_dir = os.path.join(output_base_dir, newspaper_dir)
                result = process_func(image_path, output_dir, verbose)
                
                if result:
                    success_count += 1
                else:
                    failure_count += 1
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print(f"Batch {batch_count}/{len(batches)} completed. Stats: {success_count} succeeded, {failure_count} failed")
            batch_count += 1
        
        return total_images
    else:
        # Process all at once
        for image_path, newspaper_dir, filename in tqdm(image_files, desc=f"Processing {period}"):
            output_dir = os.path.join(output_base_dir, newspaper_dir)
            process_func(image_path, output_dir, verbose)
        
        return len(image_files)