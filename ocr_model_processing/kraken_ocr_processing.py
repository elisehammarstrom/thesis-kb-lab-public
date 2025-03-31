#!/usr/bin/env python3
import subprocess
import os
import sys
import logging
import time
import re
from tqdm import tqdm
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_chosen = os.getenv('KRAKEN_MODEL', '10.5281/zenodo.10519596')
model_name = os.getenv('KRAKEN_MODEL_NAME', 'german_print.mlmodel')
kraken_model_dir = '/usr/local/share/ocrd-resources/kraken/'
model_path = os.path.join(kraken_model_dir, model_name)

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def list_models():
    """List available models in Kraken"""
    result = subprocess.run(['kraken', 'list'], capture_output=True, text=True)
    print(result.stdout)
    return result.stdout

def download_model(model_identifier):
    """Download or locate model"""
    try:
        # Define the global path and kraken model directory
        global model_path
        global model_name
        
        # Check if model_identifier is a direct path to an existing model file
        if os.path.exists(model_identifier) and model_identifier.endswith('.mlmodel'):
            print(f"Using existing model file: {model_identifier}")
            model_path = model_identifier
            return "Using existing model file"
        
        # First try to create the target directory if it doesn't exist
        if not os.path.exists(kraken_model_dir):
            try:
                os.makedirs(kraken_model_dir, exist_ok=True)
                print(f"Created directory: {kraken_model_dir}")
            except Exception as e:
                print(f"Failed to create directory {kraken_model_dir}: {e}")
                print("Will try downloading to the current directory instead")
        
        # Define the expected path for the model
        expected_model_path = os.path.join(kraken_model_dir, model_name)
        model_path = expected_model_path  # Update the global model path
        
        # If model_identifier is a Zenodo ID, download it
        if model_identifier.startswith('10.5281/zenodo.'):
            # Try downloading with kraken get
            print(f"Downloading model {model_identifier}...")
            result = subprocess.run(['kraken', 'get', model_identifier], capture_output=True, text=True)
            print(f"Model {model_identifier} download attempt result: {result.returncode}")
            
            # Check if the model was downloaded successfully
            if result.returncode == 0:
                print(f"Model {model_identifier} downloaded successfully with kraken")
                
                # Verify the model file exists
                if os.path.exists(expected_model_path):
                    print(f"Verified model exists at: {expected_model_path}")
                    return "Downloaded successfully"
                else:
                    print(f"Model not found at expected path: {expected_model_path}")
                    print("Searching for the downloaded model...")
                    # Search for the model in possible locations
                    possible_locations = [
                        kraken_model_dir,
                        '/app/models',
                        os.path.expanduser('~/.local/share/ocrd-resources/kraken/'),
                        os.path.expanduser('~/.kraken'),
                        '/usr/share/ocrd-resources/kraken/',
                        '/tmp'
                    ]
                    
                    for location in possible_locations:
                        if os.path.exists(location):
                            import glob
                            models = glob.glob(os.path.join(location, '*.mlmodel'))
                            if models:
                                # Look specifically for the model with our filename
                                for model_file in models:
                                    if os.path.basename(model_file) == model_name:
                                        print(f"Found {model_name} at: {model_file}")
                                        model_path = model_file
                                        return "Found in alternate location"
                                
                                # If we didn't find our specific model, use the first model
                                print(f"{model_name} not found. Using first available model: {models[0]}")
                                model_path = models[0]
                                return "Using alternative model"
            
            # If download failed, try direct download
            print(f"Standard download failed. Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            print("Trying alternative download method...")
            
            # Try a direct download using requests
            try:
                import requests
                
                # Construct URL based on model ID pattern for Zenodo
                if model_identifier.startswith('10.5281/zenodo.'):
                    zenodo_id = model_identifier.split('.')[-1]
                    url = f"https://zenodo.org/record/{zenodo_id}/files/{model_name}?download=1"
                    print(f"Trying direct download from: {url}")
                    
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    # Save to the expected location
                    with open(expected_model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"Successfully downloaded model to {expected_model_path}")
                    model_path = expected_model_path
                    return "Downloaded with requests"
                else:
                    print(f"Unknown model format: {model_identifier}")
            except Exception as e:
                print(f"Alternative download failed: {e}")
        else:
            print(f"Model identifier {model_identifier} is not a valid Zenodo ID or file path.")
            
        # If all download attempts failed, try to find an existing model
        print("Will try to use an existing model...")
        find_downloaded_model_paths()
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        print("Will try to use an existing model...")
        find_downloaded_model_paths()

def find_downloaded_model_paths():
    """Find the downloaded model paths in the Docker container"""
    global model_path
    
    # First check if the expected model exists
    expected_path = os.path.join(kraken_model_dir, "german_print.mlmodel")
    if os.path.exists(expected_path):
        print(f"Found expected model at: {expected_path}")
        model_path = expected_path
        return [expected_path]
    
    # First check the standard location
    if not os.path.exists(kraken_model_dir):
        print(f"Config directory {kraken_model_dir} not found")
        # Try to create the directory
        try:
            os.makedirs(kraken_model_dir, exist_ok=True)
            print(f"Created directory: {kraken_model_dir}")
        except Exception as e:
            print(f"Failed to create directory: {e}")
    
    # Search for models in standard location
    import glob
    model_paths = glob.glob(os.path.join(kraken_model_dir, '*.mlmodel'))
    
    # Prioritize finding german_print.mlmodel
    for path in model_paths:
        if os.path.basename(path) == model_name:
            print(f"Found german_print.mlmodel at: {path}")
            model_path = path
            return [path]
    
    if not model_paths:
        # Try to find models in alternative locations
        alt_locations = [
            '/app/models',
            os.path.expanduser('~/.local/share/ocrd-resources/kraken/'),
            os.path.expanduser('~/.kraken'),
            '/usr/share/ocrd-resources/kraken/',
            '/tmp'
        ]
        
        # First try to find german_print.mlmodel specifically
        for location in alt_locations:
            if os.path.exists(location):
                print(f"Checking alternative location: {location}")
                german_print_path = os.path.join(location, "german_print.mlmodel")
                if os.path.exists(german_print_path):
                    print(f"Found german_print.mlmodel at: {german_print_path}")
                    model_path = german_print_path
                    return [german_print_path]
                
                # If not found, check for any .mlmodel files
                alt_models = glob.glob(os.path.join(location, '*.mlmodel'))
                if alt_models:
                    print(f"Found {len(alt_models)} models in {location}:")
                    for model in alt_models:
                        print(model)
                    
                    # Use the first model found as our model path
                    model_path = alt_models[0]
                    print(f"Using model: {model_path}")
                    return alt_models
    
    if model_paths:
        print(f"Found {len(model_paths)} models in {kraken_model_dir}:")
        for model in model_paths:
            print(model)
        # Update the global model path
        model_path = model_paths[0]
        print(f"Using model: {model_path}")
    else:
        print(f"No models found in {kraken_model_dir} or alternative locations")
        # Set a default search path for the OCR command
        model_path = "german_print.mlmodel"
        print(f"Using default model name: {model_path}")
    
    return model_paths

def verify_model_before_ocr():
    """Verify that the correct model is being used before running OCR"""
    global model_path
    
    print(f"\nVerifying model path before OCR: {model_path}")
    
    # Check if the model exists at the specified path
    if os.path.exists(model_path):
        print(f"Model file exists at: {model_path}")
        print(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        return model_path
    
    # If model_path is a Zenodo ID or similar, try to download it
    if model_path.startswith('10.5281/zenodo.'):
        print(f"Model path appears to be a Zenodo ID: {model_path}")
        download_model(model_path)
        if os.path.exists(model_path):
            print(f"Downloaded model to: {model_path}")
            return model_path
    
    # If all else fails, try to find any available model
    print(f"WARNING: Model file not found at: {model_path}")
    print("Attempting to find any available model...")
    find_downloaded_model_paths()
    
    # Check again after find_downloaded_model_paths
    if os.path.exists(model_path):
        print(f"Model found at: {model_path}")
    else:
        print(f"Failed to find model. OCR may fail.")
            
    return model_path

def determine_period(image_path):
    """Determine if a file belongs to 'old' or 'new' period based on path or filename"""
    # Extract year from filename if possible
    # Common pattern: bib11653809_18450503_146885_34_0002_003.png (year is 1845)
    filename = os.path.basename(image_path)
    year_match = re.search(r'_(\d{4})\d{4}_', filename)
    
    if year_match:
        year = int(year_match.group(1))
        if year < 1871:
            return "old"
        else:
            return "new"
    
    # If can't determine from filename, check parent directory name
    # Some newspapers might have year in directory name
    parent_dir = os.path.basename(os.path.dirname(image_path))
    year_match = re.search(r'(\d{4})', parent_dir)
    
    if year_match:
        year = int(year_match.group(1))
        if year < 1871:
            return "old"
        else:
            return "new"
    
    # Default to "old" if we can't determine
    return "old"

def convert_to_png(image_path, output_dir):
    """Convert image to PNG format if needed"""
    try:
        ensure_dir_exists(output_dir)
        
        # Get file extension and create output path
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}.png")
        
        # If already PNG, just copy the file
        if ext.lower() == '.png':
            import shutil
            shutil.copy2(image_path, output_path)
            return output_path
        
        # Otherwise convert using PIL
        from PIL import Image
        img = Image.open(image_path)
        img.save(output_path, 'PNG')
        return output_path
    except Exception as e:
        logger.error(f"Error converting to PNG: {e}")
        return None

class TimingTracker:
    """Track processing times for OCR operations"""
    def __init__(self, model_name, metrics_dir):
        self.model_name = model_name
        self.metrics_dir = metrics_dir
        self.stats = {
            'total_preprocessing': 0.0,
            'total_recognition': 0.0,
            'total_count': 0,
            'success_count': 0,
            'newspaper_stats': {}
        }
        ensure_dir_exists(metrics_dir)
    
    def add_timing(self, newspaper, filename, preprocessing_time, recognition_time, status):
        """Add timing information for an OCR operation"""
        # Update newspaper stats if it doesn't exist
        if newspaper not in self.stats['newspaper_stats']:
            self.stats['newspaper_stats'][newspaper] = {
                'total_preprocessing': 0.0,
                'total_recognition': 0.0,
                'total_count': 0,
                'success_count': 0,
                'files': {}
            }
        
        # Add file stats
        self.stats['newspaper_stats'][newspaper]['files'][filename] = {
            'preprocessing_time': preprocessing_time,
            'recognition_time': recognition_time,
            'total_time': preprocessing_time + recognition_time,
            'status': status
        }
        
        # Update counters
        self.stats['total_preprocessing'] += preprocessing_time
        self.stats['total_recognition'] += recognition_time
        self.stats['total_count'] += 1
        
        # Update success counters if successful
        if status == 'success':
            self.stats['success_count'] += 1
            self.stats['newspaper_stats'][newspaper]['success_count'] += 1
        
        # Update newspaper counters
        self.stats['newspaper_stats'][newspaper]['total_preprocessing'] += preprocessing_time
        self.stats['newspaper_stats'][newspaper]['total_recognition'] += recognition_time
        self.stats['newspaper_stats'][newspaper]['total_count'] += 1
        
        # Save to file after each update
        self._save_stats()
        
        # Return averages for display
        return self.get_averages()
    
    def get_averages(self):
        """Calculate average processing times"""
        if self.stats['total_count'] == 0:
            return {
                'avg_preprocessing': 0.0,
                'avg_recognition': 0.0,
                'avg_total': 0.0,
                'success_rate': 0.0
            }
        
        return {
            'avg_preprocessing': self.stats['total_preprocessing'] / self.stats['total_count'],
            'avg_recognition': self.stats['total_recognition'] / self.stats['total_count'],
            'avg_total': (self.stats['total_preprocessing'] + self.stats['total_recognition']) / self.stats['total_count'],
            'success_rate': (self.stats['success_count'] / self.stats['total_count']) * 100.0
        }
    
    def _save_stats(self):
        """Save statistics to a file"""
        import json
        output_file = os.path.join(self.metrics_dir, f"{self.model_name}_timing_stats.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)

def get_image_files(input_dir, extensions=('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
    """Get all image files in directory and subdirectories"""
    image_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                newspaper_dir = os.path.basename(root)
                image_files.append((full_path, newspaper_dir, file))
    
    return image_files

def process_batch(input_dir, output_base_dir, metrics_dir, model_name, verbose=False, batch_size=10):
    """Process images using batch processing with Kraken's batch features"""
    # Initialize timing tracker 
    ensure_dir_exists(metrics_dir)
    timing_tracker = TimingTracker(model_name, metrics_dir)
    
    # Group the images by directory to facilitate batch processing
    print("Collecting image files...")
    image_files = get_image_files(input_dir)
    print(f"Found {len(image_files)} image files")
    
    if not image_files:
        print("No files to process")
        return 0
    
    # Group images by newspaper directory and period
    grouped_images = {}
    for image_path, newspaper_dir, filename in image_files:
        # Determine period (old or new)
        period = determine_period(image_path)
        
        # Create key with both period and newspaper directory
        key = f"{period}/{newspaper_dir}"
        
        if key not in grouped_images:
            grouped_images[key] = []
        grouped_images[key].append((image_path, filename))
    
    print(f"Found {len(grouped_images)} newspaper directories")
    
    # Setup for tracking results
    total_processed = 0
    success_count = 0
    failure_count = 0
    
    # Create overall progress bar
    overall_progress = tqdm(total=len(image_files), desc="Overall Progress", 
                           position=0, leave=True, 
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    
    # Process each newspaper directory as a batch
    for key, files in grouped_images.items():
        period, newspaper_dir = key.split('/', 1)
        print(f"\nProcessing {len(files)} files from {newspaper_dir} (period: {period})")
        
        # Create new directory structure: output_base_dir/period/newspaper_dir
        output_dir = os.path.join(output_base_dir, period, newspaper_dir)
        ensure_dir_exists(output_dir)
        
        # Prepare PNG files for batch processing - store directly in output_dir
        print("Converting images to PNG format...")
        png_files = []
        for image_path, filename in files:
            try:
                # Convert directly to output_dir (no png subdirectory)
                png_file = convert_to_png(image_path, output_dir) 
                if png_file:
                    png_files.append((png_file, filename))
            except Exception as e:
                logger.error(f"Error converting {image_path} to PNG: {e}")
                failure_count += 1
        
        if not png_files:
            print(f"No valid PNG files created for {newspaper_dir} in {period}. Skipping.")
            overall_progress.update(len(files))
            total_processed += len(files)
            continue
        
        # Split into smaller batches if necessary
        png_batches = [png_files[i:i + batch_size] for i in range(0, len(png_files), batch_size)]
        
        for batch_idx, batch in enumerate(png_batches):
            start_time = time.time()
            batch_file_count = len(batch)
            
            print(f"Running batch OCR for {newspaper_dir} in {period} (batch {batch_idx+1}/{len(png_batches)})")
            print(f"Processing {batch_file_count} files in this batch")
            
            # Process only the files in the current batch, not all PNG files
            batch_file_patterns = []
            batch_filenames = []
            
            for png_file, filename in batch:
                # Extract just the filename without path
                png_filename = os.path.basename(png_file)
                batch_file_patterns.append(os.path.join(output_dir, png_filename))
                batch_filenames.append(filename)
            
            # Create a temporary file with the batch file patterns
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                for pattern in batch_file_patterns:
                    f.write(f"{pattern}\n")
                batch_list_file = f.name
            
            # Run the batch OCR command using the file list
            try:
                # Use -i with each file individually instead of a glob pattern
                cmd = ['kraken']
                for pattern in batch_file_patterns:
                    cmd.extend(['-i', pattern, f"{pattern}.txt"])
                
                cmd.extend([
                    '-v',  # Verbose output
                    'binarize', 'segment', 'ocr',
                    '-m', model_path
                ])
                
                if verbose:
                    print(f"Running command: {' '.join(cmd)}")
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                # Process each line of output to update progress
                files_processed_in_batch = 0
                for line in iter(process.stdout.readline, ''):
                    if verbose:
                        print(line.strip())
                    if 'processing' in line.lower():
                        files_processed_in_batch += 1
                        overall_progress.update(1)
                        total_processed += 1
                
                # If the output didn't update progress for all files, update the remainder
                if files_processed_in_batch < batch_file_count:
                    remaining = batch_file_count - files_processed_in_batch
                    overall_progress.update(remaining)
                    total_processed += remaining
                
                # Wait for process to complete
                returncode = process.wait()
                
                # Check for errors
                if returncode != 0:
                    error_output = process.stderr.read()
                    logger.error(f"Batch OCR failed: {error_output}")
                    failure_count += batch_file_count
                else:
                    # Count successes by checking output files
                    batch_success = 0
                    for i, (png_file, filename) in enumerate(batch):
                        # Check if text file exists and is not empty
                        txt_path = f"{png_file}.txt"
                        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                            batch_success += 1
                            success_count += 1
                            
                            # Move the file to the expected location if necessary
                            base_name = os.path.splitext(filename)[0]
                            expected_txt_path = os.path.join(output_dir, f"{base_name}.txt")
                            if txt_path != expected_txt_path:
                                import shutil
                                shutil.move(txt_path, expected_txt_path)
                        else:
                            failure_count += 1
                    
                    batch_time = time.time() - start_time
                    avg_time = batch_time / batch_file_count if batch_file_count else 0
                    
                    print(f"Batch {batch_idx+1}/{len(png_batches)} completed: {batch_success}/{batch_file_count} succeeded")
                    print(f"Average time per image: {avg_time:.3f}s")
                    
                    # Update timing statistics
                    for i, (png_file, filename) in enumerate(batch):
                        base_name = os.path.splitext(filename)[0]
                        txt_path = os.path.join(output_dir, f"{base_name}.txt")
                        status = "success" if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0 else "failure"
                        
                        timing_tracker.add_timing(
                            newspaper_dir, 
                            filename, 
                            avg_time / 3,  # Estimate preprocessing time as 1/3
                            avg_time * 2/3,  # Estimate recognition time as 2/3
                            status
                        )
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                failure_count += batch_file_count
                # Update progress bar for the whole batch
                overall_progress.update(batch_file_count)
                total_processed += batch_file_count
            
            # Clean up the temporary file
            try:
                os.unlink(batch_list_file)
            except:
                pass
            
            # Update progress bar stats
            overall_progress.set_postfix(
                success=success_count,
                failed=failure_count,
                newspaper=newspaper_dir
            )
        
        # Make sure we update progress for any files we might have missed
        remaining = len(files) - (total_processed - (len(image_files) - len(files)))
        if remaining > 0:
            overall_progress.update(remaining)
            total_processed += remaining
    
    # Close progress bar
    overall_progress.close()
    
    # Final summary
    avg_stats = timing_tracker.get_averages()
    print(f"\nProcessed {total_processed} files: {success_count} succeeded, {failure_count} failed")
    print(f"Final average processing time: {avg_stats['avg_total']:.3f}s")
    print(f"Success rate: {avg_stats['success_rate']:.1f}%")
    
    return total_processed

def check_gpu_availability():
    """Check if CUDA GPU is available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("GPU detected. Using CUDA acceleration.")
            return True
        else:
            print("No GPU detected. Falling back to CPU.")
            return False
    except Exception:
        print("Could not check for GPU. Falling back to CPU.")
        return False

def main(model_name="kraken-default", verbose=False, batch_size=10, resume_from=None):
    """Main function to process all images with batch processing"""
    # Check GPU availability
    gpu_available = check_gpu_availability()
    if not gpu_available:
        print("Note: Processing will be slower without GPU acceleration.")
        model_name += "-cpu"
    else:
        model_name += "-gpu"
    
    # Download the model if needed
    try:
        download_model(model_chosen)
        find_downloaded_model_paths()
    except Exception as e:
        logger.error(f"Error downloading or finding model: {e}")
        return
    
    # Get input directory from environment variable or use default
    input_dir = os.getenv('KRAKEN_INPUT_DIR', '/app/input')
    
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        # Try the backup path
        backup_input_dir = '/app/output/extracted_text_boxes'
        if os.path.exists(backup_input_dir):
            print(f"Primary input directory not found, using backup: {backup_input_dir}")
            input_dir = backup_input_dir
        else:
            print(f"Error: Input directory does not exist: {input_dir}")
            print(f"Backup input directory also does not exist: {backup_input_dir}")
            return
    
    # Get output directory from environment variable or use default
    output_base_dir = os.getenv('KRAKEN_OUTPUT_DIR', '/app/ocr_outputs/ocr_kraken/german_print_model')
    
    # Create metrics directory
    metrics_dir = os.path.join(output_base_dir, "metrics")
    ensure_dir_exists(metrics_dir)
    
    # Create period directories
    ensure_dir_exists(os.path.join(output_base_dir, "old"))
    ensure_dir_exists(os.path.join(output_base_dir, "new"))
    
    # Verify model before starting OCR
    verify_model_before_ocr()
    
    if verbose:
        print(f"\nInitial setup:")
        print(f"Input directory: {input_dir}")
        print(f"Input directory exists: {os.path.exists(input_dir)}")
        print(f"Base output directory: {output_base_dir}")
        print(f"Metrics directory: {metrics_dir}")
    
    # Process using batch processing
    total_files_processed = process_batch(
        input_dir, 
        output_base_dir,
        metrics_dir,
        model_name,
        verbose,
        batch_size
    )
    
    print("\nKraken OCR batch processing completed")
    print(f"Total files processed: {total_files_processed}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run Kraken OCR with batch processing')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-b', '--batch-size', type=int, default=10, help='Number of images to process in a batch')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to use (cpu, cuda:0, etc.)')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('-i', '--input', type=str, help='Input directory with images')
    parser.add_argument('-o', '--output', type=str, help='Output directory for OCR results')
    parser.add_argument('-m', '--model', type=str, help='Model name or path')
    parser.add_argument('-I', '--batch-input', type=str, help='Glob pattern for batch input')
    parser.add_argument('--resume-from', type=str, help='Resume processing from a specific file pattern')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (default: 0)')
    
    args = parser.parse_args()
    
    # Set environment variables from arguments if provided
    if args.input:
        os.environ['KRAKEN_INPUT_DIR'] = args.input
    if args.output:
        os.environ['KRAKEN_OUTPUT_DIR'] = args.output
    if args.model:
        os.environ['KRAKEN_MODEL'] = args.model
    if args.device and args.device != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]
    
    # Force set smaller batch size to prevent memory issues
    batch_size = args.batch_size if args.batch_size <= 20 else 10
    
    print(f"\n=== Kraken OCR Batch Processing ===")
    print(f"- Using batch size: {batch_size}")
    print(f"- Verbose mode: {'On' if args.verbose else 'Off'}")
    print(f"- Device: {args.device}")
    if args.resume_from:
        print(f"- Resuming from: {args.resume_from}")
    print("============================\n")
    
    try:
        main(
            model_name=args.model if args.model else "kraken-default",
            verbose=args.verbose, 
            batch_size=batch_size,
            resume_from=args.resume_from
        )
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        print("You can resume processing later with the --resume-from option.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)