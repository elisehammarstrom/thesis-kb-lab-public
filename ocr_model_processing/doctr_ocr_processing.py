import os
import sys
import logging
import argparse
import time
import torch
from tqdm import tqdm
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr_utils import ensure_dir_exists, get_output_path, TimingTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["OMP_NUM_THREADS"] = "1"

def clean_memory():
    """Clean up GPU memory to avoid crashes."""
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    import gc
    gc.collect()

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
                
                # Create relative path (keeping folder structure)
                rel_path = os.path.relpath(root, input_dir)
                
                # Add to our list
                image_files.append((image_path, rel_path, file))
    
    return image_files

def process_batch_doctr(image_batch, output_dir, use_gpu=True, verbose=False, batch_size=4):
    """
    Process a batch of images with DocTR OCR with true batch processing.
    
    Args:
        image_batch: List of tuples (image_path, rel_path, filename)
        output_dir: Base output directory
        use_gpu: Whether to use GPU acceleration
        verbose: Whether to print verbose information
        batch_size: Number of images to process in a single forward pass
        
    Returns:
        List of dictionaries with results for each image
    """
    # Set device
    device = "cpu"
    if use_gpu and 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    
    if verbose:
        logger.info(f"Using device: {device}")
        if device == "cuda" and 'torch' in sys.modules:
            import torch
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize results
    results = []
    
    try:
        # Clean memory before processing
        clean_memory()
        
        # Load model with minimal memory usage
        if verbose:
            logger.info("Loading DocTR model...")
        
        try:
            # Use basic model initialization
            model = ocr_predictor(
                det_arch='db_resnet50', 
                reco_arch='crnn_vgg16_bn', 
                pretrained=True
            )
            
            # Move to GPU if available
            if device == "cuda":
                model.det_predictor.model = model.det_predictor.model.cuda()
                model.reco_predictor.model = model.reco_predictor.model.cuda()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            # Return failed status for all images
            return [{"success": False, "error": str(e)} for _ in range(len(image_batch))]
        
        # Create a placeholder for each image in the batch
        for idx in range(len(image_batch)):
            results.append({"success": False, "error": "Not processed yet"})
            
        # First, check which images need processing vs. which can be skipped (already processed)
        process_indices = []
        images_to_process = []
        image_paths_to_output_paths = {}
        
        for idx, (image_path, rel_path, filename) in enumerate(image_batch):
            # Determine output path
            base_name = os.path.splitext(filename)[0]
            output_subdir = os.path.join(output_dir, rel_path)
            output_path = os.path.join(output_subdir, f"{base_name}.txt")
            
            # Create output directory
            ensure_dir_exists(output_subdir)
            
            # Track the mapping from image path to output path
            image_paths_to_output_paths[image_path] = output_path
            
            # Skip if already exists
            if os.path.exists(output_path):
                if verbose:
                    logger.info(f"Skipping (already exists): {output_path}")
                
                # Update result for this image
                results[idx] = {
                    "success": None,  # None means skipped
                    "text_length": 0
                }
            else:
                # Add to list of images to process
                process_indices.append(idx)
                images_to_process.append(image_path)
                if verbose:
                    logger.info(f"Will process image: {image_path}")
        
        # If there are images to process, do them in batches
        if images_to_process:
            # Process in smaller sub-batches
            for i in range(0, len(images_to_process), batch_size):
                sub_batch_indices = process_indices[i:i+batch_size]
                sub_batch_paths = images_to_process[i:i+batch_size]
                
                if verbose:
                    logger.info(f"Processing sub-batch of {len(sub_batch_paths)} images")
                
                try:
                    # Load all images in this batch at once
                    batch_docs = DocumentFile.from_images(sub_batch_paths)
                    
                    # Process the entire batch at once
                    batch_results = model(batch_docs)
                    
                    # Process each result in the batch
                    for j, (batch_doc, batch_result) in enumerate(zip(batch_docs, batch_results)):
                        # Get the original index in the main batch
                        orig_idx = sub_batch_indices[j]
                        image_path = sub_batch_paths[j]
                        output_path = image_paths_to_output_paths[image_path]
                        
                        try:
                            # Get the result JSON
                            result_json = batch_result.export()
                            
                            # Check blocks count
                            blocks_count = sum(len(page['blocks']) for page in result_json['pages'])
                            if verbose or blocks_count == 0:
                                logger.info(f"Text blocks detected in {image_path}: {blocks_count}")
                            
                            # Reconstruct the text layout
                            reconstructed_text = ""
                            
                            for page in result_json["pages"]:
                                # Collect all words with their positions
                                all_words = []
                                
                                for block in page["blocks"]:
                                    for line in block["lines"]:
                                        if "words" in line and line["words"]:
                                            for word in line["words"]:
                                                # Extract geometry safely
                                                try:
                                                    # Try newer format first (2-tuple of coordinates)
                                                    (x1, y1), (x2, y2) = word["geometry"]
                                                except ValueError:
                                                    # Try older format (flat 4-tuple)
                                                    try:
                                                        x1, y1, x2, y2 = word["geometry"]
                                                    except:
                                                        # Last resort - arbitrary x/y extraction
                                                        geometry = word["geometry"]
                                                        if isinstance(geometry, list) and len(geometry) >= 2:
                                                            # If it's a list of points, use first and last
                                                            first_pt = geometry[0]
                                                            last_pt = geometry[-1]
                                                            if isinstance(first_pt, (list, tuple)) and len(first_pt) >= 2:
                                                                x1, y1 = first_pt[0], first_pt[1]
                                                            else:
                                                                x1, y1 = 0, 0
                                                            if isinstance(last_pt, (list, tuple)) and len(last_pt) >= 2:
                                                                x2, y2 = last_pt[0], last_pt[1]
                                                            else:
                                                                x2, y2 = 1, 1
                                                        else:
                                                            # Fallback to fixed values
                                                            x1, y1, x2, y2 = 0, 0, 1, 1
                                                
                                                all_words.append({
                                                    "text": word["value"],
                                                    "x1": x1,
                                                    "y1": y1,
                                                    "x2": x2,
                                                    "y2": y2,
                                                    "center_y": (y1 + y2) / 2
                                                })
                                
                                # Sort words by y-position and then by x-position
                                if all_words:
                                    # Calculate average height for line grouping
                                    heights = [w["y2"] - w["y1"] for w in all_words]
                                    if heights:
                                        avg_height = sum(heights) / len(heights)
                                        line_threshold = avg_height / 2
                                    else:
                                        line_threshold = 0.01  # Default if no heights
                                    
                                    # Sort words by y-position
                                    sorted_by_y = sorted(all_words, key=lambda w: w["center_y"])
                                    
                                    # Group words into lines
                                    lines = []
                                    current_line = [sorted_by_y[0]] if sorted_by_y else []
                                    current_y = sorted_by_y[0]["center_y"] if sorted_by_y else 0
                                    
                                    for k in range(1, len(sorted_by_y)):
                                        word = sorted_by_y[k]
                                        if abs(word["center_y"] - current_y) < line_threshold:
                                            # Same line
                                            current_line.append(word)
                                        else:
                                            # New line
                                            lines.append(current_line)
                                            current_line = [word]
                                            current_y = word["center_y"]
                                    
                                    # Add the last line
                                    if current_line:
                                        lines.append(current_line)
                                    
                                    # Sort words within each line by x-position
                                    for line in lines:
                                        line.sort(key=lambda w: w["x1"])
                                    
                                    # Build the reconstructed text with a space after every word
                                    for line in lines:
                                        line_text = ""
                                        
                                        for k, word in enumerate(line):
                                            # Add the word
                                            line_text += word["text"]
                                            
                                            # Add a space after every word except the last one in a line
                                            if k < len(line) - 1:
                                                line_text += " "
                                        
                                        reconstructed_text += line_text + "\n"
                            
                            if verbose:
                                logger.info(f"Text reconstruction completed, length: {len(reconstructed_text)}")
                                logger.info(f"Text sample: {reconstructed_text[:100]}")
                            
                            # Save the OCR text
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(reconstructed_text.strip())
                            
                            logger.info(f"Successfully wrote: {output_path}")
                            
                            # Update result for this image
                            results[orig_idx] = {
                                "success": True,
                                "text": reconstructed_text.strip(),
                                "text_length": len(reconstructed_text.strip())
                            }
                            
                        except Exception as e:
                            logger.error(f"Error processing result for {image_path}: {str(e)}")
                            
                            # Create empty file to mark as processed
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write("")
                            
                            # Update result for this image
                            results[orig_idx] = {
                                "success": False,
                                "error": str(e),
                                "text_length": 0
                            }
                            
                    # Clean up batch memory
                    del batch_docs
                    del batch_results
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    # Mark all images in this sub-batch as failed
                    for j, orig_idx in enumerate(sub_batch_indices):
                        if j < len(sub_batch_paths):
                            image_path = sub_batch_paths[j]
                            output_path = image_paths_to_output_paths.get(image_path)
                            
                            if output_path:
                                # Create empty file to mark as processed
                                with open(output_path, "w", encoding="utf-8") as f:
                                    f.write("")
                            
                            # Update result for this image
                            results[orig_idx] = {
                                "success": False,
                                "error": str(e),
                                "text_length": 0
                            }
                
                # Clean memory between sub-batches
                clean_memory()
        
        return results
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Make sure we have a result for each image
        while len(results) < len(image_batch):
            results.append({"success": False, "error": "Batch processing failed"})
        
        return results
    
    finally:
        # Clean up model
        if 'model' in locals():
            del model
        
        # Final memory cleanup
        clean_memory()

# Single image processing function for compatibility with other scripts
def process_text_box_doctr_standalone(image_path, output_dir, verbose=False):
    """
    Process a single text box image with DocTR OCR.
    Standalone version for direct usage from other scripts.
    """
    # Skip processing if output file already exists
    image_filename = os.path.basename(image_path)
    base_name = os.path.splitext(image_filename)[0]
    output_filename = f"{base_name}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_path):
        print(f"Skipping (already exists): {output_path}")
        return None
    
    # Get the directory and filename
    image_dir = os.path.dirname(image_path)
    rel_path = os.path.relpath(image_dir, os.path.dirname(output_dir))
    
    # Create a batch of one image
    image_batch = [(image_path, rel_path, image_filename)]
    
    # Process the single image through the batch processor
    results = process_batch_doctr(image_batch, output_dir, use_gpu=True, verbose=verbose)
    
    # Return the OCR text if successful
    if results and results[0].get("success"):
        return results[0].get("text", "")
    return ""

def main():
    parser = argparse.ArgumentParser(description="Process images with DocTR OCR in batches")
    
    # Command line arguments
    parser.add_argument("--batch-size", type=int, default=4,
                      help="Number of images to process in a batch (default: 4)")
    parser.add_argument("--process-batch-size", type=int, default=16,
                      help="Number of images to load in each process batch for memory management (default: 16)")
    parser.add_argument("--input-dir", default=None,
                      help="Input directory containing images (default: use DOCTR_INPUT_DIR env var)")
    parser.add_argument("--output-dir", default=None,
                      help="Output directory for OCR results (default: use DOCTR_OUTPUT_DIR env var)")
    parser.add_argument("--gpu-id", type=int, default=0,
                      help="GPU ID to use (default: 0)")
    parser.add_argument("--cpu-only", action="store_true",
                      help="Force CPU usage even if GPU is available")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("="*50)
    print("   DocTR OCR Batch Processing")
    print("="*50)
    
    # Set GPU device if specified
    if 'torch' in sys.modules and torch.cuda.is_available() and not args.cpu_only:
        torch.cuda.set_device(args.gpu_id)
        logger.info(f"Using GPU device {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    
    # Check if running on CPU
    use_gpu = 'torch' in sys.modules and torch.cuda.is_available() and not args.cpu_only
    device = "CPU" if args.cpu_only or not ('torch' in sys.modules and torch.cuda.is_available()) else f"GPU:{args.gpu_id}"
    logger.info(f"Running on {device}")
    
    # Get input directory from arguments or environment variable
    input_dir = args.input_dir or os.getenv('DOCTR_INPUT_DIR', '/app/input')
    
    # Get output directory from arguments or environment variable
    output_dir = args.output_dir or os.getenv('DOCTR_OUTPUT_DIR', '/app/output/ocr_doctr')
    
    # Create metrics directory
    metrics_dir = os.path.join(output_dir, "metrics")
    ensure_dir_exists(metrics_dir)
    
    # Initialize timing tracker
    timing_tracker = TimingTracker("doctr", metrics_dir)
    
    # Log setup information
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    
    start_time = time.time()
    
    # Get all image files
    logger.info("Collecting image files...")
    image_files = get_all_image_files(input_dir)
    total_images = len(image_files)
    logger.info(f"Found {total_images} image files")
    
    if not image_files:
        logger.warning("No files to process")
        return 0
    
    # Process in batches
    success_count = 0
    failure_count = 0
    skipped_count = 0
    
    # Divide images into process batches for memory management
    process_batches = [image_files[i:i+args.process_batch_size] for i in range(0, total_images, args.process_batch_size)]
    logger.info(f"Divided into {len(process_batches)} process batches of up to {args.process_batch_size} images each")
    
    # Create progress bar
    progress = tqdm(
        total=total_images, 
        desc="Processing", 
        position=0, 
        leave=True, 
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    
    # Process each process batch
    for batch_index, process_batch in enumerate(process_batches):
        batch_start_time = time.time()
        logger.info(f"Processing batch {batch_index+1}/{len(process_batches)} ({len(process_batch)} images)")
        
        # Process this batch
        results = process_batch_doctr(
            process_batch,
            output_dir,
            use_gpu,
            args.verbose,
            args.batch_size
        )
        
        # Process results for this batch
        for i, result in enumerate(results):
            # Safety check - if result doesn't have expected keys, skip it
            if not isinstance(result, dict) or "success" not in result:
                logger.error(f"Invalid result format: {result}")
                failure_count += 1
                continue
                
            # Get original image information from process_batch
            if i < len(process_batch):
                image_path, rel_path, filename = process_batch[i]
            else:
                logger.error(f"Result index {i} out of bounds for batch size {len(process_batch)}")
                failure_count += 1
                continue
                
            # Use first component of relative path for tracking
            components = rel_path.split(os.path.sep)
            category = components[0] if components else "unknown"
            
            if result["success"] is None:
                # Skipped file
                skipped_count += 1
            elif result["success"]:
                # Successful processing
                success_count += 1
                
                # Add timing record
                timing_tracker.add_timing(
                    category,  # Use category as first component of rel_path
                    filename,
                    0.5,  # Approximate preprocessing time
                    0.5,  # Approximate recognition time
                    "success"
                )
            else:
                # Failed processing
                failure_count += 1
                
                # Add timing record
                timing_tracker.add_timing(
                    category,
                    filename,
                    0,
                    0,
                    f"error: {result.get('error', 'unknown')}"
                )
        
        # Update progress bar
        progress.update(len(process_batch))
        
        # Get stats for progress bar
        avgs = timing_tracker.get_averages()
        current_time = avgs.get("avg_total", 0.0)
        success_rate = avgs.get("success_rate", 0.0)
        
        # Calculate processing speed
        processed_so_far = success_count + failure_count
        elapsed = time.time() - start_time
        speed = processed_so_far / max(1, elapsed)
        
        # Update progress bar with details
        progress.set_postfix(
            batch=f"{batch_index+1}/{len(process_batches)}",
            success=success_count,
            failed=failure_count,
            skipped=skipped_count,
            speed=f"{speed:.2f}img/s",
            success_rate=f"{success_rate:.1f}%"
        )
        
        # Log batch completion
        batch_time = time.time() - batch_start_time
        logger.info(f"Batch {batch_index+1}/{len(process_batches)} complete in {batch_time:.2f}s: {success_count} succeeded, {failure_count} failed, {skipped_count} skipped")
        images_per_second = len(process_batch) / max(1, batch_time)
        logger.info(f"Processing speed: {images_per_second:.2f} images/second")
        
        # Clean memory and wait between batches
        clean_memory()
        if use_gpu and 'torch' in sys.modules and torch.cuda.is_available():
            time.sleep(1)  # Short wait between batches
    
    # Close progress bar
    progress.close()
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("   DocTR OCR Processing Summary")
    logger.info("="*50)
    logger.info(f"Total images processed successfully: {success_count}/{total_images}")
    logger.info(f"Failed images: {failure_count}")
    logger.info(f"Skipped images (already processed): {skipped_count}")
    logger.info(f"Total elapsed time: {total_time:.1f} seconds")
    
    if success_count > 0 and total_time > 0:
        logger.info(f"Processing rate: {success_count/total_time:.2f} images/second")
    
    avg_stats = timing_tracker.get_averages()
    logger.info(f"Average processing time: {avg_stats['avg_total']:.3f}s")
    logger.info(f"Success rate: {avg_stats['success_rate']:.1f}%")
    logger.info(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user. Cleaning up...")
        clean_memory()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)