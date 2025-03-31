import os
import sys
import logging
import argparse
import time
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_assistant_response(full_text):
    """Extract only the assistant's response from the full text output"""
    if "Assistant:" in full_text:
        # Extract everything after "Assistant:"
        response = full_text.split("Assistant:", 1)[1].strip()
        return response
    else:
        # If the format is different than expected, return the original text
        return full_text

def clean_memory():
    """Clean up GPU memory to avoid crashes."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Synchronize CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

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

def process_batch_idefics(image_batch, output_dir, use_gpu=True, verbose=False, batch_size=8):
    """
    Process a batch of images with IDEFICS in a single forward pass.
    
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
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    if verbose:
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize results
    results = []
    
    try:
        clean_memory()
        if verbose:
            logger.info("Loading IDEFICS model...")
        
        try:
            model_name = "HuggingFaceM4/Idefics3-8B-Llama3"
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16, 
                device_map=device,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return [{"success": False, "error": str(e)} for _ in range(len(image_batch))]
        
        # Create a placeholder for each image in the batch, such that we can see failed and success rate per image
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
            
            # Skip if already processed
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
                try:
                    # Try opening the image to validate it
                    with Image.open(image_path) as img:
                        process_indices.append(idx)
                        images_to_process.append(image_path)
                        if verbose:
                            logger.info(f"Will process image: {image_path}")
                except Exception as e:
                    logger.error(f"Error opening image {image_path}: {str(e)}")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write("")
                    
                    results[idx] = {
                        "success": False,
                        "error": f"Error opening image: {str(e)}",
                        "text_length": 0
                    }
        
        # If there are images to process, do them in a single forward pass for the entire batch
        if images_to_process:
            try:
                # Process images in smaller sub-batches if needed
                for i in range(0, len(images_to_process), batch_size):
                    sub_batch_indices = process_indices[i:i+batch_size]
                    sub_batch_paths = images_to_process[i:i+batch_size]
                    
                    if verbose:
                        logger.info(f"Processing sub-batch of {len(sub_batch_paths)} images")
                    
                    # Load images and filter in one pass
                    valid_images = []
                    valid_indices = []
                    for j, img_path in enumerate(sub_batch_paths):
                        try:
                            image = Image.open(img_path)
                            valid_images.append(image)
                            valid_indices.append(sub_batch_indices[j])
                        except Exception as e:
                            logger.error(f"Error reading image {img_path}: {str(e)}")
                            # Update the result for this failed image
                            orig_idx = sub_batch_indices[j]
                            output_path = image_paths_to_output_paths[img_path]
                            # Create empty file to mark as processed
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write("")
                            results[orig_idx] = {
                                "success": False,
                                "error": f"Error opening image: {str(e)}",
                                "text_length": 0
                            }
                    
                    if not valid_images:
                        continue
                    
                    # Prepare message with the prompt
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "If there is text in this image it is in Swedish and then I want you to transcribe it. If no text is visible in the image, leave the output blank. Always have no additional commentary."}
                            ]
                        }
                    ]
                    
                    # Process the input for all images at once
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(text=prompt, images=valid_images, return_tensors="pt")
                    
                    # Move to device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Generate with minimal memory settings
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=1500,
                            do_sample=False,
                            num_beams=1,  # Use greedy search (minimum memory)
                        )
                    
                    # Decode the outputs for all images
                    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # Process each result
                    for j, (idx, text) in enumerate(zip(valid_indices, generated_texts)):
                        # Extract the image path and output path
                        image_path = sub_batch_paths[j]
                        output_path = image_paths_to_output_paths[image_path]
                        
                        # Extract only the assistant's response
                        assistant_response = extract_assistant_response(text)
                        
                        # Save the output
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(assistant_response)
                        
                        # Update result for this image
                        results[idx] = {
                            "success": True,
                            "text_length": len(assistant_response)
                        }
                        
                        if verbose:
                            logger.info(f"Successfully saved OCR text to: {output_path}")
                            logger.info(f"Text length: {len(assistant_response)}")
                    
                    # Clean up images
                    for img in valid_images:
                        try:
                            img.close()
                        except:
                            pass
                    
                    # Clean memory after each sub-batch
                    del inputs, generated_ids
                    clean_memory()
            
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Mark all images in this batch as failed
                for idx in process_indices:
                    # Get the output path for this image
                    image_path = image_batch[idx][0]
                    output_path = image_paths_to_output_paths.get(image_path)
                    
                    if output_path:
                        # Create empty file to mark as processed
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write("")
                    
                    # Update result for this image
                    results[idx] = {
                        "success": False,
                        "error": str(e),
                        "text_length": 0
                    }
        
        # Return results for all images in the batch
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
        # Clean up model and processor
        if 'model' in locals():
            del model
        if 'processor' in locals():
            del processor
        
        # Final memory cleanup
        clean_memory()

def main():
    parser = argparse.ArgumentParser(description="Process images with IDEFICS OCR")
    
    # Command line arguments
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Number of images to process in a batch (default: 8)")
    parser.add_argument("--process-batch-size", type=int, default=96,
                      help="Number of images to load in each process batch for memory management (default: 96)")
    parser.add_argument("--input-dir", default=None,
                      help="Input directory containing images (default: use IDEFICS_INPUT_DIR env var)")
    parser.add_argument("--output-dir", default=None,
                      help="Output directory for OCR results (default: use IDEFICS_OUTPUT_DIR env var)")
    parser.add_argument("--gpu-id", type=int, default=0,
                      help="GPU ID to use (default: 0)")
    parser.add_argument("--cpu-only", action="store_true",
                      help="Force CPU usage even if GPU is available")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("="*50)
    print("   IDEFICS OCR Processing")
    print("="*50)
    
    # Set GPU device if specified
    if torch.cuda.is_available() and not args.cpu_only:
        torch.cuda.set_device(args.gpu_id)
        logger.info(f"Using GPU device {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    
    # Check if running on CPU
    use_gpu = torch.cuda.is_available() and not args.cpu_only
    device = "CPU" if args.cpu_only or not torch.cuda.is_available() else f"GPU:{args.gpu_id}"
    logger.info(f"Running on {device}")
    
    # Get input directory from arguments or environment variable
    input_dir = args.input_dir or os.getenv('IDEFICS_INPUT_DIR', '/app/input')
    
    # Get output directory from arguments or environment variable
    output_dir = args.output_dir or os.getenv('IDEFICS_OUTPUT_DIR', '/app/output/ocr_idefics')
    
    # Create metrics directory
    metrics_dir = os.path.join(output_dir, "metrics")
    ensure_dir_exists(metrics_dir)
    
    # Initialize timing tracker
    timing_tracker = TimingTracker("idefics", metrics_dir)
    
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
        results = process_batch_idefics(
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
                    category,  # Use period (old/new) as category
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
        if use_gpu and torch.cuda.is_available():
            time.sleep(1)  # Shorter wait between batches on DGX
    
    # Close progress bar
    progress.close()
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("   IDEFICS OCR Processing Summary")
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