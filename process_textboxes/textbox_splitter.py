#!/usr/bin/env python3
import os
import sys
from PIL import Image
import pdfplumber
import argparse
import re

def process_pdf_file(pdf_path, jpg_path, output_dir):
    """Process a single PDF file and its corresponding JPG"""
    print(f"Processing: {pdf_path}")
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract rectangles from PDF
    rectangles = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]


            for rect in page.rects:
                x0 = float(rect['x0'])
                x1 = float(rect['x1'])
                y0 = max(0, min(page.height - rect['y1'], page.height))
                y1 = max(0, min(page.height - rect['y0'], page.height))

                # Skip rectangles that touch any edge of the page
                if (x0 <= 0 or 
                    x1 >= page.width or 
                    y0 <= 0 or 
                    y1 >= page.height):
                    continue

                rectangles.append({
                    'x0': x0,
                    'y0': y0,
                    'x1': x1,
                    'y1': y1,
                    'width': float(rect['width']),
                    'height': float(rect['height'])
                })
                
        print(f"Found {len(rectangles)} rectangles in {pdf_path}")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return 0, 0
    
    if not rectangles:
        print(f"No rectangles found in {pdf_path}, skipping")
        return 0, 0
    
    # Process JPG
    extracted_count = 0
    try:
        with Image.open(jpg_path) as jpg_image:
            jpg_width, jpg_height = jpg_image.size
            
            # Calculate scaling factors
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[0]
                pdf_width, pdf_height = page.width, page.height
                
                scale_x = jpg_width / pdf_width
                scale_y = jpg_height / pdf_height
            
            # Process every 3rd rectangle
            for i, rect in enumerate(rectangles[::3], 1):
                # Scale PDF coordinates to JPG dimensions
                x0 = int(rect['x0'] * scale_x)
                y0 = int(rect['y0'] * scale_y)
                x1 = int((rect['x0'] + rect['width']) * scale_x)
                y1 = int((rect['y0'] + rect['height']) * scale_y)
                
                # Validate coordinates
                if x0 >= x1 or y0 >= y1 or x0 < 0 or y0 < 0 or x1 > jpg_width or y1 > jpg_height:
                    continue
                
                # Extract and save section
                section = jpg_image.crop((x0, y0, x1, y1))
                output_filename = f'{base_name}_{i:03d}.jpg'
                output_path = os.path.join(output_dir, output_filename)
                section.save(output_path, quality=95)
                extracted_count += 1
                
        print(f"Extracted {extracted_count} JPG sections from {pdf_path}")
        return 1, extracted_count  # Return pair processed and number of extracted images
    except Exception as e:
        print(f"Error processing JPG {jpg_path}: {e}")
        return 0, 0

def find_and_process_files():
    """Find all PDF files and their corresponding JPGs across the directory structure"""
    # Base directories
    input_pdf_base = os.environ.get('INPUT_PDF_DIR', '/app/images_pdf')
    input_jpg_base = os.environ.get('INPUT_JPG_DIR', '/app/images_jpg')
    output_base = os.environ.get('OUTPUT_DIR', '/app/output')
    
    # Counters
    total_pdf_files = 0
    total_jpg_files = 0 
    total_processed_pairs = 0
    total_extracted_sections = 0
    period_stats = {}
    
    # Find all period directories
    periods = []
    for item in os.listdir(input_pdf_base):
        period_dir = os.path.join(input_pdf_base, item)
        if os.path.isdir(period_dir):
            periods.append(item)
    
    print(f"Found period directories: {periods}")
    
    # Debug directory structure
    for period in periods:
        pdf_period_dir = os.path.join(input_pdf_base, period)
        print(f"\nDEBUGGING: Period directory structure for {period}:")
        print(f"PDF period dir: {pdf_period_dir}")
        
        # Print the directory structure
        print(f"Directory structure for {period}:")
        for root, dirs, files in os.walk(pdf_period_dir):
            rel_path = os.path.relpath(root, pdf_period_dir)
            print(f"  Root: {root}")
            print(f"  Relative path: {rel_path}")
            print(f"  Subdirectories: {dirs}")
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            print(f"  PDF files: {len(pdf_files)}")
            if pdf_files and len(pdf_files) > 0:
                print(f"    Sample file: {pdf_files[0]}")
            print("  ---")
    
    # Process each period directory
    for period in periods:
        pdf_period_dir = os.path.join(input_pdf_base, period)
        jpg_period_dir = os.path.join(input_jpg_base, period)
        output_period_dir = os.path.join(output_base, "extracted_text_boxes", period)
        
        period_pdfs = 0
        period_jpgs = 0
        period_processed = 0
        period_extracted = 0
        
        print(f"\n{'='*60}\nProcessing period: {period}\n{'='*60}")
        
        # Walk through all subdirectories in the period directory
        for root, dirs, files in os.walk(pdf_period_dir):
            # Get the relative path from the period directory
            rel_path = os.path.relpath(root, pdf_period_dir)
            if rel_path == '.':
                rel_path = ''
            
            # Construct corresponding jpg directory path
            jpg_dir = os.path.join(jpg_period_dir, rel_path)
            
            # Construct output directory path maintaining the structure
            curr_output_dir = os.path.join(output_period_dir, rel_path)
            
            # Process PDF files in this directory
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            period_pdfs += len(pdf_files)
            
            if pdf_files:
                print(f"Found {len(pdf_files)} PDF files in {root}")
                
                # Make sure output directory exists
                os.makedirs(curr_output_dir, exist_ok=True)
                
                for pdf_file in pdf_files:
                    pdf_path = os.path.join(root, pdf_file)
                    
                    # Construct JPG path directly by replacing .pdf with .jpg
                    jpg_path = pdf_path.replace('/images_pdf/', '/images_jpg/').replace('.pdf', '.jpg')
                    
                    # Check if JPG exists
                    if os.path.exists(jpg_path):
                        print(f"Found matching JPG: {jpg_path}")
                        period_jpgs += 1
                        
                        # Process the PDF-JPG pair
                        pair_processed, images_extracted = process_pdf_file(pdf_path, jpg_path, curr_output_dir)
                        period_processed += pair_processed
                        period_extracted += images_extracted
                        total_processed_pairs += pair_processed
                        total_extracted_sections += images_extracted
                    else:
                        print(f"No corresponding JPG found for {pdf_path}")
                        print(f"Tried: {jpg_path}")
        
        # Store period statistics - FIXED INDENTATION
        period_stats[period] = {
            'pdf_files': period_pdfs,
            'jpg_files': period_jpgs,
            'processed_pairs': period_processed,
            'extracted_sections': period_extracted
        }
        
        total_pdf_files += period_pdfs
        total_jpg_files += period_jpgs
    
    # Print summary - FIXED INDENTATION
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    for period, stats in period_stats.items():
        print(f"Period {period}:")
        print(f"  - PDF files found: {stats['pdf_files']}")
        print(f"  - Matching JPG files found: {stats['jpg_files']}")
        print(f"  - Successfully processed pairs: {stats['processed_pairs']}")
        print(f"  - Total JPG sections extracted: {stats['extracted_sections']}")
        
    print("\nTOTAL STATISTICS:")
    print(f"  - Total PDF files found: {total_pdf_files}")
    print(f"  - Total matching JPG files found: {total_jpg_files}")
    print(f"  - Total successfully processed PDF-JPG pairs: {total_processed_pairs}")
    print(f"  - Total JPG sections (textboxes) extracted: {total_extracted_sections}")
    
    return total_processed_pairs, total_extracted_sections

def main():
    if len(sys.argv) > 1:
        # Command-line arguments mode
        parser = argparse.ArgumentParser(description="Process newspaper JPGs using PDF coordinates")
        parser.add_argument("--input_pdf", required=True, help="PDF file path")
        parser.add_argument("--input_jpg", required=True, help="JPG file path")
        parser.add_argument("--output", required=True, help="Output directory")
        
        args = parser.parse_args()
        process_pdf_file(args.input_pdf, args.input_jpg, args.output)
    else:
        # Auto-discovery mode
        find_and_process_files()

if __name__ == "__main__":
    main()