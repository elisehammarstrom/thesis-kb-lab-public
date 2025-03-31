#!/bin/bash
# run_ocr_pipeline.sh - Script to run multiple OCR models in sequence
# chmod +x run_ocr_pipeline.sh
# Example usage: ./run_ocr_model_pipeline.sh
# or: ./run_ocr_pipeline.sh --models "kraken,tesseract"
# or: ./run_ocr_pipeline.sh --batch-size 20 --verbose --cpu-only
# NOTE: When running this script, ensure that extracted text boxes are in output/extracted_text_boxes
#       or set RUN_EXTRACT_TEXT_BOXES=true and mount input directories correctly in docker-compose.yml.

set -e
# Create output directories with proper permissions early in the script
mkdir -p ./ocr_outputs
chmod 777 ./ocr_outputs  # Gives everyone write permission to the output directory
ADD_OCR_DATA_TO_JSON=false # Add OCR data to JSON file
RUN_EXTRACT_TEXT_BOXES=false
MODEL_PROCESSING_SCRIPT_DIR="ocr_model_processing"
TESSERACT_SCRIPT="tesseract_ocr_processing.py"
# Set input directory
INPUT_DIR="./output/extracted_text_boxes"
# Specify default models to use for OCR
RUN_TESSERACT=false
RUN_KRAKEN=false
RUN_TROCR=false
RUN_DOCTR=true
RUN_IDEFICS=false

# Specify which models to run on CPU or GPU
TESSERACT_USE_CPU=true   # Tesseract always on CPU
KRAKEN_USE_CPU=true      # Kraken on CPU
TROCR_USE_CPU=false      # TrOCR on GPU
DOCTR_USE_CPU=false      # DocTR on GPU
IDEFICS_USE_CPU=false    # IDEFICS on GPU
# The default model configuration parameters
BATCH_SIZE=10
VERBOSE=false
GPU_ID=0
USE_CPU=true  # Default to CPU for all models - used only for command line flag
DOCKER_COMPOSE_FILE="docker-compose.yml"  # Path to your docker-compose file
# Specified Output directories
BASE_OUTPUT_DIR="./ocr_outputs"
# Define output directories for each model
KRAKEN_END_OF_DIR="ocr_kraken"
TESSERACT_END_OF_DIR="ocr_tesseract_v5.5.0"
IDEFICS_END_OF_DIR="ocr_idefics"
TROCR_END_OF_DIR="ocr_trocr"
DOCTR_END_OF_DIR="ocr_doctr"
KRAKEN_OUTPUT_BASE="$BASE_OUTPUT_DIR/$KRAKEN_END_OF_DIR"
TESSERACT_OUTPUT_BASE="$BASE_OUTPUT_DIR/$TESSERACT_END_OF_DIR"
IDEFICS_OUTPUT_BASE="$BASE_OUTPUT_DIR/$IDEFICS_END_OF_DIR"
TROCR_OUTPUT_BASE="$BASE_OUTPUT_DIR/$TROCR_END_OF_DIR"
DOCTR_OUTPUT_BASE="$BASE_OUTPUT_DIR/$DOCTR_END_OF_DIR"
# Kraken models
RUN_KRAKEN_GERMAN_PRINT=true
RUN_KRAKEN_AUSTRIAN_NEWSPAPERS=true
KRAKEN_GERMAN_END_OF_DIR="german_print_model"
KRAKEN_AUSTRIAN_END_OF_DIR="austrian_newspapers_model"
KRAKEN_OUTPUT_GERMAN_PRINT="$KRAKEN_OUTPUT_BASE/$KRAKEN_GERMAN_END_OF_DIR"
KRAKEN_OUTPUT_AUSTRIAN_NEWSPAPERS="$KRAKEN_OUTPUT_BASE/$KRAKEN_AUSTRIAN_END_OF_DIR"
# DocTR models to run
RUN_DOCTR_DEFAULT=true             # Run the default DocTR model
RUN_DOCTR_PARSEQ_MULTILINGUAL=true # Run the Felix92/doctr-torch-parseq-multilingual-v1 model
DOCTR_DEFAULT_MODEL_DIR="default_model"
DOCTR_PARSEQ_MODEL_DIR="parseq_multilingual"
DOCTR_OUTPUT_DEFAULT="$DOCTR_OUTPUT_BASE/$DOCTR_DEFAULT_MODEL_DIR"
DOCTR_OUTPUT_PARSEQ="$DOCTR_OUTPUT_BASE/$DOCTR_PARSEQ_MODEL_DIR"
# Build Docker image(s)
echo "Building Docker containers..."
if [ "$RUN_EXTRACT_TEXT_BOXES" = true ]; then
  echo "Building Text box container..."
  docker compose -f "$DOCKER_COMPOSE_FILE" build textbox-splitter
fi
if [ "$RUN_TESSERACT" = true ]; then
  echo "Building Tesseract container..."
  docker compose -f "$DOCKER_COMPOSE_FILE" build tesseract
fi
if [ "$RUN_KRAKEN" = true ]; then
  echo "Building Kraken container..."
  docker compose -f "$DOCKER_COMPOSE_FILE" build kraken
fi
if [ "$RUN_TROCR" = true ]; then
  echo "Building TrOCR container..."
  docker compose -f "$DOCKER_COMPOSE_FILE" build trocr
fi
if [ "$RUN_DOCTR" = true ]; then
  echo "Building DocTR container..."
  docker compose -f "$DOCKER_COMPOSE_FILE" build doctr
fi
if [ "$RUN_IDEFICS" = true ]; then
  echo "Building IDEFICS container..."
  docker compose -f "$DOCKER_COMPOSE_FILE" build idefics
fi
echo "Docker build completed"
# Run text box splitter only if enabled
if [ "$RUN_EXTRACT_TEXT_BOXES" = true ]; then
  echo "====================================="
  echo "   Running Text Box Splitting       "
  echo "====================================="
  textbox_start_time=$(date +%s)
  sudo chown -R $(id -u):$(id -g) ./output
  chmod -R 755 ./output
  docker compose -f "$DOCKER_COMPOSE_FILE" run --rm textbox-splitter
  if [ $? -ne 0 ]; then
    echo "Text Box splitting failed!"
    EXIT_CODE=1
  else
    textbox_end_time=$(date +%s)
    textbox_runtime=$((textbox_end_time - textbox_start_time))
    echo "Text Box splitting completed in $((textbox_runtime/3600))h:$((textbox_runtime%3600/60))m:$((textbox_runtime%60))s"
  fi
else
  echo "Skipping text box splitting (RUN_EXTRACT_TEXT_BOXES=$RUN_EXTRACT_TEXT_BOXES)"
fi
# Parse command line arguments used for model processing
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --cpu-only)
      USE_CPU=true
      # Apply to all models if specified globally
      TESSERACT_USE_CPU=true
      KRAKEN_USE_CPU=true
      TROCR_USE_CPU=true
      DOCTR_USE_CPU=true
      IDEFICS_USE_CPU=true
      shift
      ;;
    --output-dir)
      BASE_OUTPUT_DIR="$2"
      # Update individual output directories based on new base directory
      KRAKEN_OUTPUT_BASE="$BASE_OUTPUT_DIR/$KRAKEN_END_OF_DIR"
      TESSERACT_OUTPUT_BASE="$BASE_OUTPUT_DIR/$TESSERACT_END_OF_DIR"
      IDEFICS_OUTPUT_BASE="$BASE_OUTPUT_DIR/$IDEFICS_END_OF_DIR"
      TROCR_OUTPUT_BASE="$BASE_OUTPUT_DIR/$TROCR_END_OF_DIR"
      DOCTR_OUTPUT_BASE="$BASE_OUTPUT_DIR/$DOCTR_END_OF_DIR"
      shift 2
      ;;
    --models)
      IFS=',' read -ra MODELS <<< "$2"
      # Reset all models to false
      RUN_KRAKEN=false
      RUN_TESSERACT=false
      RUN_IDEFICS=false
      RUN_DOCTR=false
      RUN_TROCR=false
      for model in "${MODELS[@]}"; do
        case "$model" in
          "kraken") RUN_KRAKEN=true ;;
          "tesseract") RUN_TESSERACT=true ;;
          "idefics") RUN_IDEFICS=true ;;
          "trocr") RUN_TROCR=true ;;
          "doctr") RUN_DOCTR=true ;;
          *) echo "Unknown model: $model" ;;
        esac
      done
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --batch-size N     Number of images to process in a batch (default: 10)"
      echo "  --verbose, -v      Enable verbose output"
      echo "  --gpu-id N         GPU ID to use (default: 0)"
      echo "  --cpu-only         Force CPU usage for all models even if GPU is available"
      echo "  --output-dir DIR   Base directory for all OCR outputs (default: ./ocr_outputs)"
      echo "  --models LIST      Comma-separated list of models to run (e.g., 'kraken,tesseract')"
      echo "  --help, -h         Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done
# Create output directories
create_output_directories() {
  local base_dir=$1
  
  mkdir -p "$base_dir"
  echo "Created output directory: $base_dir"
  
  mkdir -p "$base_dir/metrics"
  echo "Created metrics directory: $base_dir/metrics"
}
echo "Creating output directories..."
if [ "$RUN_KRAKEN" = true ]; then
  create_output_directories "$KRAKEN_OUTPUT_GERMAN_PRINT"
  create_output_directories "$KRAKEN_OUTPUT_AUSTRIAN_NEWSPAPERS"
fi
if [ "$RUN_TESSERACT" = true ]; then
  create_output_directories "$TESSERACT_OUTPUT_BASE"
fi
if [ "$RUN_TROCR" = true ]; then
  create_output_directories "$TROCR_OUTPUT_BASE"
fi
if [ "$RUN_DOCTR" = true ]; then
  create_output_directories "$DOCTR_OUTPUT_BASE"
  if [ "$RUN_DOCTR_DEFAULT" = true ]; then
    create_output_directories "$DOCTR_OUTPUT_DEFAULT"
  fi
  if [ "$RUN_DOCTR_PARSEQ_MULTILINGUAL" = true ]; then
    create_output_directories "$DOCTR_OUTPUT_PARSEQ"
  fi
fi
if [ "$RUN_IDEFICS" = true ]; then
  create_output_directories "$IDEFICS_OUTPUT_BASE"
fi
# Common arguments (without CPU settings)
COMMON_ARGS=()
if [ "$VERBOSE" = true ]; then
  COMMON_ARGS+=("-v")
fi
COMMON_ARGS+=("--batch-size" "$BATCH_SIZE")
COMMON_ARGS+=("--gpu-id" "$GPU_ID")
echo "==============================================="
echo "    Running OCR Model Processing Pipeline  "
echo "==============================================="
echo "Running with the following configuration:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Verbose: $VERBOSE"
echo "  GPU ID: $GPU_ID"
echo "  Base Output Directory: $BASE_OUTPUT_DIR"
echo "Models to run:"
if [ "$RUN_KRAKEN" = true ]; then
  echo "  - Kraken (Output: $KRAKEN_OUTPUT_BASE, Using CPU: $KRAKEN_USE_CPU)"
fi
if [ "$RUN_TESSERACT" = true ]; then
  echo "  - Tesseract (Output: $TESSERACT_OUTPUT_BASE, Using CPU: $TESSERACT_USE_CPU)"
fi
if [ "$RUN_IDEFICS" = true ]; then
  echo "  - IDEFICS (Output: $IDEFICS_OUTPUT_BASE, Using CPU: $IDEFICS_USE_CPU)"
fi
if [ "$RUN_TROCR" = true ]; then
  echo "  - TrOCR (Output: $TROCR_OUTPUT_BASE, Using CPU: $TROCR_USE_CPU)"
fi
if [ "$RUN_DOCTR" = true ]; then
  echo "  - DocTR (Output: $DOCTR_OUTPUT_BASE, Using CPU: $DOCTR_USE_CPU)"
  if [ "$RUN_DOCTR_DEFAULT" = true ]; then
    echo "    - Default model (Output: $DOCTR_OUTPUT_DEFAULT)"
  fi
  if [ "$RUN_DOCTR_PARSEQ_MULTILINGUAL" = true ]; then
    echo "    - Parseq Multilingual model (Output: $DOCTR_OUTPUT_PARSEQ)"
  fi
fi
# Run the OCR models in sequence
EXIT_CODE=0
start_time=$(date +%s)
echo "Models will run in sequence"
if [ "$RUN_TESSERACT" = true ]; then
  echo "====================================="
  echo "   Running Tesseract OCR Processing "
  echo "====================================="
  tesseract_start_time=$(date +%s)
  TESSERACT_ARGS=("${COMMON_ARGS[@]}")
  TESSERACT_ARGS+=("--input-dir" "$INPUT_DIR")
  if [ "$TESSERACT_USE_CPU" = true ]; then
    TESSERACT_ARGS+=("--cpu-only")
  fi
  
  # Old Tesseract command (commented out)
  # docker compose -f "$DOCKER_COMPOSE_FILE" run --rm \
  # -e INPUT_DIR="/app/input" \
  # tesseract \
  # python /app/ocr_model_processing/tesseract_ocr_processing.py --input-dir "/app/input" --tesseract-config "--oem 3 --psm 3 -l swe"
  
  # New Tesseract command with not documented batch processing in one single model pass 
  docker compose -f "$DOCKER_COMPOSE_FILE" run --rm \
  -e INPUT_DIR="/app/input" \
  tesseract \
  python /app/ocr_model_processing/tesseract_no_documentation_batch_ocr_processing.py --input-dir "/app/input" --tesseract-config "--oem 3 --psm 3 -l swe"

  if [ $? -ne 0 ]; then
    echo "Tesseract OCR processing failed"
    EXIT_CODE=1
  else
    tesseract_end_time=$(date +%s)
    tesseract_runtime=$((tesseract_end_time - tesseract_start_time))
    echo "Tesseract OCR processing completed in $((tesseract_runtime/3600))h:$((tesseract_runtime%3600/60))m:$((tesseract_runtime%60))s"
  fi
fi
if [ "$RUN_KRAKEN" = true ]; then
  echo "====================================="
  echo "   Running Kraken OCR Processing    "
  echo "====================================="
  kraken_start_time=$(date +%s)
  
  # Filter out gpu-id from common args for Kraken
  KRAKEN_ARGS=()
  skip_next=false
  for arg in "${COMMON_ARGS[@]}"; do
    if $skip_next; then
      skip_next=false
      continue
    fi
    
    if [ "$arg" = "--gpu-id" ]; then
      skip_next=true
      continue
    fi
    
    KRAKEN_ARGS+=("$arg")
  done
  
  if [ "$KRAKEN_USE_CPU" = true ]; then
    KRAKEN_ARGS+=("--cpu-only")
  fi
  
  if [ "$RUN_KRAKEN_GERMAN_PRINT" = true ]; then
    echo "Running Kraken with German print model..."
    docker compose -f "$DOCKER_COMPOSE_FILE" run \
      -e KRAKEN_INPUT_DIR="/app/input" \
      -e KRAKEN_OUTPUT_DIR="/app/ocr_outputs/ocr_kraken/german_print_model" \
      -e KRAKEN_MODEL="10.5281/zenodo.10519596" \
      -e KRAKEN_MODEL_NAME="german_print.mlmodel" \
      -e KRAKEN_MODEL_DIR="/root/.config/kraken/" \
      --rm kraken python /app/ocr_model_processing/kraken_ocr_processing.py "${KRAKEN_ARGS[@] -v}"
    
    if [ $? -ne 0 ]; then
      echo "Kraken processing with German print model failed"
      EXIT_CODE=1
    else
      echo "Kraken processing with German print model completed successfully"
    fi
  fi
  
  if [ "$RUN_KRAKEN_AUSTRIAN_NEWSPAPERS" = true ]; then
    echo "Running Kraken with Austrian newspapers model..."
    docker compose -f "$DOCKER_COMPOSE_FILE" run \
      -e KRAKEN_INPUT_DIR="/app/input" \
      -e KRAKEN_OUTPUT_DIR="/app/ocr_outputs/ocr_kraken/austrian_newspapers_model" \
      -e KRAKEN_MODEL="10.5281/zenodo.7933402" \
      -e KRAKEN_MODEL_NAME="austriannewspapers.mlmodel" \
      -e KRAKEN_MODEL_DIR="/root/.config/kraken/" \
      --rm kraken python /app/ocr_model_processing/kraken_ocr_processing.py "${KRAKEN_ARGS[@]}"
    
    if [ $? -ne 0 ]; then
      echo "Kraken processing with Austrian newspapers model failed"
      EXIT_CODE=1
    else
      echo "Kraken processing with Austrian newspapers model completed successfully"
    fi
  fi
  
  kraken_end_time=$(date +%s)
  kraken_runtime=$((kraken_end_time - kraken_start_time))
  echo "Kraken processing completed in $((kraken_runtime/3600))h:$((kraken_runtime%3600/60))m:$((kraken_runtime%60))s"
fi

if [ "$RUN_TROCR" = true ]; then
  echo "====================================="
  echo "   Running TrOCR Processing         "
  echo "====================================="
  trocr_start_time=$(date +%s)
  TROCR_ARGS=("${COMMON_ARGS[@]}")
  if [ "$TROCR_USE_CPU" = true ]; then
    TROCR_ARGS+=("--cpu-only")
  fi
  
  docker compose -f "$DOCKER_COMPOSE_FILE" run \
    -e TROCR_INPUT_DIR="/app/input" \
    -e TROCR_OUTPUT_DIR="/app/ocr_outputs/ocr_trocr" \
    --rm trocr python /app/ocr_model_processing/trocr_ocr_processing.py "${TROCR_ARGS[@]}"
  
  if [ $? -ne 0 ]; then
    echo "TrOCR processing failed"
    EXIT_CODE=1
  else
    trocr_end_time=$(date +%s)
    trocr_runtime=$((trocr_end_time - trocr_start_time))
    echo "TrOCR processing completed in $((trocr_runtime/3600))h:$((trocr_runtime%3600/60))m:$((trocr_runtime%60))s"
  fi
fi


# For Nividia DGX:es 
# if [ "$RUN_DOCTR" = true ]; then
#   echo "====================================="
#   echo "   Running DocTR Batch Processing    "
#   echo "====================================="
#   doctr_start_time=$(date +%s)
#   DOCTR_ARGS=("${COMMON_ARGS[@]}")
  
#   # Use same value for both batch parameters (16 for DGX)
#   BATCH_SIZE=16
#   DOCTR_ARGS+=("--batch-size" "$BATCH_SIZE" "--process-batch-size" "$BATCH_SIZE")
  
#   # Only add --cpu-only if forced; otherwise, we want to use GPU
#   if [ "$DOCTR_USE_CPU" = true ]; then
#     DOCTR_ARGS+=("--cpu-only")
#   fi
  
#   # When running on GPU, add CUDA environment variable so the container sees the proper GPU
#   GPU_ENV="-e CUDA_VISIBLE_DEVICES=${GPU_ID}"
  
#   # Process with default model if enabled
#   if [ "$RUN_DOCTR_DEFAULT" = true ]; then
#     echo "Running DocTR batch processing with default model (batch size: $BATCH_SIZE)..."
#     docker compose -f "$DOCKER_COMPOSE_FILE" run \
#       -e DOCTR_INPUT_DIR="/app/input" \
#       -e DOCTR_OUTPUT_DIR="/app/ocr_outputs/ocr_doctr/default_model" \
#       -e DOCTR_MODEL="default" \
#       -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256" \
#       ${GPU_ENV} \
#       --rm doctr python /app/ocr_model_processing/doctr_ocr_processing.py \
#       --gpu-id ${GPU_ID} -v "${DOCTR_ARGS[@]}"
    
#     if [ $? -ne 0 ]; then
#       echo "DocTR batch processing with default model failed"
#       EXIT_CODE=1
#     else
#       echo "DocTR batch processing with default model completed successfully"
#     fi
#   fi
  
#   # Process with Parseq multilingual model if enabled
#   if [ "$RUN_DOCTR_PARSEQ_MULTILINGUAL" = true ]; then
#     echo "Running DocTR batch processing with Parseq Multilingual model (batch size: $BATCH_SIZE)..."
#     docker compose -f "$DOCKER_COMPOSE_FILE" run \
#       -e DOCTR_INPUT_DIR="/app/input" \
#       -e DOCTR_OUTPUT_DIR="/app/ocr_outputs/ocr_doctr/parseq_multilingual" \
#       -e DOCTR_MODEL="Felix92/doctr-torch-parseq-multilingual-v1" \
#       -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256" \
#       ${GPU_ENV} \
#       --rm doctr python /app/ocr_model_processing/doctr_ocr_processing.py \
#       --gpu-id ${GPU_ID} -v "${DOCTR_ARGS[@]}"
    
#     if [ $? -ne 0 ]; then
#       echo "DocTR batch processing with Parseq Multilingual model failed"
#       EXIT_CODE=1
#     else
#       echo "DocTR batch processing with Parseq Multilingual model completed successfully"
#     fi
#   fi
  
#   doctr_end_time=$(date +%s)
#   doctr_runtime=$((doctr_end_time - doctr_start_time))
#   echo "DocTR batch processing completed in $((doctr_runtime/3600))h:$((doctr_runtime%3600/60))m:$((doctr_runtime%60))s"
# fi

if [ "$RUN_DOCTR" = true ]; then
  echo "====================================="
  echo "   Running DocTR Batch Processing    "
  echo "====================================="
  doctr_start_time=$(date +%s)
  DOCTR_ARGS=("${COMMON_ARGS[@]}")
  
  # Use same value for both batch parameters (4 for M2)
  BATCH_SIZE=4
  DOCTR_ARGS+=("--batch-size" "$BATCH_SIZE" "--process-batch-size" "$BATCH_SIZE")
  
  # Process with default model if enabled
  if [ "$RUN_DOCTR_DEFAULT" = true ]; then
    echo "Running DocTR batch processing with default model (batch size: $BATCH_SIZE)..."
    docker compose -f "$DOCKER_COMPOSE_FILE" run \
      -e DOCTR_INPUT_DIR="/app/input" \
      -e DOCTR_OUTPUT_DIR="/app/ocr_outputs/ocr_doctr/default_model" \
      -e DOCTR_MODEL="default" \
      -e OMP_NUM_THREADS="4" \
      --rm doctr python /app/ocr_model_processing/doctr_ocr_processing.py \
      -v "${DOCTR_ARGS[@]}"
    
    if [ $? -ne 0 ]; then
      echo "DocTR batch processing with default model failed"
      EXIT_CODE=1
    else
      echo "DocTR batch processing with default model completed successfully"
    fi
  fi
  
  # Process with Parseq multilingual model if enabled
  if [ "$RUN_DOCTR_PARSEQ_MULTILINGUAL" = true ]; then
    echo "Running DocTR batch processing with Parseq Multilingual model (batch size: $BATCH_SIZE)..."
    docker compose -f "$DOCKER_COMPOSE_FILE" run \
      -e DOCTR_INPUT_DIR="/app/input" \
      -e DOCTR_OUTPUT_DIR="/app/ocr_outputs/ocr_doctr/parseq_multilingual" \
      -e DOCTR_MODEL="Felix92/doctr-torch-parseq-multilingual-v1" \
      -e OMP_NUM_THREADS="4" \
      --rm doctr python /app/ocr_model_processing/doctr_ocr_processing.py \
      -v "${DOCTR_ARGS[@]}"
    
    if [ $? -ne 0 ]; then
      echo "DocTR batch processing with Parseq Multilingual model failed"
      EXIT_CODE=1
    else
      echo "DocTR batch processing with Parseq Multilingual model completed successfully"
    fi
  fi
  
  doctr_end_time=$(date +%s)
  doctr_runtime=$((doctr_end_time - doctr_start_time))
  echo "DocTR batch processing completed in $((doctr_runtime/3600))h:$((doctr_runtime%3600/60))m:$((doctr_runtime%60))s"
fi

if [ "$RUN_IDEFICS" = true ]; then
  echo "====================================="
  echo "   Running IDEFICS Processing       "
  echo "====================================="
  idefics_start_time=$(date +%s)
  CMD_ARGS=""
  if [ "$VERBOSE" = true ]; then
    CMD_ARGS="$CMD_ARGS -v"
  fi
  CMD_ARGS="$CMD_ARGS --batch-size 1 --gpu-id $GPU_ID"
  if [ "$IDEFICS_USE_CPU" = true ]; then
    CMD_ARGS="$CMD_ARGS --cpu-only"
  fi
  
  docker compose -f "$DOCKER_COMPOSE_FILE" run \
    -e IDEFICS_INPUT_DIR="/app/input" \
    -e IDEFICS_OUTPUT_DIR="/app/output/ocr_idefics" \
    -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
    --entrypoint="" \
    --rm idefics python3 /app/ocr_model_processing/idefics_ocr_processing.py $CMD_ARGS
  
  if [ $? -ne 0 ]; then
    echo "IDEFICS processing failed!"
    EXIT_CODE=1
  else
    idefics_end_time=$(date +%s)
    idefics_runtime=$((idefics_end_time - idefics_start_time))
    echo "IDEFICS processing completed in $((idefics_runtime/3600))h:$((idefics_runtime%3600/60))m:$((idefics_runtime%60))s"
  fi
fi
if [ "$ADD_OCR_DATA_TO_JSON" = true ]; then
  echo "====================================="
  echo "   Adding OCR outputs to JSON file  "
  echo "====================================="
  echo '{}' > ocr_results.json
  docker compose -f "$DOCKER_COMPOSE_FILE" run --rm \
    -v "$(pwd)/ocr_results.json:/app/ocr_results.json" \
    ocr-json-integration
  
  if [ $? -ne 0 ]; then
    echo "JSON integration failed"
    EXIT_CODE=1
  else
    echo "Successfully added OCR outputs to JSON file"
  fi
fi
end_time=$(date +%s)
total_runtime=$((end_time - start_time))
echo "====================================="
echo "   OCR Processing Pipeline Summary   "
echo "====================================="
echo "Total runtime: $((total_runtime/3600))h:$((total_runtime%3600/60))m:$((total_runtime%60))s"
if [ $EXIT_CODE -eq 0 ]; then
  echo "All OCR processes completed successfully!"
else
  echo "One or more OCR processes failed. Check logs for details."
fi
exit $EXIT_CODE