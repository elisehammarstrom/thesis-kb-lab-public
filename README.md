
# thesis-kb-labb
My thesis repo for KB-labb spring/summer 2025. 

Overleaf doc for authorized users: https://www.overleaf.com/project/678e9a7735606e81ed3f6405

# OCR Pipeline Setup Instructions

## Prerequisites

### Install Docker and Docker Compose

- Ensure Docker and Docker Compose are installed on your system.
- For GPU support, install NVIDIA Docker support (nvidia-docker2).

## Setup Steps

### 1. Create Environment File

Create a `.env` file in the same directory as your `docker-compose.yml`:

```properties
HF_TOKEN=your_huggingface_token_here
HF_CACHE_DIR=~/.cache/huggingface
```

### 2. Get a Hugging Face Token
Go to Hugging Face Tokens
Create a new token with read access.
Add this token to your .env file.


### 3. Extract Text Boxes on first run (default) & save ocr output and processing times to JSON (default)
If you don't have text boxes extracted yet:

Set RUN_EXTRACT_TEXT_BOXES=true in the script.
Ensure your input JPG images are in the proper directory structure, something like this: 
./images_jpg/{period}/{newspaper}/{image}.jpg

If you have extracted text boxes from each JPG, then the extracted textboxes must be in output/extracted_text_boxes for the models to find the images. They place themselves there naturally. 
If you have the images there, you dont need to set RUN_EXTRACT_TEXT_BOXES to true. 

If you don't have the text boxes extracted, then set RUN_EXTRACT_TEXT_BOXES to true and make sure the input directories are mounted correctly in docker-compose.yml for the service textbox-splitter. 
This will be dependend on the structure of your input data.
Here I've the structure for the open data to be within images_jpg/old/AFTONBLADET 1827-07-02/bib22576754_18270702_152900_74_0002.Jpg and images_pdf/old/AFTONBLADET 1827-07-02/bib22576754_18270702_152900_74_0002.Pdf respectively. For the data set we use, we have divided them in to parent subdirectories to be "old" and "new" as seen in the example above. As such, for new data it could also be
images_jpg/**new**/{newspaper_name}/{filename}.Jpg and not only images_jpg/**old**/{newspaper_name}/{filename}.Jpg. Same thing for the PDF files. Before you run your script, I recommend you to move your input data into this structure
```
./input_jpg/
  ├── old/
  │   └── {newspaper_name}/
  │        └── {filename}
  └── new/
     └── {newspaper_name}/
         └── {filename}
```

and 
```
./input_pdf/
  ├── old/
  │   └── {newspaper_name}/
  │        └── {filename}
  └── new/
     └── {newspaper_name}/
         └── {filename}
```
### 4. Make the script executable
``` 
chmod +x run_ocr_model_pipeline.sh
```

### 5. Check the model selection, all models will run by default
Ensure the models you want to run are set to true in the script:
```
RUN_TESSERACT=true/false
RUN_KRAKEN=true/false
RUN_TROCR=true/false
RUN_DOCTR=true/false
RUN_IDEFICS=true/false
```

### If run Kraken, then you must do this step
Kraken models needs to be downloaded and located manually, as it is dependent on which hardware you're on

First, start the kraken container in interactive mode:
```
docker compose run --rm kraken bash
```
Optional, assess which Kraken models exist:
```
kraken list
```

Download the models I've chosen, german print model and austrian newspaper model (trained on Fraktur).
Download german print: 
```
kraken get 10.5281/zenodo.10519596
```

Download austrian newspapers model:
```
kraken get 10.5281/zenodo.7933402
```

Assess where the downloaded models are stored on your computer: 
```
find / -name "*.mlmodel" 2>/dev/null
```
This will show output, like: 
```
/usr/local/lib/python3.8/site-packages/kraken/blla.mlmodel
/usr/local/share/ocrd-resources/kraken/austriannewspapers.mlmodel
/usr/local/share/ocrd-resources/kraken/german_print.mlmodel
```
For me, downloaded models are always in this path: "/usr/local/share/ocrd-resources/kraken/" 
Take the path you see and open the file ocr_model_processing/kraken_ocr_processing.py and edit this line to match your path:
```
kraken_model_dir = '/usr/local/share/ocrd-resources/kraken/' 
```

Now you should be ready to run the two models I've selected from Kraken. 
If you want to, you can disable the models in run_ocr_model_pipeline.sh by setting these to false:
```
RUN_KRAKEN_GERMAN_PRINT=true
RUN_KRAKEN_AUSTRIAN_NEWSPAPERS=true
```

### 6. Run the script
Run with default settings (recommended)
```
./run_ocr_model_pipeline.sh
```

Run specific models
```
./run_ocr_model_pipeline.sh --models "doctr,idefics"
```

Run with verbose output
```
./run_ocr_model_pipeline.sh --verbose
```

Force CPU usage for all models
```
./run_ocr_model_pipeline.sh --cpu-only
```

Specify batch size
```
./run_ocr_model_pipeline.sh --batch-size 5
```

Combine multiple options
```
./run_ocr_model_pipeline.sh --models "trocr,doctr" --verbose
```

### 7 Check output
The output structure will be:
```
./ocr_outputs/
  ├── ocr_tesseract_v5.3.0/
  ├── ocr_kraken/
  │   ├── german_print_model/
  │   └── austrian_newspapers_model/
  ├── ocr_trocr/
  ├── ocr_doctr/
  │   ├── default_model/
  │   └── parseq_multilingual/
  └── ocr_idefics/
```
Where each directory will contain:
- Newspaper subdirectories
- Text files with OCR results, named after the original images
- Metrics directory with performance statistics
