import time
import os
import shutil
import requests
import base64
from PyPDF2 import PdfReader
from unstructured.partition.pdf import partition_pdf
from pix2text import Pix2Text
import fitz
from IPython.display import display, Image
from langchain_community.chat_models import ChatSnowflakeCortex
from PIL import Image
import io
from snowflake.snowpark import Session
import snowflake.connector 
from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType
from snowflake.cortex import complete
import uuid
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from io import BytesIO
from IPython.display import display 


# Function to fetch papers from arXiv API
def fetch_arxiv_papers(keyword, max_results=5):
    import arxiv  # Ensure arxiv library is installed
    search = arxiv.Search(
        query=keyword,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "pdf_url": result.pdf_url
        })
    return papers

# Function to clear the downloads folder
def clear_downloads_folder(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove all files and subdirectories
    os.makedirs(output_dir)  # Recreate the folder

# Function to download PDF
def download_pdf(pdf_url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = pdf_url.split("/")[-1] + ".pdf"
    pdf_path = os.path.join(output_dir, pdf_name)

    # Save the PDF locally
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        return pdf_path
    else:
        print(f"Failed to download {pdf_url} with status code {response.status_code}")
        return None

# Function to extract text using fitz and Pix2Text
def extract_text_from_pdf_with_latex(pdf_file):
    math_extractor = Pix2Text()
    pdf_document = fitz.open(pdf_file)
    extracted_text = ""

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        # Extract raw text
        text = page.get_text()
        extracted_text += f"Page {page_num + 1} Text:\n{text}\n"

        # Extract images for potential LaTeX expressions
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            # Extract LaTeX code using Pix2Text
            latex_code = math_extractor(image)
            extracted_text += f"\nMath Expression (Image {img_index + 1}):\n{latex_code}\n"

    pdf_document.close()
    return extracted_text

# Function to check if the image is valid or just lines or small elements
def is_valid_image(img):
    if len(img)>20000:
        return True
    else:
        return False

# Function to process PDF for tables and images using partition_pdf
def process_pdf_with_partition(file_path, output_path):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image","Table"],
        image_output_dir_path=output_path,
        extract_image_block_to_payload=True,
    )

    tables = []
    images = []

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        elif "Image" in str(type(chunk)):
            if(is_valid_image(chunk.metadata.image_base64)):
                images.append(chunk.metadata.image_base64)

    return tables, images

def display_base64_img(base64_code):
    image_data = base64.b64decode(base64_code)
    image = Image.open(BytesIO(image_data))  # Open the image from a file-like object
    display(image) 
