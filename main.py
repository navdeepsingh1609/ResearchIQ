import toml
import time
import os
import shutil
import requests
import streamlit as st
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
from pdf2image import convert_from_path
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

# Helper function to convert a PIL image (thumbnail) to base64
def thumbnail_to_base64(image):
    from io import BytesIO
    import base64

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

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

# Helper function to generate thumbnails
def generate_thumbnail(pdf_url):
    try:
        # Download the PDF
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        pdf_content = BytesIO(response.content)
        
        # Create a thumbnail using PyMuPDF
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        pix = doc[0].get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Save to BytesIO
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        return buffered
    except Exception as e:
        print(f"Error generating thumbnail for {pdf_url}: {e}")
        return None


def display_base64_img(base64_code):
    image_data = base64.b64decode(base64_code)
    image = Image.open(BytesIO(image_data))  # Open the image from a file-like object
    display(image) 

def sendToMistral(prompt):
    # session = connect_to_snowflake()
    output = complete(
    "mistral-large2",
    prompt,
    session=session)  
    return output

# Chunk text using RecursiveCharacterTextSplitter
def chunk_text_with_langchain(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

# Snowflake Table Creation
def create_snowflake_tables(cursor):
    # Papers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Papers (
            paper_id VARCHAR PRIMARY KEY,
            title VARCHAR,
            domain VARCHAR,
            keyword VARCHAR
        );
    """)

    # TextChunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TextChunks (
            id INTEGER AUTOINCREMENT PRIMARY KEY,
            paper_id VARCHAR REFERENCES Papers(paper_id),
            chunk_index INTEGER,
            chunk_text TEXT
        );
    """)

    # TextSummary table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TextSummary (
            id INTEGER AUTOINCREMENT PRIMARY KEY,
            paper_id VARCHAR REFERENCES Papers(paper_id),
            summary_index INTEGER,
            summary_text TEXT
        );
    """)

    # TablesContent table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TablesContent (
            id INTEGER AUTOINCREMENT PRIMARY KEY,
            paper_id VARCHAR REFERENCES Papers(paper_id),
            table_index INTEGER,
            table_content TEXT
        );
    """)

    # TableSummary table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TableSummary (
            id INTEGER AUTOINCREMENT PRIMARY KEY,
            paper_id VARCHAR REFERENCES Papers(paper_id),
            summary_index INTEGER,
            summary_text TEXT
        );
    """)

    # Images table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Images (
            id INTEGER AUTOINCREMENT PRIMARY KEY,
            paper_id VARCHAR REFERENCES Papers(paper_id),
            image_index INTEGER,
            image_metadata TEXT
        );
    """)

    # ImageSummary table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ImageSummary (
            id INTEGER AUTOINCREMENT PRIMARY KEY,
            paper_id VARCHAR REFERENCES Papers(paper_id),
            summary_index INTEGER,
            summary_text TEXT
        );
    """)

# Insert Data into Snowflake
def insert_data_into_snowflake(cursor, extracted_data, domain, keyword):
    for paper_id, content in extracted_data.items():
        # Check if the paper already exists
        cursor.execute("""
            SELECT COUNT(*) FROM Papers WHERE title = %s AND domain = %s AND keyword = %s
        """, (content["title"], domain, keyword))
        exists = cursor.fetchone()[0]

        if exists:
            print(f"Skipping existing paper: {content['title']}")
            continue

        # Insert into Papers table
        cursor.execute("""
            INSERT INTO Papers (paper_id, title, domain, keyword)
            VALUES (%s, %s, %s, %s)
        """, (paper_id, content["title"], domain, keyword))

        # Insert into TextChunks
        for i, chunk in enumerate(content["text_chunks"]):
            cursor.execute("""
                INSERT INTO TextChunks (paper_id, chunk_index, chunk_text)
                VALUES (%s, %s, %s)
            """, (paper_id, i, chunk))

        # Insert into TextSummary
        for i, summary in enumerate(content["textSummary"]):
            cursor.execute("""
                INSERT INTO TextSummary (paper_id, summary_index, summary_text)
                VALUES (%s, %s, %s)
            """, (paper_id, i, summary))

        # Insert into TablesContent
        for i, table in enumerate(content["tables"]):
            cursor.execute("""
                INSERT INTO TablesContent (paper_id, table_index, table_content)
                VALUES (%s, %s, %s)
            """, (paper_id, i, table))

        # Insert into TableSummary
        for i, summary in enumerate(content["tableSummary"]):
            cursor.execute("""
                INSERT INTO TableSummary (paper_id, summary_index, summary_text)
                VALUES (%s, %s, %s)
            """, (paper_id, i, summary))

        # Insert into Images
        for i, image in enumerate(content["images"]):
            cursor.execute("""
                INSERT INTO Images (paper_id, image_index, image_metadata)
                VALUES (%s, %s, %s)
            """, (paper_id, i, image))

        # Insert into ImageSummary
        for i, summary in enumerate(content["imageSummary"]):
            cursor.execute("""
                INSERT INTO ImageSummary (paper_id, summary_index, summary_text)
                VALUES (%s, %s, %s)
            """, (paper_id, i, summary))

        print(f"Inserted data for paper: {content['title']}")

results_image=None

import streamlit as st
import uuid

# Assume the following functions are defined and available in your codebase
# - fetch_arxiv_papers: Fetches papers based on a keyword
# - download_pdf: Downloads a PDF from a URL
# - extract_text_from_pdf_with_latex: Extracts text with LaTeX from a PDF
# - chunk_text_with_langchain: Chunks text using LangChain
# - process_pdf_with_partition: Processes PDFs to extract tables and images
# - display_base64_img: Converts base64 image data to a format Streamlit can display
# - sendToMistral: Sends prompts to the Mistral model for response
# - create_snowflake_tables: Creates necessary Snowflake tables
# - insert_data_into_snowflake: Inserts extracted data into Snowflake

# Helper function to display base64 image data
def display_base64_img(img_data):
    return f"data:image/png;base64,{img_data}"

# Step 0: Load the environment variables and initialize the Snowflake Connection
# load_dotenv()
# snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
# snowflake_user = os.getenv("SNOWFLAKE_USER")
# snowflake_password = os.getenv("SNOWFLAKE_PASSWORD")
# snowflake_role = os.getenv("SNOWFLAKE_ROLE")
# snowflake_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
# snowflake_database = os.getenv("SNOWFLAKE_DATABASE")
# snowflake_schema = os.getenv("SNOWFLAKE_SCHEMA")
# snowflake_username=os.getenv("SNOWFLAKE_USERNAME")

import toml  # Use `toml` for older Python versions

# Load secrets from secrets.toml
with open("secrets.toml", "r") as f:
    secrets = toml.load(f)

# Retrieve Snowflake configuration values
snowflake_account = secrets.get("SNOWFLAKE_ACCOUNT")
snowflake_user = secrets.get("SNOWFLAKE_USER")
snowflake_password = secrets.get("SNOWFLAKE_PASSWORD")
snowflake_role = secrets.get("SNOWFLAKE_ROLE")
snowflake_warehouse = secrets.get("SNOWFLAKE_WAREHOUSE")
snowflake_database = secrets.get("SNOWFLAKE_DATABASE")
snowflake_schema = secrets.get("SNOWFLAKE_SCHEMA")
snowflake_username = secrets.get("SNOWFLAKE_USERNAME")


# Snowflake configuration
snowflake_config = {
    "account": snowflake_account,
    "user": snowflake_user,
    "password": snowflake_password,
    "role": snowflake_role,
    "warehouse": snowflake_warehouse,
    "database": snowflake_database,
    "schema": snowflake_schema
}

session = Session.builder.configs(snowflake_config).create()

# Snowflake connection details
conn = snowflake.connector.connect(
    user=snowflake_user,
    password=snowflake_password,
    account=snowflake_account,
    role=snowflake_role,
    database=snowflake_database,
    warehouse=snowflake_warehouse,
    schema=snowflake_schema
)

cursor = conn.cursor()

# Initialize the ChatSnowflakeCortex instance
chat = ChatSnowflakeCortex(
    # Change the default cortex model and function
    model="mistral-large2",
    cortex_function="complete",

    # Change the default generation parameters
    temperature=0,
    max_tokens=10,
    top_p=0.95,

    # Specify your Snowflake Credentials
    account=snowflake_account,
    username=snowflake_username,
    password=snowflake_password,
    database=snowflake_database,
    schema=snowflake_schema,
    role=snowflake_role,
    warehouse=snowflake_warehouse
)

def decode_base64_image(base64_data):
    """
    Decode a Base64 string into an image format that Streamlit can display.

    Args:
        base64_data (str): Base64-encoded string of an image.

    Returns:
        BytesIO: Decoded image data.
    """
    from io import BytesIO
    image_data = base64.b64decode(base64_data)
    return BytesIO(image_data)

def process_query(user_query,domain,keyword,result_limit):
    """
    Process the user query by fetching relevant data and generating a response.
    """

    # Step 7: Using Cortex Vector Search for Semantic Textual Search and Image-Summary Search

    # Query for Semantic Textual Search
    search_query_textual = f"""
    SELECT PARSE_JSON(
    SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
        'RESEARCH_PAPERS_DB.PUBLIC.TEXTSERVICE', 
        '{{ 
            "query": "{user_query}",
            "columns": ["content_text", "content_summary", "content_table", "content_table_summary"],
            "filters": {{
            "domain": "{domain}",
            "keyword": "{keyword}"
            }},
            "limit": {result_limit}
        }}'
    )
    )['results'] AS results;
    """
    # Query for Image-Based Search
    search_query_image = f"""
    SELECT PARSE_JSON(
    SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
        'RESEARCH_PAPERS_DB.PUBLIC.IMAGESERVICE', 
        '{{ 
            "query": "{user_query}",
            "columns": ["image_metadata", "image_summary"],
            "filters": {{
            "keyword": "{keyword}"
            }},
            "limit": {2}
        }}'
    )
    )['results'] AS results;
    """

    # Execute the query
    results_image = session.sql(search_query_image).collect()
    results_textual = session.sql(search_query_textual).collect()

    context_textual=""
    for row in results_textual:
        context_textual+=f"{json.dumps(row['RESULTS'], indent=2)}"
        print(context_textual)
    prompt = f"Context: {context_textual}\nUser Query: {user_query}\nAnswer:"
    answer = sendToMistral(prompt)

    imageMapping=json.loads(results_image[0].RESULTS[::])

    # Simulate image generation (if applicable)
    images = []
    for img in imageMapping[:2]:
        images.append(img["image_metadata"])

    return answer, images

def clear_chat_history():
    """
    Clear the chat history in session state.
    """
    st.session_state.chat_history = []

def add_message(role, content, images=None):
    """
    Add a message to the chat history. Include role, content, and optional images.
    """
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "images": images or []
    })

# Main application function
def main():
    st.set_page_config(page_title="Research IQ", layout="wide")
    st.title("📚 Research IQ - A Research Paper Explorer")

    if "pdf_list" not in st.session_state:
        st.session_state.pdf_list = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "search_triggered" not in st.session_state:
        st.session_state.search_triggered = False


    # UI 1: Sidebar for Domain and Keyword Input
    st.sidebar.header("Search Parameters")
    domain_options = ["Physics", "Computer Science", "Mathematics", "Quantitative Biology","Quantitative Finance","Statistics","Economics","Electrical Engineering"]  # Predefined domains
    domain = st.sidebar.selectbox("Select Domain", domain_options)
    keyword = st.sidebar.text_input("Enter Keyword", placeholder="e.g., NL2SQL or electron")

    if st.sidebar.button("Search Papers"):
        st.session_state.search_triggered = True
        if keyword:
            with st.spinner("Fetching and processing papers..."):

                max_results = 3
                output_dir = "downloads"
                
                # Step 1: Fetch papers from arXiv
                papers = fetch_arxiv_papers(keyword, max_results)

                # Step 2: Clear downloads folder
                clear_downloads_folder(output_dir)
                
                # Step 3: Check if papers already exist in the database
                query = f"""
                SELECT COUNT(*) AS paper_count
                FROM papers
                WHERE keyword = '{keyword}'
                """
                cursor.execute(query)
                paper_count = cursor.fetchone()[0]

                # Process each paper and display its information
                extracted_data = {}

                # Sidebar section for displaying papers
                if papers:
                    for paper in papers:
                        if paper not in st.session_state.pdf_list:
                            st.session_state.pdf_list.append(paper)

                if "pdf_list" in st.session_state and st.session_state.pdf_list:
                    st.sidebar.subheader(f"Research Work Found for '{keyword}' in {domain}")
                    for paper in st.session_state.pdf_list:
                        pdf_url = paper["pdf_url"]
                        pdf_title = paper["title"]
                        pdf_authors = paper["authors"]

                        # Generate thumbnail and display
                        thumbnail = generate_thumbnail(pdf_url)  # Ensure this function generates a valid base64 image
                        if thumbnail:
                            st.sidebar.image(thumbnail, use_container_width=False, width=150)
                        else:
                            st.sidebar.error(f"Error displaying thumbnail for {pdf_url}")

                        st.sidebar.markdown(f"[{pdf_title}]({pdf_url})", unsafe_allow_html=True)
                        st.sidebar.write(f"**Authors**: {', '.join(pdf_authors)}")
                if paper_count >= max_results: # Skip processing if papers already exist
                    print(f"Found {paper_count} papers for keyword '{keyword}' in the database.")
                    for paper in papers:
                        print(f"Saving: {paper['title']}")
                        pdf_path = download_pdf(paper["pdf_url"], output_dir)
                        if pdf_path:
                            print(f"Saved PDF: {pdf_path}")
                        else:
                            print(f"Failed to save {paper['title']}")
                else:
                    print(f"Fetching papers from arXiv for keyword '{keyword}'...")
                                        
                    # Step 3: Process papers to extract images, texts, and tables along with latex text
                    for paper in papers:
                        print(f"Processing: {paper['title']}")
                        pdf_path = download_pdf(paper["pdf_url"], output_dir)

                        if pdf_path:
                            print(f"Saved PDF: {pdf_path}")

                            # Generate a unique ID for the paper
                            paper_id = str(uuid.uuid4())

                            # Initialize storage for this paper
                            extracted_data[paper_id] = {
                                "title": paper["title"],
                                "text_chunks": [],
                                "tables": [],
                                "images": [],
                            }

                            # Extract text and chunk it
                            text = extract_text_from_pdf_with_latex(pdf_path)
                            text_chunks = chunk_text_with_langchain(text)
                            extracted_data[paper_id]["text_chunks"].extend(text_chunks)

                            # Extract tables and images
                            tables, images = process_pdf_with_partition(pdf_path, output_dir)
                            extracted_data[paper_id]["tables"].extend([table.text for table in tables])
                            extracted_data[paper_id]["images"].extend(images)
                            # for image in images:
                            #     print(image)
                            #     print(len(image))
                            #     display_base64_img(image)
                            print(f"Data collected for paper: {paper['title']}")
                        else:
                            print(f"Failed to process {paper['title']}")
                            
                    # Printing extractedData
                    for paper_id, content in extracted_data.items():
                        print(f"Inserting data for paper: {paper_id} -> {content['title']}")
                        print(f"Inserting text chunks...:/n {content['text_chunks'][:50]}")
                        print(f"Inserting tables...:/n {content['tables'][:50]}")
                        print(f"Inserting images...:/n {content['images'][:50]}")
                        if 'tableSummary' in content:
                            print(f"Inserting table summaries...:/n {content['tableSummary']}")
                        if 'textSummary' in content:
                            print(f"Inserting text summaries...:/n {content['textSummary']}")
                        if 'imageSummary' in content:
                            print(f"Inserting image summaries...:/n {content['imageSummary']}")

                    # Step 4: Generate summaries for tables, text, and images 
                    
                    # Load model for generating image summary
                    import google.generativeai as genai
                    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
                    gemini_api_key=os.getenv("GEMINI_API_KEY") 
                    
                    # Function to generate summary of tables, text and images
                    for paper_id, content in extracted_data.items():
                        tableSummaries=[]
                        print(f"processing {paper_id}")
                        for table in content["tables"]:
                            prompt = f"""Your task it to summarize tables for retrieval. \
                            These summaries will be embedded and used to retrieve the raw table elements. \
                            Give a concise summary of the table that is well optimized for retrieval. {table} """
                            tableSummaries.append(sendToMistral(prompt)) #using mistral to generate table summary
                        extracted_data[paper_id]["tableSummary"]=tableSummaries
                
                        textSummaries=[]
                        for text in content["text_chunks"][:3]:
                            prompt = f"""Your task it to summarize text for retrieval. \
                            These summaries will be embedded and used to retrieve the raw text elements. \
                            Give a concise summary of the table or text that is well optimized for {text} """
                            textSummaries.append(sendToMistral(prompt))
                        extracted_data[paper_id]["textSummary"]=textSummaries
                        
                        imageSummaries=[]
                        prompt = """You are an assistant tasked with summarizing images for retrieval.
                            These summaries will be embedded and used to retrieve the raw image.
                            Give a concise summary of the image that is well optimized for retrieval."""
                        for encoded_image in content["images"]:  
                            response = model.generate_content(
                                [
                                    {
                                        'mime_type': 'image/png',
                                        'data': encoded_image
                                    },
                                    prompt
                                ]
                            )
                            print(response.text)
                            imageSummaries.append(response.text)
                            time.sleep(3) 
                        extracted_data[paper_id]["imageSummary"]=imageSummaries

                    # Step 5: Insert extracted data into Snowflake
                    create_snowflake_tables(cursor)
                    insert_data_into_snowflake(cursor, extracted_data, domain=domain, keyword=keyword)        
                    st.session_state.pdf_list = papers
    
    # """
    # Handle the chatbot interface and interactions.
    # """
    # st.subheader("💬 Chat with Research Papers")
    # for message in st.session_state.chat_history:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # Step 6: Input the user queries and search for papers
    user_query = st.chat_input("Ask something about the research work:")
    result_limit = 5
    if user_query:
        # Add user message to chat history
        add_message("user", user_query)

        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Thinking..."):
                response, images = process_query(user_query,domain,keyword,result_limit)
            response_placeholder.markdown(response)
            for img_base64 in images:
                decoded_image = decode_base64_image(img_base64)
                st.image(decoded_image, use_container_width=False, width=400)
        
        # Add assistant response to chat history, including images
        add_message("assistant", response, images)

if __name__ == "__main__":
    main()
