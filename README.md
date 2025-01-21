Research IQ** was created to address the growing need for interactive, efficient, and in-depth exploration of academic research papers. With the goal of enhancing how researchers interact with academic content, the platform empowers users to search for papers, extract key information such as tables, images, and text, and query research papers using an advanced AI system, It replies not only by text but also by images. Additionally, **Research IQ** includes the capability to process mathematical expressions, symbols, and special characters—such as Greek letters (e.g., psi)—using Optical Character Recognition (OCR), which were not available in traditional document parsing tools.

## What it Does

Research IQ provides a powerful and intuitive platform that enables users to:

- **Search for research papers** based on specific domains and keywords.
- **Fetch and process papers** from arXiv.
- **Extract and query key content** such as text, tables, and images from the papers.
- **Interact with research papers** through a chatbot interface, where users can ask questions and receive detailed responses, not only in text but also enriched with images.

## How We Built It

The project is composed of several key components:

- **Streamlit**: The front-end framework that allows for easy deployment and provides an interactive interface.
- **arXiv API**: Used to fetch research papers based on user-defined keywords and domains.
- **Snowflake**: The database platform for storing and managing research data, including text, tables, and images.
- **Mistral Large 2 (AI model)**: Utilized for generating concise summaries and providing answers to user queries based on extracted content.
- **LangChain**: A tool used to chunk and organize text into smaller, more manageable segments for easier processing.
- **Python Libraries**: Key libraries, such as `uuid`, `json`, and `google-generativeai`, facilitate various operations throughout the platform.
