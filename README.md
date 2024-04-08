# AI_research_Tool📈
Sample output:
![Screenshot 2024-02-16 221207](https://github.com/poornachandra24/AI_Research_Tool/assets/85609708/07fb0eb7-b6c1-48a0-88e9-c29da0ac58d8)

## Overview
This a research tool allows users to input  article URLs,  create an aggregrate knowledge base of information based on the URLs, process them, and then query for answers to questions related to the content of those articles. It is built using Python , Langchain , OpenAI API and Streamlit.

## Instructions
1. **Input News Article URLs:**
    - In the sidebar, enter the URLs of news articles you want to analyze. You can input up to three URLs.

2. **Process URLs:**
    - Click the "Process URLs" button to initiate the processing of the provided URLs. This step involves loading the data, splitting the text, creating embeddings, and saving the FAISS index.

3. **Ask Questions:**
    - Once the URLs are processed, you can enter your questions in the text box provided. The application will then use the processed data to find answers to your questions.

4. **View Answers:**
    - The application will display the answers to your questions along with any relevant sources extracted from the processed news articles.

## Implementation Details
- The application uses Streamlit for the user interface.
- It leverages the LangChain library, specifically the `OpenAI` language model (LLM) and various components for text processing and retrieval.
- The `AsyncHtmlLoader` is used to asynchronously load HTML content from the provided URLs.
- After processing the data, embeddings are generated using the `OpenAIEmbeddings` module and saved to a FAISS index and pickled. This prevents recompute units in recreating the VectoR DB for every new query.
- When a question is asked, the application retrieves relevant information from the FAISS index and provides answers using the LLM-based `RetrievalQAWithSourcesChain`.

## Dependencies
- `os`: Operating system interfaces.
- `streamlit`: For creating the web application interface.
- `pickle`: For serializing and deserializing Python objects.
- `time`: For time-related operations.
- `langchain_community`: LangChain library for NLP-related tasks.
- `apikey`: File containing API keys.
- `dotenv`: For loading environment variables from a `.env` file.

## Usage
1. Clone the repository.
    ```
    git clone https://github.com/poornachandra24/AI_Research_Tool.git
    ```
2. Set up the environment by installing the required dependencies.
    ```
    pip install -r requirements.txt
    ```
3. Run the Streamlit application using the command `
    ```
    streamlit run app.py
    ```
4. Input news article URLs, process them, and ask questions related to the content.

Feel free to customize and extend the functionality as needed!

## Demo:
[![AI Researcher video demo link](https://x.com/iris24244242/status/1777302368486359489)]
