# AI-Powered Chatbot for Dynamic Question-Answering Based on Web Content

An advanced chatbot that answers user queries by dynamically extracting and processing content from live websites and news articles. It leverages web scraping techniques and state-of-the-art AI models like **BERT embeddings**, **FAISS**, and **Llama LLM with GroQ** to provide fast, accurate, and contextually relevant responses.

## Features

- **URL-based Chatbot**: Answers queries by scraping live content from websites and news articles.
- **Efficient Semantic Search**: Uses **BERT embeddings** and **FAISS** to perform fast and accurate semantic search for relevant content.
- **Contextual Answer Generation**: Integrates **Llama LLM with GroQ** to generate answers that are contextually relevant and coherent, even for complex queries.
- **Real-Time Content Retrieval**: Retrieves live content from websites, ensuring up-to-date and relevant responses.
  
## Technologies Used

- **Web Scraping**: BeautifulSoup, Requests, Scrapy (for scraping live web content)
- **Semantic Search**: BERT embeddings, FAISS (for efficient similarity search)
- **Language Model**: Llama LLM, GroQ (for context-aware answer generation)
- **Python Libraries**: pandas, numpy, transformers, and more
- **Deployment**: Flask/FastAPI (for API development)

## Installation

Follow these steps to set up and run the project locally:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AI-Powered-Chatbot.git
cd AI-Powered-Chatbot
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

pip install -r requirements.txt

WEB_SCRAPING_API_KEY=your_api_key_here
BERT_MODEL=path_to_pretrained_model
