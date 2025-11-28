# Multimodal Search Engine with Flickr-8k Dataset

This project implements a multimodal search engine that can perform both text-to-image and image-to-text search using the Flickr-8k dataset and a pre-trained multimodal model.

## Project Structure

```
project1_Search_engine/
├── data/                           # Dataset directory
├── embeddings/                     # Stored embeddings
├── models/                         # Saved models
├── notebooks/                      # Jupyter notebooks
├── src/                           # Source code
│   ├── data_preparation.py        # Part 1: Data preparation and embedding
│   ├── search_engine.py           # Part 2: Search functionality
│   └── web_interface.py           # Part 3: Web application
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Features

### Part 1: Data Preparation & Embedding

- Load Flickr-8k dataset
- Use CLIP (Contrastive Language-Image Pre-training) model for multimodal embeddings
- Generate and store vector embeddings for images and text
- Efficient data processing pipeline

### Part 2: Search Functionality

- Text-to-image search with cosine similarity
- Top-5 most similar image retrieval
- Analysis of search results
- Interactive search interface

### Part 3: Multimodal & Interface Upgrade

- Image-to-text search capability
- Streamlit web application
- Dual search modes (text-to-image and image-to-text)
- User-friendly interface with project description

## Installation

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the Flickr-8k dataset and place it in the `data/` directory

## Usage

### Running the Data Preparation

```bash
python src/data_preparation.py
```

### Running the Search Engine

```bash
python src/search_engine.py
```

### Running the Web Interface

```bash
streamlit run src/web_interface.py
```

## Technology Stack

- **Model**: CLIP (Contrastive Language-Image Pre-training) from OpenAI
- **Framework**: PyTorch, Transformers
- **Web Interface**: Streamlit
- **Data Processing**: Hugging Face Datasets, Pandas
- **Similarity Search**: Cosine similarity with scikit-learn

## Dataset

The project uses the Flickr-8k dataset, which contains:

- 8,000 images
- 40,000 captions (5 per image)
- Diverse visual content with natural language descriptions

## Model Architecture

CLIP (Contrastive Language-Image Pre-training) is used for this project because:

- It's pre-trained on 400 million image-text pairs
- Can handle both image and text inputs
- Produces aligned embeddings in the same vector space
- Excellent performance on zero-shot tasks
- Efficient for similarity search applications

## License

Educational project for academic course in Machine Learning and Deep Learning.
