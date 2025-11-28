"""
Part 3: Multimodal & Interface Upgrade
Streamlit web application for multimodal search
"""

import streamlit as st
import os
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CLIPProcessor, CLIPModel
import tempfile
import io
import base64

# Set page config
st.set_page_config(
    page_title="Multimodal Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitSearchEngine:
    def __init__(self, embeddings_dir="embeddings", data_dir="data"):
        """Initialize the search engine for Streamlit"""
        self.embeddings_dir = embeddings_dir
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model
        @st.cache_resource
        def load_clip_model():
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.to(self.device)
            model.eval()
            return model, processor
        
        self.model, self.processor = load_clip_model()
        
        # Load embeddings
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load saved embeddings"""
        try:
            with open(os.path.join(self.embeddings_dir, 'image_embeddings.pkl'), 'rb') as f:
                self.image_embeddings = pickle.load(f)
            
            with open(os.path.join(self.embeddings_dir, 'text_embeddings.pkl'), 'rb') as f:
                self.text_embeddings = pickle.load(f)
            
            with open(os.path.join(self.embeddings_dir, 'image_text_pairs.pkl'), 'rb') as f:
                self.image_text_pairs = pickle.load(f)
            
            with open(os.path.join(self.embeddings_dir, 'metadata.json'), 'r') as f:
                self.metadata = json.load(f)
                
        except FileNotFoundError:
            st.error("No embeddings found. Please run data preparation first.")
            st.stop()
    
    def embed_text_query(self, text_query):
        """Generate embedding for text query"""
        with torch.no_grad():
            inputs = self.processor(text=text_query, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            text_embedding = text_features.cpu().numpy().flatten()
        return text_embedding
    
    def embed_image_query(self, image):
        """Generate embedding for image query"""
        try:
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_features = self.model.get_image_features(**inputs)
                image_embedding = image_features.cpu().numpy().flatten()
            return image_embedding
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    def search_text_to_image(self, text_query, top_k=5):
        """Search for images using text query"""
        query_embedding = self.embed_text_query(text_query)
        
        # Calculate similarities
        image_ids = list(self.image_embeddings.keys())
        image_embeddings_array = np.array([self.image_embeddings[img_id] for img_id in image_ids])
        similarities = cosine_similarity([query_embedding], image_embeddings_array)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            image_id = image_ids[idx]
            similarity_score = similarities[idx]
            
            # Find captions for this image
            captions = []
            for pair in self.image_text_pairs:
                if pair['image_id'] == image_id:
                    captions.append(pair['caption'])
            
            result = {
                'rank': i + 1,
                'image_id': image_id,
                'similarity_score': similarity_score,
                'captions': captions,
                'image_path': self._get_image_path(image_id)
            }
            results.append(result)
        
        return results
    
    def search_image_to_text(self, image, top_k=5):
        """Search for text descriptions using image query"""
        query_embedding = self.embed_image_query(image)
        if query_embedding is None:
            return []
        
        # Calculate similarities with text embeddings
        text_ids = list(self.text_embeddings.keys())
        text_embeddings_array = np.array([self.text_embeddings[text_id] for text_id in text_ids])
        similarities = cosine_similarity([query_embedding], text_embeddings_array)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            text_id = text_ids[idx]
            similarity_score = similarities[idx]
            
            # Find the corresponding caption and image
            caption = None
            image_id = None
            for pair in self.image_text_pairs:
                if pair['text_id'] == text_id:
                    caption = pair['caption']
                    image_id = pair['image_id']
                    break
            
            result = {
                'rank': i + 1,
                'text_id': text_id,
                'caption': caption,
                'image_id': image_id,
                'similarity_score': similarity_score,
                'image_path': self._get_image_path(image_id) if image_id else None
            }
            results.append(result)
        
        return results
    
    def _get_image_path(self, image_id):
        """Get image file path"""
        possible_paths = [
            os.path.join(self.data_dir, "Images", image_id),
            os.path.join(self.data_dir, image_id),
            image_id
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def display_results(self, results, query, search_type):
        """Display search results in Streamlit"""
        if not results:
            st.warning("No results found.")
            return
        
        # Display query
        st.subheader(f"Search Results for: {query}")
        st.write(f"Search type: {search_type}")
        st.write(f"Found {len(results)} results")
        
        # Create columns for results
        cols = st.columns(len(results))
        
        for i, (col, result) in enumerate(zip(cols, results)):
            with col:
                st.write(f"**Rank {result['rank']}**")
                st.write(f"Score: {result['similarity_score']:.4f}")
                
                # Display image if available
                image_path = result.get('image_path')
                if image_path and os.path.exists(image_path):
                    try:
                        img = Image.open(image_path)
                        st.image(img, caption=f"Rank {result['rank']}", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                else:
                    st.write("Image not found")
                
                # Display captions
                if search_type == 'text_to_image':
                    st.write("**Captions:**")
                    for j, caption in enumerate(result['captions']):
                        st.write(f"{j+1}. {caption}")
                else:  # image_to_text
                    st.write(f"**Caption:** {result['caption']}")
        
        # Analysis section
        with st.expander("Analysis"):
            self.analyze_results(results, query)
    
    def analyze_results(self, results, query):
        """Analyze search results"""
        scores = [result['similarity_score'] for result in results]
        
        # Score statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Highest Score", f"{max(scores):.4f}")
        with col2:
            st.metric("Lowest Score", f"{min(scores):.4f}")
        with col3:
            st.metric("Average Score", f"{np.mean(scores):.4f}")
        with col4:
            st.metric("Score Range", f"{max(scores) - min(scores):.4f}")
        
        # Score distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, len(scores) + 1), scores)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Similarity Scores by Rank')
        ax.set_xticks(range(1, len(scores) + 1))
        st.pyplot(fig)
        
        # Word overlap analysis
        query_words = set(query.lower().split())
        relevant_results = 0
        
        st.write("**Word Overlap Analysis:**")
        for result in results:
            captions = result.get('captions', [result.get('caption', '')])
            caption_words = set()
            for caption in captions:
                caption_words.update(caption.lower().split())
            
            overlap = query_words.intersection(caption_words)
            if overlap:
                relevant_results += 1
                st.write(f"Rank {result['rank']}: {len(overlap)} word(s) match: {list(overlap)}")
        
        st.write(f"Results with word overlap: {relevant_results}/{len(results)}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🔍 Multimodal Search Engine")
    st.markdown("---")
    
    # Sidebar with project information
    with st.sidebar:
        st.header("About This Project")
        st.markdown("""
        This is a multimodal search engine built with:
        
        **Technology Stack:**
        - **Model**: CLIP (Contrastive Language-Image Pre-training)
        - **Framework**: PyTorch, Transformers
        - **Web Interface**: Streamlit
        - **Dataset**: Flickr-8k (8,000 images, 40,000 captions)
        
        **Features:**
        - Text-to-Image Search
        - Image-to-Text Search
        - Cosine Similarity Matching
        - Real-time Analysis
        
        **How it works:**
        1. Images and text are embedded into the same vector space
        2. Queries are embedded using the same model
        3. Similarity is calculated using cosine similarity
        4. Top results are returned and analyzed
        """)
        
        st.markdown("---")
        
        # Dataset statistics
        try:
            with open("embeddings/metadata.json", 'r') as f:
                metadata = json.load(f)
            
            st.header("Dataset Statistics")
            st.metric("Images", metadata['num_images'])
            st.metric("Text Descriptions", metadata['num_texts'])
            st.metric("Image-Text Pairs", metadata['num_pairs'])
            st.metric("Embedding Dimension", metadata['embedding_dim'])
        except:
            st.warning("Dataset statistics not available")
    
    # Initialize search engine
    try:
        search_engine = StreamlitSearchEngine()
    except Exception as e:
        st.error(f"Failed to initialize search engine: {e}")
        st.stop()
    
    # Main content
    tab1, tab2 = st.tabs(["Text-to-Image Search", "Image-to-Text Search"])
    
    with tab1:
        st.header("Text-to-Image Search")
        st.write("Enter a text description to find similar images.")
        
        # Text input
        text_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., a dog running in the park, people playing sports, food on a table..."
        )
        
        # Search parameters
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of results:", min_value=1, max_value=10, value=5)
        with col2:
            if st.button("🔍 Search", type="primary"):
                if text_query.strip():
                    with st.spinner("Searching..."):
                        results = search_engine.search_text_to_image(text_query, top_k)
                        search_engine.display_results(results, text_query, "text_to_image")
                else:
                    st.warning("Please enter a search query.")
        
        # Example queries
        st.markdown("---")
        st.subheader("Example Queries")
        example_queries = [
            "a dog running in the park",
            "people playing sports",
            "food on a table",
            "a car on the road",
            "children playing",
            "a cat sitting on a chair",
            "a person cooking in the kitchen",
            "a beautiful sunset over the ocean"
        ]
        
        cols = st.columns(4)
        for i, query in enumerate(example_queries):
            with cols[i % 4]:
                if st.button(query, key=f"example_{i}"):
                    with st.spinner("Searching..."):
                        results = search_engine.search_text_to_image(query, top_k)
                        search_engine.display_results(results, query, "text_to_image")
    
    with tab2:
        st.header("Image-to-Text Search")
        st.write("Upload an image to find similar text descriptions.")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image file:",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Search parameters
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Number of results:", min_value=1, max_value=10, value=5, key="img_search")
            with col2:
                if st.button("🔍 Search", type="primary", key="img_search_btn"):
                    with st.spinner("Searching..."):
                        results = search_engine.search_image_to_text(image, top_k)
                        search_engine.display_results(results, "Uploaded Image", "image_to_text")
        
        # Instructions
        st.markdown("---")
        st.subheader("How to use Image-to-Text Search")
        st.markdown("""
        1. Upload an image using the file uploader above
        2. The system will analyze the image content
        3. It will find the most similar text descriptions from the dataset
        4. Results will show the matching captions with their similarity scores
        5. You can also see the original images that generated those captions
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Built with ❤️ using CLIP, PyTorch, and Streamlit<br>
        Dataset: Flickr-8k | Model: OpenAI CLIP
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 