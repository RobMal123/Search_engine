"""
Part 2: Search Functionality
Implement text-to-image search with cosine similarity and result analysis
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import torch
from transformers import CLIPProcessor, CLIPModel
import seaborn as sns
from tqdm import tqdm

class MultimodalSearchEngine:
    def __init__(self, embeddings_dir="embeddings", data_dir="data"):
        """
        Initialize the multimodal search engine
        
        Args:
            embeddings_dir (str): Directory containing saved embeddings
            data_dir (str): Directory containing the dataset
        """
        self.embeddings_dir = embeddings_dir
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model for query processing
        print(f"Loading CLIP model on {self.device}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()
        
        # Load embeddings and metadata
        self.image_embeddings = None
        self.text_embeddings = None
        self.image_text_pairs = None
        self.metadata = None
        self.load_embeddings()
        
        print("Search engine initialized successfully!")
    
    def load_embeddings(self):
        """Load saved embeddings from disk"""
        try:
            with open(os.path.join(self.embeddings_dir, 'image_embeddings.pkl'), 'rb') as f:
                self.image_embeddings = pickle.load(f)
            
            with open(os.path.join(self.embeddings_dir, 'text_embeddings.pkl'), 'rb') as f:
                self.text_embeddings = pickle.load(f)
            
            with open(os.path.join(self.embeddings_dir, 'image_text_pairs.pkl'), 'rb') as f:
                self.image_text_pairs = pickle.load(f)
            
            with open(os.path.join(self.embeddings_dir, 'metadata.json'), 'r') as f:
                self.metadata = json.load(f)
            
            print("Embeddings loaded successfully!")
            print(f"Available images: {self.metadata['num_images']}")
            print(f"Available texts: {self.metadata['num_texts']}")
            
        except FileNotFoundError:
            print("No saved embeddings found. Please run data preparation first.")
            raise
    
    def embed_text_query(self, text_query):
        """
        Generate embedding for a text query
        
        Args:
            text_query (str): Text query to embed
            
        Returns:
            np.ndarray: Text embedding
        """
        with torch.no_grad():
            inputs = self.processor(text=text_query, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            text_embedding = text_features.cpu().numpy().flatten()
        
        return text_embedding
    
    def embed_image_query(self, image_path):
        """
        Generate embedding for an image query
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Image embedding
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_features = self.model.get_image_features(**inputs)
                image_embedding = image_features.cpu().numpy().flatten()
            
            return image_embedding
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def search_text_to_image(self, text_query, top_k=5, similarity_metric='cosine'):
        """
        Search for images using a text query
        
        Args:
            text_query (str): Text query
            top_k (int): Number of top results to return
            similarity_metric (str): 'cosine' or 'euclidean'
            
        Returns:
            list: List of dictionaries containing search results
        """
        print(f"Searching for: '{text_query}'")
        
        # Generate query embedding
        query_embedding = self.embed_text_query(text_query)
        
        # Calculate similarities
        image_ids = list(self.image_embeddings.keys())
        image_embeddings_array = np.array([self.image_embeddings[img_id] for img_id in image_ids])
        
        if similarity_metric == 'cosine':
            similarities = cosine_similarity([query_embedding], image_embeddings_array)[0]
        else:  # euclidean
            distances = euclidean_distances([query_embedding], image_embeddings_array)[0]
            similarities = 1 / (1 + distances)  # Convert distance to similarity
        
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
    
    def search_image_to_text(self, image_path, top_k=5, similarity_metric='cosine'):
        """
        Search for text descriptions using an image query
        
        Args:
            image_path (str): Path to the query image
            top_k (int): Number of top results to return
            similarity_metric (str): 'cosine' or 'euclidean'
            
        Returns:
            list: List of dictionaries containing search results
        """
        print(f"Searching for image: {image_path}")
        
        # Generate query embedding
        query_embedding = self.embed_image_query(image_path)
        if query_embedding is None:
            return []
        
        # Calculate similarities with text embeddings
        text_ids = list(self.text_embeddings.keys())
        text_embeddings_array = np.array([self.text_embeddings[text_id] for text_id in text_ids])
        
        if similarity_metric == 'cosine':
            similarities = cosine_similarity([query_embedding], text_embeddings_array)[0]
        else:  # euclidean
            distances = euclidean_distances([query_embedding], text_embeddings_array)[0]
            similarities = 1 / (1 + distances)  # Convert distance to similarity
        
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
        """Get the file path for an image ID"""
        # Try different possible paths
        possible_paths = [
            os.path.join(self.data_dir, "Images", image_id),
            os.path.join(self.data_dir, image_id),
            image_id
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def display_search_results(self, results, query, search_type='text_to_image'):
        """
        Display search results with images and analysis
        
        Args:
            results (list): Search results
            query (str): Original query
            search_type (str): Type of search performed
        """
        print(f"\n=== Search Results for '{query}' ===")
        print(f"Search type: {search_type}")
        print(f"Found {len(results)} results\n")
        
        # Create subplot
        fig, axes = plt.subplots(1, len(results), figsize=(4*len(results), 4))
        if len(results) == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            ax = axes[i]
            
            # Try to display image
            image_path = result.get('image_path')
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    ax.imshow(img)
                    ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, f"Image not found\n{image_path}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f"Image not found\n{result.get('image_id', 'Unknown')}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            
            # Add title with rank and similarity score
            title = f"Rank {result['rank']}\nScore: {result['similarity_score']:.4f}"
            ax.set_title(title, fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\nDetailed Results:")
        for result in results:
            print(f"\nRank {result['rank']}:")
            print(f"  Similarity Score: {result['similarity_score']:.4f}")
            
            if search_type == 'text_to_image':
                print(f"  Image ID: {result['image_id']}")
                print(f"  Captions:")
                for j, caption in enumerate(result['captions']):
                    print(f"    {j+1}. {caption}")
            else:  # image_to_text
                print(f"  Caption: {result['caption']}")
                print(f"  Image ID: {result['image_id']}")
    
    def analyze_search_results(self, results, query):
        """
        Analyze search results and provide insights
        
        Args:
            results (list): Search results
            query (str): Original query
        """
        print(f"\n=== Analysis of Search Results for '{query}' ===")
        
        if not results:
            print("No results found.")
            return
        
        # Score analysis
        scores = [result['similarity_score'] for result in results]
        print(f"Score Statistics:")
        print(f"  Highest score: {max(scores):.4f}")
        print(f"  Lowest score: {min(scores):.4f}")
        print(f"  Average score: {np.mean(scores):.4f}")
        print(f"  Score range: {max(scores) - min(scores):.4f}")
        
        # Content analysis
        print(f"\nContent Analysis:")
        
        # Extract common words from captions
        all_captions = []
        for result in results:
            if 'captions' in result:
                all_captions.extend(result['captions'])
            elif 'caption' in result:
                all_captions.append(result['caption'])
        
        if all_captions:
            # Simple word frequency analysis
            words = []
            for caption in all_captions:
                words.extend(caption.lower().split())
            
            from collections import Counter
            word_freq = Counter(words)
            common_words = word_freq.most_common(5)
            
            print(f"  Most common words in results: {[word for word, freq in common_words]}")
        
        # Query relevance analysis
        print(f"\nQuery Relevance Analysis:")
        query_words = set(query.lower().split())
        
        relevant_results = 0
        for result in results:
            captions = result.get('captions', [result.get('caption', '')])
            caption_words = set()
            for caption in captions:
                caption_words.update(caption.lower().split())
            
            # Check for word overlap
            overlap = query_words.intersection(caption_words)
            if overlap:
                relevant_results += 1
                print(f"  Rank {result['rank']}: {len(overlap)} word(s) match: {list(overlap)}")
        
        print(f"  Results with word overlap: {relevant_results}/{len(results)}")
        
        # Model behavior insights
        print(f"\nModel Behavior Insights:")
        if max(scores) > 0.8:
            print("  - High similarity scores suggest the model found very relevant matches")
        elif max(scores) > 0.6:
            print("  - Moderate similarity scores suggest reasonable matches")
        else:
            print("  - Low similarity scores suggest the query may be challenging for the model")
        
        if len(set(scores)) < len(scores):
            print("  - Some results have identical scores, indicating similar relevance")
        
        print(f"  - The model successfully mapped text concepts to visual features")
    
    def interactive_search(self):
        """Interactive search interface"""
        print("=== Interactive Multimodal Search ===")
        print("Type 'quit' to exit")
        
        while True:
            print("\n" + "="*50)
            query = input("Enter your text query: ").strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                print("Please enter a valid query.")
                continue
            
            # Perform search
            results = self.search_text_to_image(query, top_k=5)
            
            if results:
                # Display results
                self.display_search_results(results, query)
                
                # Analyze results
                self.analyze_search_results(results, query)
            else:
                print("No results found for your query.")

def main():
    """Main function to run the search engine"""
    print("=== Multimodal Search Engine ===")
    
    try:
        # Initialize search engine
        search_engine = MultimodalSearchEngine()
        
        # Example searches
        example_queries = [
            "a dog running in the park",
            "people playing sports",
            "food on a table",
            "a car on the road",
            "children playing"
        ]
        
        print("\nRunning example searches...")
        for query in example_queries:
            results = search_engine.search_text_to_image(query, top_k=3)
            if results:
                search_engine.display_search_results(results, query)
                search_engine.analyze_search_results(results, query)
                print("\n" + "="*80 + "\n")
        
        # Interactive mode
        print("Starting interactive search mode...")
        search_engine.interactive_search()
        
    except Exception as e:
        print(f"Error initializing search engine: {e}")
        print("Please make sure you have run the data preparation first.")

if __name__ == "__main__":
    main() 