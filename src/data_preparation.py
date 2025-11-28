"""
Part 1: Data Preparation & Embedding
Load Flickr-8k dataset and generate embeddings using CLIP model
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

class Flickr8kDataProcessor:
    def __init__(self, data_dir="data", embeddings_dir="embeddings"):
        """
        Initialize the data processor for Flickr-8k dataset
        
        Args:
            data_dir (str): Directory containing the dataset
            embeddings_dir (str): Directory to store embeddings
        """
        self.data_dir = data_dir
        self.embeddings_dir = embeddings_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories if they don't exist
        os.makedirs(embeddings_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Load CLIP model
        print(f"Loading CLIP model on {self.device}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()
        
        print("CLIP model loaded successfully!")
        
    def load_flickr8k_dataset(self):
        """
        Load Flickr-8k dataset from Hugging Face datasets
        If not available, provide instructions for manual download
        """
        try:
            from datasets import load_dataset
            print("Loading Flickr-8k dataset from Hugging Face...")
            dataset = load_dataset("nlphuji/flickr8k")
            return dataset
        except Exception as e:
            print(f"Could not load dataset from Hugging Face: {e}")
            print("\nManual dataset setup required:")
            print("1. Download Flickr-8k dataset from: https://www.kaggle.com/datasets/adityajn105/flickr8k")
            print("2. Extract to the 'data' directory")
            print("3. Ensure the following structure:")
            print("   data/")
            print("   ├── Images/")
            print("   └── captions.txt")
            return None
    
    def process_manual_dataset(self):
        """
        Process manually downloaded Flickr-8k dataset
        """
        images_dir = os.path.join(self.data_dir, "Images")
        captions_file = os.path.join(self.data_dir, "captions.txt")
        
        if not os.path.exists(images_dir) or not os.path.exists(captions_file):
            print("Manual dataset not found. Please download Flickr-8k dataset.")
            return None
        
        # Load captions
        captions_df = pd.read_csv(captions_file)
        
        # Group by image filename
        image_captions = {}
        for _, row in captions_df.iterrows():
            image_name = row['image']
            caption = row['caption']
            if image_name not in image_captions:
                image_captions[image_name] = []
            image_captions[image_name].append(caption)
        
        return image_captions, images_dir
    
    def generate_embeddings(self, dataset=None):
        """
        Generate embeddings for images and text in the dataset
        
        Args:
            dataset: Hugging Face dataset or None for manual dataset
        """
        if dataset is not None:
            return self._generate_embeddings_hf(dataset)
        else:
            return self._generate_embeddings_manual()
    
    def _generate_embeddings_hf(self, dataset):
        """Generate embeddings from Hugging Face dataset"""
        print("Generating embeddings from Hugging Face dataset...")
        
        image_embeddings = {}
        text_embeddings = {}
        image_text_pairs = []
        
        # Process training split
        for item in tqdm(dataset['train'], desc="Processing training data"):
            image = item['image']
            captions = item['captions']
            
            # Generate image embedding
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_features = self.model.get_image_features(**inputs)
                image_embedding = image_features.cpu().numpy().flatten()
            
            image_embeddings[item['image_id']] = image_embedding
            
            # Generate text embeddings for each caption
            for i, caption in enumerate(captions):
                with torch.no_grad():
                    inputs = self.processor(text=caption, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    text_features = self.model.get_text_features(**inputs)
                    text_embedding = text_features.cpu().numpy().flatten()
                
                text_id = f"{item['image_id']}_caption_{i}"
                text_embeddings[text_id] = text_embedding
                image_text_pairs.append({
                    'image_id': item['image_id'],
                    'text_id': text_id,
                    'caption': caption,
                    'image_embedding': image_embedding,
                    'text_embedding': text_embedding
                })
        
        return image_embeddings, text_embeddings, image_text_pairs
    
    def _generate_embeddings_manual(self):
        """Generate embeddings from manually downloaded dataset"""
        print("Generating embeddings from manual dataset...")
        
        result = self.process_manual_dataset()
        if result is None:
            return None, None, None
        
        image_captions, images_dir = result
        
        image_embeddings = {}
        text_embeddings = {}
        image_text_pairs = []
        
        for image_name in tqdm(image_captions.keys(), desc="Processing images"):
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                continue
            
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                
                # Generate image embedding
                with torch.no_grad():
                    inputs = self.processor(images=image, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_features = self.model.get_image_features(**inputs)
                    image_embedding = image_features.cpu().numpy().flatten()
                
                image_embeddings[image_name] = image_embedding
                
                # Generate text embeddings for each caption
                captions = image_captions[image_name]
                for i, caption in enumerate(captions):
                    with torch.no_grad():
                        inputs = self.processor(text=caption, return_tensors="pt", padding=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        text_features = self.model.get_text_features(**inputs)
                        text_embedding = text_features.cpu().numpy().flatten()
                    
                    text_id = f"{image_name}_caption_{i}"
                    text_embeddings[text_id] = text_embedding
                    image_text_pairs.append({
                        'image_id': image_name,
                        'text_id': text_id,
                        'caption': caption,
                        'image_embedding': image_embedding,
                        'text_embedding': text_embedding
                    })
                    
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue
        
        return image_embeddings, text_embeddings, image_text_pairs
    
    def save_embeddings(self, image_embeddings, text_embeddings, image_text_pairs):
        """
        Save embeddings and metadata to disk
        
        Args:
            image_embeddings (dict): Dictionary of image embeddings
            text_embeddings (dict): Dictionary of text embeddings
            image_text_pairs (list): List of image-text pairs with embeddings
        """
        print("Saving embeddings...")
        
        # Save image embeddings
        with open(os.path.join(self.embeddings_dir, 'image_embeddings.pkl'), 'wb') as f:
            pickle.dump(image_embeddings, f)
        
        # Save text embeddings
        with open(os.path.join(self.embeddings_dir, 'text_embeddings.pkl'), 'wb') as f:
            pickle.dump(text_embeddings, f)
        
        # Save image-text pairs
        with open(os.path.join(self.embeddings_dir, 'image_text_pairs.pkl'), 'wb') as f:
            pickle.dump(image_text_pairs, f)
        
        # Save metadata
        metadata = {
            'num_images': len(image_embeddings),
            'num_texts': len(text_embeddings),
            'num_pairs': len(image_text_pairs),
            'embedding_dim': len(next(iter(image_embeddings.values()))) if image_embeddings else 0
        }
        
        with open(os.path.join(self.embeddings_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Embeddings saved successfully!")
        print(f"Images: {metadata['num_images']}")
        print(f"Texts: {metadata['num_texts']}")
        print(f"Pairs: {metadata['num_pairs']}")
        print(f"Embedding dimension: {metadata['embedding_dim']}")
    
    def load_embeddings(self):
        """
        Load saved embeddings from disk
        
        Returns:
            tuple: (image_embeddings, text_embeddings, image_text_pairs, metadata)
        """
        print("Loading saved embeddings...")
        
        try:
            image_embeddings_path = os.path.join(self.embeddings_dir, 'image_embeddings.pkl')
            text_embeddings_path = os.path.join(self.embeddings_dir, 'text_embeddings.pkl')
            pairs_path = os.path.join(self.embeddings_dir, 'image_text_pairs.pkl')
            metadata_path = os.path.join(self.embeddings_dir, 'metadata.json')

            with open(image_embeddings_path, 'rb') as f:
                image_embeddings = pickle.load(f)

            with open(text_embeddings_path, 'rb') as f:
                text_embeddings = pickle.load(f)

            with open(pairs_path, 'rb') as f:
                image_text_pairs = pickle.load(f)

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            print("Embeddings loaded successfully!")
            return image_embeddings, text_embeddings, image_text_pairs, metadata

        except FileNotFoundError:
            print("No saved embeddings found. Please run data preparation first.")
            return None, None, None, None
        except Exception as e:
            print(f"Failed to load embeddings: {e}")
            print("Your embeddings may be corrupted or incomplete. Consider regenerating them by running:")
            print("  python src/data_preparation.py")
            return None, None, None, None

def main():
    """Main function to run data preparation"""
    print("=== Flickr-8k Data Preparation & Embedding Generation ===")
    
    # Initialize processor
    processor = Flickr8kDataProcessor()
    
    # Check if embeddings already exist
    if os.path.exists(os.path.join(processor.embeddings_dir, 'metadata.json')):
        print("Found existing embeddings. Loading...")
        image_embeddings, text_embeddings, image_text_pairs, metadata = processor.load_embeddings()
        if image_embeddings is not None:
            print("Embeddings loaded successfully!")
            return
    
    # Try to load dataset from Hugging Face
    dataset = processor.load_flickr8k_dataset()
    
    # Generate embeddings
    image_embeddings, text_embeddings, image_text_pairs = processor.generate_embeddings(dataset)
    
    if image_embeddings is not None:
        # Save embeddings
        processor.save_embeddings(image_embeddings, text_embeddings, image_text_pairs)
        print("Data preparation completed successfully!")
    else:
        print("Failed to generate embeddings. Please check your dataset setup.")

if __name__ == "__main__":
    main() 