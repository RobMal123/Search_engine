#!/usr/bin/env python3
"""
Setup script for Flickr-8k dataset
This script helps users download and set up the Flickr-8k dataset
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def setup_flickr8k():
    """Set up Flickr-8k dataset"""
    print("=== Flickr-8k Dataset Setup ===")
    print()
    
    # Create data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("📁 Dataset Setup Options:")
    print("1. Download from Hugging Face (Recommended)")
    print("2. Manual download instructions")
    print("3. Use existing dataset")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🔄 Setting up Hugging Face dataset...")
        try:
            from datasets import load_dataset
            print("Loading Flickr-8k dataset from Hugging Face...")
            dataset = load_dataset("nlphuji/flickr8k")
            print("✅ Dataset loaded successfully!")
            print(f"Available splits: {list(dataset.keys())}")
            print(f"Training samples: {len(dataset['train'])}")
            print(f"Validation samples: {len(dataset['validation'])}")
            print(f"Test samples: {len(dataset['test'])}")
            return True
        except Exception as e:
            print(f"❌ Error loading from Hugging Face: {e}")
            print("Falling back to manual setup...")
            return setup_manual()
    
    elif choice == "2":
        return setup_manual()
    
    elif choice == "3":
        return check_existing_dataset()
    
    else:
        print("❌ Invalid choice. Please run the script again.")
        return False

def setup_manual():
    """Provide manual download instructions"""
    print("\n📋 Manual Dataset Setup Instructions:")
    print("=" * 50)
    print()
    print("1. Download Flickr-8k dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/adityajn105/flickr8k")
    print()
    print("2. Extract the downloaded file to the 'data' directory")
    print()
    print("3. Ensure the following structure:")
    print("   data/")
    print("   ├── Images/")
    print("   │   ├── 1000268201_693b08cb0e.jpg")
    print("   │   ├── 1000268201_693b08cb0e.jpg")
    print("   │   └── ... (8000 images)")
    print("   └── captions.txt")
    print()
    print("4. The captions.txt file should have columns: image, caption")
    print()
    
    # Check if dataset is already present
    if check_existing_dataset():
        return True
    
    print("❌ Dataset not found in expected location.")
    print("Please follow the instructions above and run this script again.")
    return False

def check_existing_dataset():
    """Check if dataset is already set up"""
    images_dir = os.path.join("data", "Images")
    captions_file = os.path.join("data", "captions.txt")
    
    if os.path.exists(images_dir) and os.path.exists(captions_file):
        # Count images
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print("✅ Existing dataset found!")
        print(f"📁 Images directory: {images_dir}")
        print(f"📄 Captions file: {captions_file}")
        print(f"🖼️  Number of images: {len(image_files)}")
        
        # Check captions file
        try:
            import pandas as pd
            captions_df = pd.read_csv(captions_file)
            print(f"📝 Number of captions: {len(captions_df)}")
            print(f"📊 Captions per image: {len(captions_df) / len(image_files):.1f}")
        except Exception as e:
            print(f"⚠️  Could not read captions file: {e}")
        
        return True
    
    return False

def main():
    """Main setup function"""
    print("🔍 Multimodal Search Engine - Dataset Setup")
    print("=" * 50)
    print()
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("📁 Creating data directory...")
        os.makedirs("data", exist_ok=True)
    
    # Run setup
    success = setup_flickr8k()
    
    if success:
        print("\n✅ Dataset setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Run data preparation: python src/data_preparation.py")
        print("2. Test search engine: python src/search_engine.py")
        print("3. Launch web interface: streamlit run src/web_interface.py")
        print("4. Open notebook: jupyter notebook notebooks/multimodal_search_demo.ipynb")
    else:
        print("\n❌ Dataset setup failed. Please check the instructions above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 