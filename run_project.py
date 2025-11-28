#!/usr/bin/env python3
"""
Main run script for the Multimodal Search Engine project
This script orchestrates the entire workflow from setup to web interface
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print project banner"""
    print("=" * 80)
    print("🔍 Multimodal Search Engine with Flickr-8k Dataset")
    print("=" * 80)
    print()
    print("This project implements a complete multimodal search engine with:")
    print("• Text-to-Image Search")
    print("• Image-to-Text Search") 
    print("• CLIP Model Integration")
    print("• Streamlit Web Interface")
    print("• Comprehensive Analysis")
    print()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'streamlit', 
        'PIL', 'numpy', 'pandas', 'matplotlib', 'seaborn',
        'scikit-learn', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def run_command(command, description):
    """Run a command with description"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Command completed successfully!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        sys.exit(1)
    
    print("\n🚀 Project Workflow Options:")
    print("1. Complete setup (Dataset + Embeddings + Web Interface)")
    print("2. Dataset setup only")
    print("3. Generate embeddings only")
    print("4. Run search engine only")
    print("5. Launch web interface only")
    print("6. Run Jupyter notebook")
    print("7. Exit")
    print()
    
    choice = input("Enter your choice (1-7): ").strip()
    
    if choice == "1":
        # Complete setup
        print("\n🎯 Running complete setup...")
        
        # Step 1: Dataset setup
        print("\n📁 Step 1: Dataset Setup")
        if not run_command("python setup_dataset.py", "Setting up Flickr-8k dataset"):
            print("❌ Dataset setup failed. Please check the setup script.")
            return
        
        # Step 2: Generate embeddings
        print("\n🔧 Step 2: Generate Embeddings")
        if not run_command("python src/data_preparation.py", "Generating embeddings"):
            print("❌ Embedding generation failed. Please check the data preparation script.")
            return
        
        # Step 3: Test search engine
        print("\n🔍 Step 3: Test Search Engine")
        print("Testing search functionality...")
        try:
            from src.search_engine import MultimodalSearchEngine
            search_engine = MultimodalSearchEngine()
            print("✅ Search engine initialized successfully!")
        except Exception as e:
            print(f"❌ Search engine test failed: {e}")
            return
        
        # Step 4: Launch web interface
        print("\n🌐 Step 4: Launch Web Interface")
        print("Starting Streamlit web interface...")
        print("The web interface will open in your browser.")
        print("Press Ctrl+C to stop the server.")
        
        try:
            subprocess.run("streamlit run src/web_interface.py", shell=True)
        except KeyboardInterrupt:
            print("\n👋 Web interface stopped.")
    
    elif choice == "2":
        # Dataset setup only
        run_command("python setup_dataset.py", "Setting up Flickr-8k dataset")
    
    elif choice == "3":
        # Generate embeddings only
        run_command("python src/data_preparation.py", "Generating embeddings")
    
    elif choice == "4":
        # Run search engine only
        run_command("python src/search_engine.py", "Running search engine")
    
    elif choice == "5":
        # Launch web interface only
        print("\n🌐 Launching Streamlit web interface...")
        print("The web interface will open in your browser.")
        print("Press Ctrl+C to stop the server.")
        
        try:
            subprocess.run("streamlit run src/web_interface.py", shell=True)
        except KeyboardInterrupt:
            print("\n👋 Web interface stopped.")
    
    elif choice == "6":
        # Run Jupyter notebook
        print("\n📓 Launching Jupyter notebook...")
        print("The notebook will open in your browser.")
        print("Press Ctrl+C to stop the server.")
        
        try:
            subprocess.run("jupyter notebook notebooks/multimodal_search_demo.ipynb", shell=True)
        except KeyboardInterrupt:
            print("\n👋 Jupyter notebook stopped.")
    
    elif choice == "7":
        print("\n👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice. Please run the script again.")
        return
    
    print("\n✅ Project workflow completed successfully!")
    print("\n📚 Additional Resources:")
    print("• README.md - Project documentation")
    print("• src/ - Source code directory")
    print("• notebooks/ - Jupyter notebooks")
    print("• embeddings/ - Generated embeddings")
    print("• data/ - Dataset directory")

if __name__ == "__main__":
    main() 