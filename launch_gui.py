#!/usr/bin/env python3
"""
Launch Script for Credit Default Risk Prediction GUI

This script launches the Streamlit GUI with proper configuration and error handling.
"""

import subprocess
import sys
import os

def launch_gui():
    """Launch the Streamlit GUI"""
    
    print("🚀 Launching Credit Default Risk Prediction GUI...")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully")
    
    # Check if plotly is installed
    try:
        import plotly
        print("✅ Plotly is installed")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        print("✅ Plotly installed successfully")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gui_file = os.path.join(script_dir, "gui.py")
    
    if not os.path.exists(gui_file):
        print(f"❌ GUI file not found: {gui_file}")
        return False
    
    print(f"📁 GUI file found: {gui_file}")
    print("🌐 Starting Streamlit server...")
    print("📱 The GUI will open in your default web browser")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", gui_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  GUI stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        return False

def main():
    """Main function"""
    print("🎯 Credit Default Risk Prediction - GUI Launcher")
    print("=" * 60)
    
    success = launch_gui()
    
    if success:
        print("✅ GUI launched successfully")
    else:
        print("❌ Failed to launch GUI")
        sys.exit(1)

if __name__ == "__main__":
    main()
