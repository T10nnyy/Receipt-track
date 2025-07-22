"""
Receipt Processing Application - Main Entry Point
A comprehensive receipt processing application using Streamlit.
"""

import streamlit as st
import sys
import os
import logging
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.database import DatabaseManager
from core.parsing import FileProcessor
from ui.components import setup_sidebar, display_upload_section

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('receipt_processor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def initialize_app():
    """Initialize the application and database."""
    try:
        # Initialize database
        db_manager = DatabaseManager()
        db_manager.initialize_database()
        
        # Initialize session state
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = db_manager
        
        if 'file_processor' not in st.session_state:
            st.session_state.file_processor = FileProcessor()
            
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
            
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Receipt Processing Application",
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize application
    initialize_app()
    
    # Main header
    st.title("🧾 Receipt Processing Application")
    st.markdown("---")
    
    # Welcome section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Receipt Processing System
        
        This application helps you extract, analyze, and manage receipt data with the following features:
        
        ### 📄 **Multi-Format Support**
        - Upload PDFs, images (PNG, JPG, JPEG)
        - Intelligent format detection and processing
        - Hybrid OCR approach for optimal accuracy
        
        ### 🔍 **Smart Data Extraction**
        - Automatic vendor detection
        - Date and amount parsing
        - Category classification
        - Manual correction capabilities
        
        ### 📊 **Analytics & Insights**
        - Spending trends and patterns
        - Vendor analysis
        - Interactive visualizations
        - Export functionality
        
        ### 🛠️ **Data Management**
        - Search and filter receipts
        - Edit and update records
        - Bulk operations
        - Data integrity validation
        """)
    
    with col2:
        st.info("""
        **Quick Start:**
        1. Use the sidebar to upload receipt files
        2. Review extracted data in Data Explorer
        3. View insights in Analytics Dashboard
        4. Export data as needed
        """)
    
    # Setup sidebar
    setup_sidebar()
    
    # Display upload section
    display_upload_section()
    
    # Display recent activity
    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("📈 Recent Activity")
        
        recent_files = st.session_state.processed_files[-5:]  # Show last 5
        for file_info in reversed(recent_files):
            with st.expander(f"📁 {file_info['filename']} - {file_info['status']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Processed:** {file_info['timestamp']}")
                with col2:
                    st.write(f"**Size:** {file_info.get('size', 'N/A')}")
                with col3:
                    if file_info['status'] == 'Success':
                        st.success("✅ Processed")
                    else:
                        st.error("❌ Failed")
                
                if 'extracted_data' in file_info:
                    st.json(file_info['extracted_data'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Receipt Processing Application | Built with Streamlit | 
        <a href='#' style='color: #666;'>Documentation</a> | 
        <a href='#' style='color: #666;'>Support</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
