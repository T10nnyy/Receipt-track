"""
UI components for the receipt processing application.
Provides reusable interface elements for file upload, search, and data display.
"""

import streamlit as st
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any
import io
from pathlib import Path

from core.models import SearchFilters, ProcessingResult
from core.parsing import FileProcessor

logger = logging.getLogger(__name__)

def setup_sidebar():
    """Setup the main sidebar with global controls and information."""
    with st.sidebar:
        st.header("🧾 Receipt Processor")
        
        # Application info
        st.markdown("""
        **Version:** 1.0.0  
        **Status:** ✅ Ready
        """)
        
        # Quick stats
        if 'db_manager' in st.session_state:
            try:
                total_receipts = st.session_state.db_manager.get_receipt_count()
                if total_receipts > 0:
                    analytics_data = st.session_state.db_manager.get_analytics_data()
                    
                    st.markdown("---")
                    st.subheader("📊 Quick Stats")
                    st.metric("Total Receipts", total_receipts)
                    st.metric("Total Spending", f"${analytics_data.total_amount:,.2f}")
                    st.metric("Average Amount", f"${analytics_data.average_amount:.2f}")
                    
                    if analytics_data.date_range["earliest"] and analytics_data.date_range["latest"]:
                        date_range_days = (analytics_data.date_range["latest"] - analytics_data.date_range["earliest"]).days
                        st.metric("Date Range", f"{date_range_days} days")
                
            except Exception as e:
                logger.error(f"Error loading sidebar stats: {e}")
                st.error("Error loading statistics")
        
        st.markdown("---")
        
        # Navigation info
        st.markdown("""
        ### 📍 Navigation
        - **Home**: Upload & process receipts
        - **Data Explorer**: Search & edit receipts  
        - **Analytics**: View spending insights
        """)
        
        # Help section
        with st.expander("❓ Help & Tips"):
            st.markdown("""
            **Supported Formats:**
            - PDF files (text or scanned)
            - Images: PNG, JPG, JPEG, BMP, TIFF
            
            **Best Results:**
            - Clear, well-lit images
            - Straight orientation
            - Minimal background noise
            - High resolution (300+ DPI)
            
            **Need Help?**
            - Check file format compatibility
            - Ensure good image quality
            - Contact support for issues
            """)

def display_upload_section():
    """Display the file upload section with processing capabilities."""
    st.subheader("📁 Upload Receipts")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose receipt files",
        type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'],
        accept_multiple_files=True,
        help="Upload receipt files in PDF or image format"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        
        # Processing options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            processing_options = st.expander("⚙️ Processing Options", expanded=False)
            with processing_options:
                auto_categorize = st.checkbox(
                    "Auto-categorize receipts",
                    value=True,
                    help="Automatically assign categories based on vendor patterns"
                )
                
                confidence_threshold = st.slider(
                    "Minimum confidence threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Minimum confidence score for automatic processing"
                )
                
                manual_review = st.checkbox(
                    "Manual review for low confidence",
                    value=True,
                    help="Flag receipts with low confidence for manual review"
                )
        
        with col2:
            process_button = st.button(
                "🚀 Process Files",
                type="primary",
                use_container_width=True
            )
        
        # Process files when button is clicked
        if process_button:
            process_uploaded_files(
                uploaded_files, 
                auto_categorize, 
                confidence_threshold, 
                manual_review
            )

def create_processing_result_from_dict(result_dict: Dict[str, Any], processing_time: float = 0.0) -> ProcessingResult:
    """Convert dictionary result to ProcessingResult object.
    
    Args:
        result_dict: Dictionary containing processing results
        processing_time: Time taken for processing
        
    Returns:
        ProcessingResult object
    """
    try:
        # Create a ProcessingResult object from the dictionary
        processing_result = ProcessingResult()
        
        # Set basic attributes
        processing_result.success = result_dict.get("success", False)
        processing_result.processing_time = processing_time
        processing_result.errors = result_dict.get("errors", [])
        
        # Handle confidence score
        processing_result.confidence_score = result_dict.get("confidence", 0.0)
        
        # If successful and has receipt data, set the receipt
        if processing_result.success and "receipt" in result_dict:
            processing_result.receipt = result_dict["receipt"]
        elif processing_result.success and "text" in result_dict:
            # If we only have text, we need to parse it into a receipt
            # This is a placeholder - you'll need to implement text parsing
            processing_result.receipt = None  # Implement text-to-receipt parsing here
        else:
            processing_result.receipt = None
            
        # Add any error messages
        if not processing_result.success:
            error_msg = result_dict.get("error", "Unknown processing error")
            if error_msg not in processing_result.errors:
                processing_result.errors.append(error_msg)
        
        return processing_result
        
    except Exception as e:
        logger.error(f"Error creating ProcessingResult from dict: {str(e)}")
        # Return a failed result
        failed_result = ProcessingResult()
        failed_result.success = False
        failed_result.errors = [f"Error converting result: {str(e)}"]
        failed_result.processing_time = processing_time
        failed_result.confidence_score = 0.0
        failed_result.receipt = None
        return failed_result

def process_uploaded_files(uploaded_files: List, auto_categorize: bool, confidence_threshold: float, manual_review: bool):
    """Process uploaded files and display results.
    
    Args:
        uploaded_files: List of uploaded file objects
        auto_categorize: Whether to auto-categorize receipts
        confidence_threshold: Minimum confidence threshold
        manual_review: Whether to flag low confidence items
    """
    if 'file_processor' not in st.session_state:
        st.error("File processor not initialized")
        return
    
    if 'db_manager' not in st.session_state:
        st.error("Database manager not initialized")
        return
    
    file_processor = st.session_state.file_processor
    db_manager = st.session_state.db_manager
    
    # Initialize processed files list if not exists
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    successful_processes = 0
    failed_processes = 0
    low_confidence_items = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        start_time = datetime.now()
        
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}... ({i + 1}/{len(uploaded_files)})")
            
            # Read file content
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Process file - handle both dict and object returns
            raw_result = file_processor.process_file(file_content, uploaded_file.name)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to ProcessingResult if it's a dictionary
            if isinstance(raw_result, dict):
                result = create_processing_result_from_dict(raw_result, processing_time)
            else:
                result = raw_result  # Assume it's already a ProcessingResult object
                if not hasattr(result, 'processing_time'):
                    result.processing_time = processing_time
            
            # Record processing info
            file_info = {
                'filename': uploaded_file.name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'size': f"{len(file_content)} bytes",
                'status': 'Success' if result.success else 'Failed',
                'processing_time': result.processing_time
            }
            
            if result.success and result.receipt:
                # Check confidence
                confidence = result.confidence_score or 0.0
                
                if confidence >= confidence_threshold:
                    # Auto-categorize if enabled
                    if auto_categorize:
                        result.receipt.category = categorize_receipt(result.receipt)
                    
                    # Save to database
                    receipt_id = db_manager.add_receipt(result.receipt)
                    
                    file_info['extracted_data'] = {
                        'receipt_id': receipt_id,
                        'vendor': result.receipt.vendor,
                        'amount': float(result.receipt.amount),
                        'date': result.receipt.transaction_date.isoformat(),
                        'category': result.receipt.category,
                        'confidence': confidence
                    }
                    
                    successful_processes += 1
                    
                elif manual_review:
                    # Flag for manual review
                    low_confidence_items.append({
                        'filename': uploaded_file.name,
                        'result': result,
                        'confidence': confidence
                    })
                    file_info['status'] = 'Needs Review'
                else:
                    # Skip low confidence items
                    file_info['status'] = 'Skipped (Low Confidence)'
                    file_info['confidence'] = confidence
                    failed_processes += 1
                    
            else:
                # Processing failed
                file_info['errors'] = result.errors if hasattr(result, 'errors') else ["Unknown error"]
                failed_processes += 1
            
            # Add to processed files history
            st.session_state.processed_files.append(file_info)
            
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
            failed_processes += 1
            
            file_info = {
                'filename': uploaded_file.name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'Error',
                'error': str(e)
            }
            st.session_state.processed_files.append(file_info)
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Display results summary
    with results_container:
        st.markdown("---")
        st.subheader("📋 Processing Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Successful", successful_processes)
        with col2:
            st.metric("❌ Failed", failed_processes)
        with col3:
            st.metric("⚠️ Needs Review", len(low_confidence_items))
        
        # Show detailed results
        if successful_processes > 0:
            st.success(f"Successfully processed {successful_processes} receipt(s)")
        
        if failed_processes > 0:
            st.error(f"Failed to process {failed_processes} file(s)")
            
            # Show error details for failed files
            with st.expander("❌ View Failed Files", expanded=False):
                failed_files = [f for f in st.session_state.processed_files if f['status'] in ['Failed', 'Error']]
                for file_info in failed_files[-5:]:  # Show last 5 failed files
                    st.write(f"**{file_info['filename']}**")
                    if 'error' in file_info:
                        st.text(f"Error: {file_info['error']}")
                    if 'errors' in file_info:
                        for error in file_info['errors']:
                            st.text(f"• {error}")
                    st.markdown("---")
        
        # Manual review section
        if low_confidence_items:
            st.warning(f"{len(low_confidence_items)} item(s) need manual review")
            
            with st.expander("🔍 Review Low Confidence Items", expanded=True):
                for item in low_confidence_items:
                    display_manual_review_item(item, db_manager, auto_categorize)
    
    # Clear progress indicators after a delay
    import time
    time.sleep(2)
    progress_bar.empty()
    status_text.empty()

def display_manual_review_item(item: Dict[str, Any], db_manager, auto_categorize: bool):
    """Display manual review interface for low confidence items.
    
    Args:
        item: Dictionary containing filename, result, and confidence
        db_manager: Database manager instance
        auto_categorize: Whether to auto-categorize
    """
    st.markdown(f"**📄 {item['filename']}** (Confidence: {item['confidence']:.1%})")
    
    result = item['result']
    
    if result.receipt:
        with st.form(f"review_{item['filename']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                vendor = st.text_input("Vendor", value=result.receipt.vendor or "")
                amount = st.number_input(
                    "Amount", 
                    value=float(result.receipt.amount) if result.receipt.amount else 0.0,
                    min_value=0.01,
                    step=0.01,
                    format="%.2f"
                )
                date_value = st.date_input("Date", value=result.receipt.transaction_date or datetime.now().date())
            
            with col2:
                categories = ["Food & Dining", "Gas & Fuel", "Shopping", "Entertainment", 
                             "Healthcare", "Travel", "Business", "Education", "Other"]
                
                initial_category = result.receipt.category
                if auto_categorize and (not initial_category or initial_category not in categories):
                    initial_category = categorize_receipt(result.receipt)
                
                category_index = 0
                if initial_category and initial_category in categories:
                    category_index = categories.index(initial_category)
                
                category = st.selectbox("Category", categories, index=category_index)
                currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"], 
                                      index=["USD", "EUR", "GBP", "JPY"].index(result.receipt.currency) if result.receipt.currency in ["USD", "EUR", "GBP", "JPY"] else 0)
                description = st.text_area("Description", value=result.receipt.description or "")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                save_button = st.form_submit_button("💾 Save Receipt", type="primary")
            with col2:
                skip_button = st.form_submit_button("⏭️ Skip")
            with col3:
                discard_button = st.form_submit_button("🗑️ Discard")
            
            if save_button:
                try:
                    # Update receipt with reviewed data
                    result.receipt.vendor = vendor
                    result.receipt.amount = Decimal(str(amount))
                    result.receipt.transaction_date = date_value
                    result.receipt.category = category
                    result.receipt.currency = currency
                    result.receipt.description = description
                    result.receipt.processing_confidence = 1.0  # Manual review = high confidence
                    
                    # Save to database
                    receipt_id = db_manager.add_receipt(result.receipt)
                    st.success(f"✅ Receipt saved with ID: {receipt_id}")
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Error saving reviewed receipt: {str(e)}")
                    st.error(f"Error saving receipt: {str(e)}")
            
            elif skip_button:
                st.info("⏭️ Receipt skipped")
                st.rerun()
            
            elif discard_button:
                st.warning("🗑️ Receipt discarded")
                st.rerun()
    
    else:
        st.error("❌ No receipt data extracted")
        if hasattr(result, 'errors') and result.errors:
            for error in result.errors:
                st.text(f"• {error}")

def categorize_receipt(receipt) -> str:
    """Auto-categorize receipt based on vendor name patterns.
    
    Args:
        receipt: Receipt object to categorize
        
    Returns:
        Category string
    """
    if not receipt or not hasattr(receipt, 'vendor') or not receipt.vendor:
        return "Other"
        
    vendor_lower = receipt.vendor.lower()
    
    # Food & Dining patterns
    food_keywords = [
        'restaurant', 'cafe', 'coffee', 'starbucks', 'mcdonald', 'burger', 'pizza',
        'food', 'deli', 'bakery', 'bar', 'grill', 'kitchen', 'bistro', 'diner',
        'subway', 'taco', 'chinese', 'thai', 'sushi', 'indian', 'mexican'
    ]
    
    # Gas & Fuel patterns
    fuel_keywords = [
        'gas', 'fuel', 'shell', 'exxon', 'bp', 'chevron', 'mobil', 'station',
        'petroleum', 'texaco', 'citgo', 'speedway', 'wawa'
    ]
    
    # Shopping patterns
    shopping_keywords = [
        'walmart', 'target', 'amazon', 'store', 'market', 'shop', 'mall',
        'pharmacy', 'cvs', 'walgreens', 'costco', 'depot', 'best buy',
        'clothing', 'fashion', 'outlet'
    ]
    
    # Entertainment patterns
    entertainment_keywords = [
        'movie', 'theater', 'cinema', 'game', 'sport', 'gym', 'fitness',
        'club', 'bar', 'entertainment', 'netflix', 'spotify', 'concert'
    ]
    
    # Healthcare patterns
    healthcare_keywords = [
        'hospital', 'clinic', 'doctor', 'medical', 'pharmacy', 'health',
        'dental', 'vision', 'urgent care', 'lab'
    ]
    
    # Travel patterns
    travel_keywords = [
        'hotel', 'airline', 'flight', 'rental', 'uber', 'lyft', 'taxi',
        'travel', 'booking', 'airbnb', 'airport', 'parking'
    ]
    
    # Business patterns
    business_keywords = [
        'office', 'supply', 'service', 'consulting', 'software', 'license',
        'subscription', 'professional', 'legal', 'accounting'
    ]
    
    # Education patterns
    education_keywords = [
        'school', 'university', 'college', 'education', 'tuition', 'book',
        'course', 'training', 'learning'
    ]
    
    # Check patterns
    if any(keyword in vendor_lower for keyword in food_keywords):
        return "Food & Dining"
    elif any(keyword in vendor_lower for keyword in fuel_keywords):
        return "Gas & Fuel"
    elif any(keyword in vendor_lower for keyword in shopping_keywords):
        return "Shopping"
    elif any(keyword in vendor_lower for keyword in entertainment_keywords):
        return "Entertainment"
    elif any(keyword in vendor_lower for keyword in healthcare_keywords):
        return "Healthcare"
    elif any(keyword in vendor_lower for keyword in travel_keywords):
        return "Travel"
    elif any(keyword in vendor_lower for keyword in business_keywords):
        return "Business"
    elif any(keyword in vendor_lower for keyword in education_keywords):
        return "Education"
    else:
        return "Other"

def display_search_filters(filters: SearchFilters) -> SearchFilters:
    """Display search filter interface and return updated filters.
    
    Args:
        filters: Current search filters
        
    Returns:
        Updated SearchFilters object
    """
    with st.expander("🔍 Advanced Search Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            vendor_search = st.text_input(
                "Search Vendor",
                value=filters.vendor_search or "",
                placeholder="Enter vendor name"
            )
            
            min_amount = st.number_input(
                "Minimum Amount",
                value=float(filters.min_amount) if filters.min_amount else 0.0,
                min_value=0.0,
                step=0.01,
                format="%.2f"
            )
            
            start_date = st.date_input(
                "Start Date",
                value=filters.start_date
            )
        
        with col2:
            category_filter = st.selectbox(
                "Category",
                ["All", "Food & Dining", "Gas & Fuel", "Shopping", "Entertainment", 
                 "Healthcare", "Travel", "Business", "Education", "Other"],
                index=0 if not filters.category_filter else 
                      ["All", "Food & Dining", "Gas & Fuel", "Shopping", "Entertainment", 
                       "Healthcare", "Travel", "Business", "Education", "Other"].index(filters.category_filter)
            )
            
            max_amount = st.number_input(
                "Maximum Amount",
                value=float(filters.max_amount) if filters.max_amount else 10000.0,
                min_value=0.0,
                step=0.01,
                format="%.2f"
            )
            
            end_date = st.date_input(
                "End Date",
                value=filters.end_date
            )
    
    return SearchFilters(
        vendor_search=vendor_search if vendor_search else None,
        category_filter=category_filter if category_filter != "All" else None,
        min_amount=Decimal(str(min_amount)) if min_amount > 0 else None,
        max_amount=Decimal(str(max_amount)) if max_amount != 10000.0 else None,
        start_date=start_date,
        end_date=end_date
    )

def display_edit_modal(receipt_id: int):
    """Display edit modal for a receipt (placeholder for future implementation).
    
    Args:
        receipt_id: ID of receipt to edit
    """
    st.info(f"Edit modal for receipt {receipt_id} - Feature in development")

def display_progress_bar(current: int, total: int, message: str = "Processing..."):
    """Display a progress bar with status message.
    
    Args:
        current: Current progress value
        total: Total value for completion
        message: Status message to display
    """
    progress = current / total if total > 0 else 0
    st.progress(progress)
    st.text(f"{message} ({current}/{total})")

def display_error_message(error: str, details: Optional[str] = None):
    """Display formatted error message.
    
    Args:
        error: Main error message
        details: Optional detailed error information
    """
    st.error(f"❌ {error}")
    
    if details:
        with st.expander("🔍 Error Details"):
            st.code(details, language="text")

def display_success_message(message: str, details: Optional[Dict[str, Any]] = None):
    """Display formatted success message.
    
    Args:
        message: Success message
        details: Optional additional details
    """
    st.success(f"✅ {message}")
    
    if details:
        with st.expander("📋 Details"):
            for key, value in details.items():
                st.text(f"{key}: {value}")

def format_currency(amount: Decimal, currency: str = "USD") -> str:
    """Format currency amount for display.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    currency_symbols = {
        "USD": "$",
        "EUR": "€", 
        "GBP": "£",
        "JPY": "¥"
    }
    
    symbol = currency_symbols.get(currency, currency)
    
    if currency == "JPY":
        return f"{symbol}{amount:,.0f}"
    else:
        return f"{symbol}{amount:,.2f}"

def validate_file_upload(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded file for processing.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not uploaded_file:
        return False, "No file uploaded"
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if uploaded_file.size > max_size:
        return False, f"File size ({uploaded_file.size:,} bytes) exceeds maximum allowed size (10MB)"
    
    # Check file type
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    file_ext = Path(uploaded_file.name).suffix.lower()
    
    if file_ext not in allowed_extensions:
        return False, f"File type '{file_ext}' not supported. Allowed types: {', '.join(allowed_extensions)}"
    
    return True, ""
