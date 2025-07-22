"""
Data Explorer page for receipt processing application.
Provides interactive table display, search, filter, and edit capabilities.
"""

import streamlit as st
import pandas as pd
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from core.models import SearchFilters, ReceiptUpdate
from core.algorithms import SearchEngine
from ui.components import display_search_filters, display_edit_modal

logger = logging.getLogger(__name__)

def main():
    """Main function for Data Explorer page."""
    st.set_page_config(
        page_title="Data Explorer - Receipt Processing",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Data Explorer")
    st.markdown("Search, filter, and manage your receipt data")
    
    # Check if database manager is initialized
    if 'db_manager' not in st.session_state:
        st.error("Database not initialized. Please return to the main page.")
        return
    
    db_manager = st.session_state.db_manager
    
    # Initialize search engine
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = SearchEngine(db_manager)
    
    search_engine = st.session_state.search_engine
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Search & Filter")
        
        # Search input
        search_query = st.text_input(
            "Search receipts",
            placeholder="Enter vendor name, description, etc.",
            help="Search across vendor names and descriptions"
        )
        
        # Filter controls
        st.subheader("Filters")
        
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=None,
                help="Filter receipts from this date"
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=None,
                help="Filter receipts to this date"
            )
        
        # Amount range filter
        st.subheader("Amount Range")
        col1, col2 = st.columns(2)
        with col1:
            min_amount = st.number_input(
                "Min Amount",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f"
            )
        with col2:
            max_amount = st.number_input(
                "Max Amount",
                min_value=0.0,
                value=10000.0,
                step=0.01,
                format="%.2f"
            )
        
        # Category filter
        try:
            all_receipts = db_manager.get_all_receipts()
            categories = sorted(list(set(r.category or "Other" for r in all_receipts)))
            
            category_filter = st.selectbox(
                "Category",
                options=["All"] + categories,
                index=0
            )
        except Exception as e:
            logger.error(f"Error loading categories: {e}")
            category_filter = "All"
            st.error("Error loading categories")
        
        # Currency filter
        currency_filter = st.selectbox(
            "Currency",
            options=["All", "USD", "EUR", "GBP", "JPY"],
            index=0
        )
        
        # Apply filters button
        apply_filters = st.button("🔍 Apply Filters", type="primary")
    
    # Main content area
    try:
        # Create search filters object
        filters = SearchFilters(
            vendor_search=search_query if search_query else None,
            category_filter=category_filter if category_filter != "All" else None,
            min_amount=Decimal(str(min_amount)) if min_amount > 0 else None,
            max_amount=Decimal(str(max_amount)) if max_amount != 10000.0 else None,
            start_date=start_date,
            end_date=end_date,
            currency_filter=currency_filter if currency_filter != "All" else None
        )
        
        # Get filtered receipts
        if apply_filters or search_query or any([start_date, end_date, category_filter != "All", currency_filter != "All", min_amount > 0, max_amount != 10000.0]):
            receipts = search_engine.advanced_search(filters)
        else:
            receipts = db_manager.get_all_receipts()
        
        # Display results summary
        total_count = db_manager.get_receipt_count()
        filtered_count = len(receipts)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Receipts", total_count)
        with col2:
            st.metric("Filtered Results", filtered_count)
        with col3:
            if receipts:
                total_amount = sum(r.amount for r in receipts)
                st.metric("Total Amount", f"${total_amount:,.2f}")
            else:
                st.metric("Total Amount", "$0.00")
        with col4:
            if receipts:
                avg_amount = sum(r.amount for r in receipts) / len(receipts)
                st.metric("Average Amount", f"${avg_amount:.2f}")
            else:
                st.metric("Average Amount", "$0.00")
        
        st.markdown("---")
        
        if not receipts:
            st.info("No receipts found matching your criteria. Try adjusting your filters.")
            return
        
        # Sorting controls
        col1, col2 = st.columns([3, 1])
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                options=["date", "amount", "vendor", "category"],
                index=0,
                format_func=lambda x: {
                    "date": "Transaction Date",
                    "amount": "Amount", 
                    "vendor": "Vendor",
                    "category": "Category"
                }[x]
            )
        with col2:
            sort_desc = st.checkbox("Descending", value=True)
        
        # Sort receipts
        receipts = search_engine._sort_receipts(receipts, sort_by, sort_desc)
        
        # Convert to DataFrame for display
        receipt_data = []
        for receipt in receipts:
            receipt_data.append({
                "ID": receipt.id,
                "Vendor": receipt.vendor,
                "Date": receipt.transaction_date.strftime("%Y-%m-%d"),
                "Amount": f"${receipt.amount:.2f}",
                "Category": receipt.category or "Other",
                "Currency": receipt.currency,
                "Source File": receipt.source_file,
                "Description": receipt.description or "",
                "Confidence": f"{(receipt.processing_confidence or 0) * 100:.1f}%" if receipt.processing_confidence else "N/A"
            })
        
        df = pd.DataFrame(receipt_data)
        
        # Display data with selection
        st.subheader("📋 Receipt Data")
        
        # Selection mode
        selection_mode = st.radio(
            "Selection Mode",
            options=["View Only", "Edit Mode", "Bulk Operations"],
            horizontal=True
        )
        
        if selection_mode == "View Only":
            # Display as read-only table
            st.dataframe(
                df,
                use_container_width=True,
                height=400
            )
            
        elif selection_mode == "Edit Mode":
            # Display editable grid (simulated with selection)
            st.info("Select a row to edit the receipt details")
            
            selected_indices = st.multiselect(
                "Select receipts to edit (by ID)",
                options=range(len(receipts)),
                format_func=lambda x: f"ID {receipts[x].id} - {receipts[x].vendor}",
                max_selections=1
            )
            
            # Display table
            st.dataframe(df, use_container_width=True, height=400)
            
            # Edit selected receipt
            if selected_indices:
                selected_receipt = receipts[selected_indices[0]]
                display_edit_form(selected_receipt, db_manager)
                
        elif selection_mode == "Bulk Operations":
            # Bulk operations interface
            st.info("Select multiple receipts for bulk operations")
            
            selected_indices = st.multiselect(
                "Select receipts (by ID)",
                options=range(len(receipts)),
                format_func=lambda x: f"ID {receipts[x].id} - {receipts[x].vendor}"
            )
            
            # Display table with selected rows highlighted
            st.dataframe(df, use_container_width=True, height=400)
            
            if selected_indices:
                display_bulk_operations(selected_indices, receipts, db_manager)
        
        # Export functionality
        st.markdown("---")
        st.subheader("📥 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export to CSV"):
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"receipts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Export to JSON"):
                json_data = df.to_json(orient="records", indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"receipts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📈 Export Filtered Data"):
                filtered_data = []
                for receipt in receipts:
                    filtered_data.append({
                        "id": receipt.id,
                        "vendor": receipt.vendor,
                        "transaction_date": receipt.transaction_date.isoformat(),
                        "amount": float(receipt.amount),
                        "category": receipt.category,
                        "currency": receipt.currency,
                        "source_file": receipt.source_file,
                        "description": receipt.description,
                        "processing_confidence": receipt.processing_confidence
                    })
                
                import json
                json_data = json.dumps(filtered_data, indent=2)
                st.download_button(
                    label="Download Filtered JSON",
                    data=json_data,
                    file_name=f"filtered_receipts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
    except Exception as e:
        logger.error(f"Error in Data Explorer: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def display_edit_form(receipt, db_manager):
    """Display edit form for a selected receipt.
    
    Args:
        receipt: Receipt object to edit
        db_manager: Database manager instance
    """
    st.subheader(f"✏️ Edit Receipt ID: {receipt.id}")
    
    with st.form(f"edit_receipt_{receipt.id}"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_vendor = st.text_input("Vendor", value=receipt.vendor)
            new_date = st.date_input("Transaction Date", value=receipt.transaction_date)
            new_amount = st.number_input(
                "Amount", 
                value=float(receipt.amount), 
                min_value=0.01,
                step=0.01,
                format="%.2f"
            )
        
        with col2:
            categories = ["Food & Dining", "Gas & Fuel", "Shopping", "Entertainment", 
                         "Healthcare", "Travel", "Business", "Education", "Other"]
            current_category_index = 0
            if receipt.category in categories:
                current_category_index = categories.index(receipt.category)
            
            new_category = st.selectbox("Category", categories, index=current_category_index)
            new_currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"], 
                                      index=["USD", "EUR", "GBP", "JPY"].index(receipt.currency))
            new_description = st.text_area("Description", value=receipt.description or "")
        
        # Form buttons
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("💾 Save Changes", type="primary")
        with col2:
            cancelled = st.form_submit_button("❌ Cancel")
        
        if submitted:
            try:
                # Create update object
                updates = ReceiptUpdate(
                    vendor=new_vendor if new_vendor != receipt.vendor else None,
                    transaction_date=new_date if new_date != receipt.transaction_date else None,
                    amount=Decimal(str(new_amount)) if Decimal(str(new_amount)) != receipt.amount else None,
                    category=new_category if new_category != receipt.category else None,
                    currency=new_currency if new_currency != receipt.currency else None,
                    description=new_description if new_description != (receipt.description or "") else None
                )
                
                # Update in database
                success = db_manager.update_receipt(receipt.id, updates)
                
                if success:
                    st.success("✅ Receipt updated successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to update receipt")
                    
            except Exception as e:
                logger.error(f"Error updating receipt: {str(e)}")
                st.error(f"Error updating receipt: {str(e)}")

def display_bulk_operations(selected_indices, receipts, db_manager):
    """Display bulk operations interface.
    
    Args:
        selected_indices: List of selected receipt indices
        receipts: List of all receipts
        db_manager: Database manager instance
    """
    st.subheader(f"🔧 Bulk Operations ({len(selected_indices)} receipts selected)")
    
    operation = st.selectbox(
        "Choose Operation",
        ["Select Operation", "Update Category", "Delete Receipts", "Update Currency"]
    )
    
    if operation == "Update Category":
        categories = ["Food & Dining", "Gas & Fuel", "Shopping", "Entertainment", 
                     "Healthcare", "Travel", "Business", "Education", "Other"]
        new_category = st.selectbox("New Category", categories)
        
        if st.button("🏷️ Update Categories", type="primary"):
            try:
                success_count = 0
                for idx in selected_indices:
                    receipt = receipts[idx]
                    update = ReceiptUpdate(category=new_category)
                    if db_manager.update_receipt(receipt.id, update):
                        success_count += 1
                
                st.success(f"✅ Updated {success_count} receipts successfully!")
                if success_count > 0:
                    st.rerun()
                    
            except Exception as e:
                logger.error(f"Bulk category update error: {str(e)}")
                st.error(f"Error updating categories: {str(e)}")
    
    elif operation == "Delete Receipts":
        st.warning(f"⚠️ This will permanently delete {len(selected_indices)} receipts!")
        
        # Show receipts to be deleted
        with st.expander("Receipts to be deleted"):
            for idx in selected_indices:
                receipt = receipts[idx]
                st.write(f"• ID {receipt.id}: {receipt.vendor} - ${receipt.amount} ({receipt.transaction_date})")
        
        confirm_delete = st.checkbox("I confirm I want to delete these receipts")
        
        if confirm_delete and st.button("🗑️ Delete Receipts", type="secondary"):
            try:
                success_count = 0
                for idx in selected_indices:
                    receipt = receipts[idx]
                    if db_manager.delete_receipt(receipt.id):
                        success_count += 1
                
                st.success(f"✅ Deleted {success_count} receipts successfully!")
                if success_count > 0:
                    st.rerun()
                    
            except Exception as e:
                logger.error(f"Bulk delete error: {str(e)}")
                st.error(f"Error deleting receipts: {str(e)}")
    
    elif operation == "Update Currency":
        new_currency = st.selectbox("New Currency", ["USD", "EUR", "GBP", "JPY"])
        
        if st.button("💱 Update Currency", type="primary"):
            try:
                success_count = 0
                for idx in selected_indices:
                    receipt = receipts[idx]
                    update = ReceiptUpdate(currency=new_currency)
                    if db_manager.update_receipt(receipt.id, update):
                        success_count += 1
                
                st.success(f"✅ Updated {success_count} receipts successfully!")
                if success_count > 0:
                    st.rerun()
                    
            except Exception as e:
                logger.error(f"Bulk currency update error: {str(e)}")
                st.error(f"Error updating currency: {str(e)}")

if __name__ == "__main__":
    main()
