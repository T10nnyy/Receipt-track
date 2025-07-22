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
from core.export import DataExporter
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
        
        # Enhanced Export functionality
        st.markdown("---")
        st.subheader("📥 Export Data")
        
        # Initialize data exporter
        if 'data_exporter' not in st.session_state:
            st.session_state.data_exporter = DataExporter()
        
        exporter = st.session_state.data_exporter
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format",
                options=["CSV", "JSON"],
                index=0
            )
            
            export_scope = st.selectbox(
                "Export Scope",
                options=["Current View", "All Data", "Summary", "Analytics"],
                index=0,
                help="Choose what data to export"
            )
        
        with col2:
            include_metadata = st.checkbox(
                "Include Metadata",
                value=True,
                help="Include processing confidence, timestamps, etc."
            )
            
            if st.button("📤 Generate Export", type="primary"):
                try:
                    if export_scope == "Current View":
                        export_receipts = receipts
                    elif export_scope == "All Data":
                        export_receipts = db_manager.get_all_receipts()
                    else:
                        export_receipts = receipts  # Use current view for summary/analytics
                    
                    if export_format.lower() == "csv":
                        if export_scope == "Summary":
                            st.warning("Summary export is only available in JSON format")
                        else:
                            export_data = exporter.export_to_csv(
                                export_receipts,
                                include_metadata=include_metadata
                            )
                            filename = exporter.get_export_filename(
                                "csv",
                                export_scope.lower().replace(" ", "_")
                            )
                    else:  # JSON
                        format_type = {
                            "Current View": "detailed",
                            "All Data": "detailed", 
                            "Summary": "summary",
                            "Analytics": "analytics"
                        }.get(export_scope, "detailed")
                        
                        export_data = exporter.export_to_json(
                            export_receipts,
                            include_metadata=include_metadata,
                            format_type=format_type
                        )
                        filename = exporter.get_export_filename(
                            "json",
                            export_scope.lower().replace(" ", "_")
                        )
                    
                    # Provide download
                    st.success(f"Export generated successfully! ({len(export_receipts)} receipts)")
                    st.download_button(
                        label=f"📥 Download {export_format}",
                        data=export_data,
                        file_name=filename,
                        mime="text/csv" if export_format.lower() == "csv" else "application/json",
                        key=f"download_{export_scope}_{export_format}"
                    )
                    
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
                    logger.error(f"Export error: {str(e)}")
        
        # Quick export buttons
        st.markdown("### Quick Export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Quick CSV"):
                try:
                    csv_data = exporter.export_to_csv(receipts, include_metadata=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"receipts_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="quick_csv"
                    )
                except Exception as e:
                    st.error(f"Quick CSV export failed: {str(e)}")
        
        with col2:
            if st.button("📊 Summary JSON"):
                try:
                    json_data = exporter.export_to_json(receipts, format_type="summary")
                    st.download_button(
                        label="Download Summary",
                        data=json_data,
                        file_name=f"receipts_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="quick_summary"
                    )
                except Exception as e:
                    st.error(f"Summary export failed: {str(e)}")
        
        with col3:
            if st.button("📈 Analytics Export"):
                try:
                    analytics_data = exporter.export_to_json(receipts, format_type="analytics")
                    st.download_button(
                        label="Download Analytics",
                        data=analytics_data,
                        file_name=f"receipts_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="quick_analytics"
                    )
                except Exception as e:
                    st.error(f"Analytics export failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in Data Explorer: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def display_edit_form(receipt, db_manager):
    """Display comprehensive edit form for a selected receipt.
    
    Args:
        receipt: Receipt object to edit
        db_manager: Database manager instance
    """
    st.subheader(f"✏️ Edit Receipt ID: {receipt.id}")
    
    # Show confidence indicators
    confidence = receipt.processing_confidence or 0
    if confidence < 0.7:
        st.warning(f"⚠️ Low confidence ({confidence*100:.1f}%) - Please review carefully")
    elif confidence < 0.9:
        st.info(f"ℹ️ Medium confidence ({confidence*100:.1f}%) - Some fields may need verification")
    else:
        st.success(f"✅ High confidence ({confidence*100:.1f}%) - Data appears reliable")
    
    with st.form(f"edit_receipt_{receipt.id}"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_vendor = st.text_input(
                "Vendor Name",
                value=receipt.vendor,
                help="The name of the merchant or business"
            )
            
            new_amount = st.number_input(
                "Amount",
                value=float(receipt.amount),
                min_value=0.01,
                step=0.01,
                format="%.2f",
                help="Transaction amount in the specified currency"
            )
            
            new_category = st.selectbox(
                "Category",
                options=["Food & Dining", "Gas & Fuel", "Shopping", "Entertainment", 
                        "Healthcare", "Travel", "Business", "Education", "Other"],
                index=["Food & Dining", "Gas & Fuel", "Shopping", "Entertainment", 
                       "Healthcare", "Travel", "Business", "Education", "Other"].index(
                    receipt.category or "Other"
                ),
                help="Expense category for better organization"
            )
            
            new_currency = st.selectbox(
                "Currency",
                options=["USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF"],
                index=["USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF"].index(
                    receipt.currency if receipt.currency in ["USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF"] else "USD"
                ),
                help="Currency code (3-letter ISO code)"
            )
        
        with col2:
            new_date = st.date_input(
                "Transaction Date",
                value=receipt.transaction_date,
                help="The date when the transaction occurred"
            )
            
            new_description = st.text_area(
                "Description",
                value=receipt.description or "",
                height=100,
                help="Additional notes or description about the transaction"
            )
            
            # Show original source file info
            st.text_input(
                "Source File",
                value=receipt.source_file,
                disabled=True,
                help="Original file name (read-only)"
            )
            
            # Confidence override
            new_confidence = st.slider(
                "Processing Confidence",
                min_value=0.0,
                max_value=1.0,
                value=confidence,
                step=0.1,
                help="Your confidence in the accuracy of this data"
            )
        
        # Form submission
        col1, col2, col3 = st.columns(3)
        
        with col1:
            submit_button = st.form_submit_button(
                "💾 Save Changes",
                type="primary"
            )
        
        with col2:
            if st.form_submit_button("🔄 Reset to Original"):
                st.rerun()
        
        with col3:
            delete_button = st.form_submit_button(
                "🗑️ Delete Receipt",
                type="secondary"
            )
        
        if submit_button:
            try:
                # Create update object
                update_data = ReceiptUpdate(
                    vendor=new_vendor if new_vendor != receipt.vendor else None,
                    transaction_date=new_date if new_date != receipt.transaction_date else None,
                    amount=Decimal(str(new_amount)) if new_amount != float(receipt.amount) else None,
                    category=new_category if new_category != receipt.category else None,
                    currency=new_currency if new_currency != receipt.currency else None,
                    description=new_description if new_description != (receipt.description or "") else None,
                    processing_confidence=new_confidence if new_confidence != confidence else None
                )
                
                # Update in database
                success = db_manager.update_receipt(receipt.id, update_data)
                
                if success:
                    st.success("✅ Receipt updated successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to update receipt")
                    
            except Exception as e:
                st.error(f"❌ Error updating receipt: {str(e)}")
                logger.error(f"Receipt update error: {str(e)}")
        
        if delete_button:
            # Confirmation dialog (simulated)
            st.warning("⚠️ Are you sure you want to delete this receipt? This action cannot be undone.")
            if st.checkbox(f"Yes, delete receipt ID {receipt.id}"):
                try:
                    success = db_manager.delete_receipt(receipt.id)
                    if success:
                        st.success("🗑️ Receipt deleted successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete receipt")
                except Exception as e:
                    st.error(f"❌ Error deleting receipt: {str(e)}")
                    logger.error(f"Receipt deletion error: {str(e)}")


def display_bulk_operations(selected_indices, receipts, db_manager):
    """Display bulk operations interface.
    
    Args:
        selected_indices: List of selected receipt indices
        receipts: List of all receipts
        db_manager: Database manager instance
    """
    st.subheader(f"🔧 Bulk Operations ({len(selected_indices)} receipts selected)")
    
    selected_receipts = [receipts[i] for i in selected_indices]
    
    # Show selected receipts summary
    with st.expander("📋 Selected Receipts", expanded=False):
        for receipt in selected_receipts:
            st.write(f"• **{receipt.vendor}** - ${receipt.amount} on {receipt.transaction_date}")
    
    # Bulk operations options
    operation = st.selectbox(
        "Select Operation",
        options=[
            "Update Category",
            "Update Currency", 
            "Add Description",
            "Update Confidence",
            "Export Selected",
            "Delete Selected"
        ]
    )
    
    if operation == "Update Category":
        new_category = st.selectbox(
            "New Category",
            options=["Food & Dining", "Gas & Fuel", "Shopping", "Entertainment",
                    "Healthcare", "Travel", "Business", "Education", "Other"]
        )
        
        if st.button("🏷️ Apply Category to All Selected"):
            try:
                success_count = 0
                for receipt in selected_receipts:
                    update_data = ReceiptUpdate(category=new_category)
                    if db_manager.update_receipt(receipt.id, update_data):
                        success_count += 1
                
                st.success(f"✅ Updated category for {success_count} receipts!")
                if success_count < len(selected_receipts):
                    st.warning(f"⚠️ {len(selected_receipts) - success_count} updates failed")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Bulk update failed: {str(e)}")
    
    elif operation == "Export Selected":
        if 'data_exporter' not in st.session_state:
            st.session_state.data_exporter = DataExporter()
        
        exporter = st.session_state.data_exporter
        
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox("Format", ["CSV", "JSON"])
        with col2:
            include_meta = st.checkbox("Include Metadata", value=True)
        
        if st.button("📤 Export Selected Receipts"):
            try:
                if export_format == "CSV":
                    data = exporter.export_to_csv(selected_receipts, include_meta)
                    filename = f"selected_receipts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    mime = "text/csv"
                else:
                    data = exporter.export_to_json(selected_receipts, include_meta)
                    filename = f"selected_receipts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    mime = "application/json"
                
                st.download_button(
                    label=f"📥 Download {export_format}",
                    data=data,
                    file_name=filename,
                    mime=mime
                )
                
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
    
    elif operation == "Delete Selected":
        st.warning("⚠️ **WARNING**: This will permanently delete the selected receipts!")
        
        if st.checkbox("I understand this action cannot be undone"):
            if st.button("🗑️ Delete Selected Receipts", type="secondary"):
                try:
                    success_count = 0
                    for receipt in selected_receipts:
                        if db_manager.delete_receipt(receipt.id):
                            success_count += 1
                    
                    st.success(f"🗑️ Deleted {success_count} receipts!")
                    if success_count < len(selected_receipts):
                        st.warning(f"⚠️ {len(selected_receipts) - success_count} deletions failed")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Bulk deletion failed: {str(e)}")


if __name__ == "__main__":
    main()
        
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
