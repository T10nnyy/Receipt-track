"""
Unit tests for database operations in the receipt processing application.
Tests CRUD operations, search functionality, and data integrity.
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from core.database import DatabaseManager
from core.models import Receipt, ReceiptCreate, ReceiptUpdate, SearchFilters


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        db_manager = DatabaseManager(temp_file.name)
        db_manager.initialize_database()
        
        yield db_manager
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def sample_receipt_data(self):
        """Sample receipt data for testing."""
        return ReceiptCreate(
            vendor="Test Store",
            transaction_date=date(2024, 1, 15),
            amount=Decimal("25.99"),
            category="Shopping",
            source_file="test.pdf",
            currency="USD",
            description="Test receipt",
            processing_confidence=0.95
        )
    
    def test_database_initialization(self, temp_db):
        """Test database initialization creates proper schema."""
        # Check if tables exist
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check receipts table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='receipts'
            """)
            assert cursor.fetchone() is not None
            
            # Check indexes exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name='receipts'
            """)
            indexes = cursor.fetchall()
            assert len(indexes) > 0  # Should have multiple indexes
    
    def test_add_receipt(self, temp_db, sample_receipt_data):
        """Test adding a receipt to the database."""
        receipt_id = temp_db.add_receipt(sample_receipt_data)
        
        assert receipt_id is not None
        assert isinstance(receipt_id, int)
        assert receipt_id > 0
    
    def test_get_receipt(self, temp_db, sample_receipt_data):
        """Test retrieving a receipt by ID."""
        # Add receipt first
        receipt_id = temp_db.add_receipt(sample_receipt_data)
        
        # Retrieve receipt
        retrieved = temp_db.get_receipt(receipt_id)
        
        assert retrieved is not None
        assert retrieved.id == receipt_id
        assert retrieved.vendor == sample_receipt_data.vendor
        assert retrieved.amount == sample_receipt_data.amount
        assert retrieved.transaction_date == sample_receipt_data.transaction_date
    
    def test_get_nonexistent_receipt(self, temp_db):
        """Test retrieving a receipt that doesn't exist."""
        retrieved = temp_db.get_receipt(99999)
        assert retrieved is None
    
    def test_get_all_receipts(self, temp_db):
        """Test retrieving all receipts."""
        # Initially should be empty
        receipts = temp_db.get_all_receipts()
        assert len(receipts) == 0
        
        # Add a few receipts
        receipt1 = ReceiptCreate(
            vendor="Store 1",
            transaction_date=date(2024, 1, 1),
            amount=Decimal("10.00"),
            source_file="test1.pdf"
        )
        receipt2 = ReceiptCreate(
            vendor="Store 2", 
            transaction_date=date(2024, 1, 2),
            amount=Decimal("20.00"),
            source_file="test2.pdf"
        )
        
        temp_db.add_receipt(receipt1)
        temp_db.add_receipt(receipt2)
        
        # Should have 2 receipts now
        receipts = temp_db.get_all_receipts()
        assert len(receipts) == 2
        
        # Should be sorted by date descending (newest first)
        assert receipts[0].transaction_date >= receipts[1].transaction_date
    
    def test_get_all_receipts_with_pagination(self, temp_db):
        """Test pagination in get_all_receipts."""
        # Add multiple receipts
        for i in range(5):
            receipt = ReceiptCreate(
                vendor=f"Store {i}",
                transaction_date=date(2024, 1, i + 1),
                amount=Decimal(f"{(i + 1) * 10}.00"),
                source_file=f"test{i}.pdf"
            )
            temp_db.add_receipt(receipt)
        
        # Test pagination
        page1 = temp_db.get_all_receipts(limit=2, offset=0)
        assert len(page1) == 2
        
        page2 = temp_db.get_all_receipts(limit=2, offset=2)
        assert len(page2) == 2
        
        # Ensure different receipts on different pages
        assert page1[0].id != page2[0].id
    
    def test_update_receipt(self, temp_db, sample_receipt_data):
        """Test updating an existing receipt."""
        # Add receipt first
        receipt_id = temp_db.add_receipt(sample_receipt_data)
        
        # Update some fields
        updates = ReceiptUpdate(
            vendor="Updated Store",
            amount=Decimal("30.00"),
            category="Food & Dining"
        )
        
        success = temp_db.update_receipt(receipt_id, updates)
        assert success is True
        
        # Verify updates
        updated_receipt = temp_db.get_receipt(receipt_id)
        assert updated_receipt.vendor == "Updated Store"
        assert updated_receipt.amount == Decimal("30.00")
        assert updated_receipt.category == "Food & Dining"
        
        # Verify unchanged fields
        assert updated_receipt.transaction_date == sample_receipt_data.transaction_date
        assert updated_receipt.source_file == sample_receipt_data.source_file
    
    def test_update_nonexistent_receipt(self, temp_db):
        """Test updating a receipt that doesn't exist."""
        updates = ReceiptUpdate(vendor="Nonexistent")
        success = temp_db.update_receipt(99999, updates)
        assert success is False
    
    def test_update_with_empty_changes(self, temp_db, sample_receipt_data):
        """Test update with no actual changes."""
        receipt_id = temp_db.add_receipt(sample_receipt_data)
        
        # Empty update
        updates = ReceiptUpdate()
        success = temp_db.update_receipt(receipt_id, updates)
        assert success is True  # Should succeed even with no changes
    
    def test_delete_receipt(self, temp_db, sample_receipt_data):
        """Test deleting a receipt."""
        # Add receipt first
        receipt_id = temp_db.add_receipt(sample_receipt_data)
        
        # Verify it exists
        assert temp_db.get_receipt(receipt_id) is not None
        
        # Delete it
        success = temp_db.delete_receipt(receipt_id)
        assert success is True
        
        # Verify it's gone
        assert temp_db.get_receipt(receipt_id) is None
    
    def test_delete_nonexistent_receipt(self, temp_db):
        """Test deleting a receipt that doesn't exist."""
        success = temp_db.delete_receipt(99999)
        assert success is False
    
    def test_search_receipts_by_vendor(self, temp_db):
        """Test searching receipts by vendor name."""
        # Add test receipts
        receipts = [
            ReceiptCreate(vendor="Starbucks", transaction_date=date.today(), amount=Decimal("5.00"), source_file="test1.pdf"),
            ReceiptCreate(vendor="McDonald's", transaction_date=date.today(), amount=Decimal("10.00"), source_file="test2.pdf"),
            ReceiptCreate(vendor="Starbucks Coffee", transaction_date=date.today(), amount=Decimal("6.00"), source_file="test3.pdf")
        ]
        
        for receipt in receipts:
            temp_db.add_receipt(receipt)
        
        # Search for Starbucks
        filters = SearchFilters(vendor_search="Starbucks")
        results = temp_db.search_receipts(filters)
        
        assert len(results) == 2  # Should find both Starbucks receipts
        for result in results:
            assert "Starbucks" in result.vendor
    
    def test_search_receipts_by_amount_range(self, temp_db):
        """Test searching receipts by amount range."""
        # Add receipts with different amounts
        receipts = [
            ReceiptCreate(vendor="Store A", transaction_date=date.today(), amount=Decimal("5.00"), source_file="test1.pdf"),
            ReceiptCreate(vendor="Store B", transaction_date=date.today(), amount=Decimal("15.00"), source_file="test2.pdf"),
            ReceiptCreate(vendor="Store C", transaction_date=date.today(), amount=Decimal("25.00"), source_file="test3.pdf")
        ]
        
        for receipt in receipts:
            temp_db.add_receipt(receipt)
        
        # Search for amounts between 10 and 20
        filters = SearchFilters(min_amount=Decimal("10.00"), max_amount=Decimal("20.00"))
        results = temp_db.search_receipts(filters)
        
        assert len(results) == 1
        assert results[0].amount == Decimal("15.00")
    
    def test_search_receipts_by_date_range(self, temp_db):
        """Test searching receipts by date range."""
        # Add receipts with different dates
        receipts = [
            ReceiptCreate(vendor="Store A", transaction_date=date(2024, 1, 1), amount=Decimal("10.00"), source_file="test1.pdf"),
            ReceiptCreate(vendor="Store B", transaction_date=date(2024, 1, 15), amount=Decimal("10.00"), source_file="test2.pdf"),
            ReceiptCreate(vendor="Store C", transaction_date=date(2024, 2, 1), amount=Decimal("10.00"), source_file="test3.pdf")
        ]
        
        for receipt in receipts:
            temp_db.add_receipt(receipt)
        
        # Search for January receipts
        filters = SearchFilters(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        results = temp_db.search_receipts(filters)
        
        assert len(results) == 2
        for result in results:
            assert result.transaction_date.month == 1
    
    def test_search_receipts_by_category(self, temp_db):
        """Test searching receipts by category."""
        # Add receipts with different categories
        receipts = [
            ReceiptCreate(vendor="Store A", transaction_date=date.today(), amount=Decimal("10.00"), 
                         category="Food & Dining", source_file="test1.pdf"),
            ReceiptCreate(vendor="Store B", transaction_date=date.today(), amount=Decimal("10.00"), 
                         category="Shopping", source_file="test2.pdf"),
            ReceiptCreate(vendor="Store C", transaction_date=date.today(), amount=Decimal("10.00"), 
                         category="Food & Dining", source_file="test3.pdf")
        ]
        
        for receipt in receipts:
            temp_db.add_receipt(receipt)
        
        # Search for Food & Dining receipts
        filters = SearchFilters(category_filter="Food & Dining")
        results = temp_db.search_receipts(filters)
        
        assert len(results) == 2
        for result in results:
            assert result.category == "Food & Dining"
    
    def test_get_receipt_count(self, temp_db):
        """Test getting total receipt count."""
        # Initially should be 0
        assert temp_db.get_receipt_count() == 0
        
        # Add some receipts
        for i in range(3):
            receipt = ReceiptCreate(
                vendor=f"Store {i}",
                transaction_date=date.today(),
                amount=Decimal("10.00"),
                source_file=f"test{i}.pdf"
            )
            temp_db.add_receipt(receipt)
        
        assert temp_db.get_receipt_count() == 3
    
    def test_get_receipt_count_with_filters(self, temp_db):
        """Test getting receipt count with filters."""
        # Add receipts with different categories
        receipts = [
            ReceiptCreate(vendor="Store A", transaction_date=date.today(), amount=Decimal("10.00"), 
                         category="Food & Dining", source_file="test1.pdf"),
            ReceiptCreate(vendor="Store B", transaction_date=date.today(), amount=Decimal("10.00"), 
                         category="Shopping", source_file="test2.pdf"),
            ReceiptCreate(vendor="Store C", transaction_date=date.today(), amount=Decimal("10.00"), 
                         category="Food & Dining", source_file="test3.pdf")
        ]
        
        for receipt in receipts:
            temp_db.add_receipt(receipt)
        
        # Count Food & Dining receipts
        filters = SearchFilters(category_filter="Food & Dining")
        count = temp_db.get_receipt_count(filters)
        
        assert count == 2
    
    def test_get_analytics_data(self, temp_db):
        """Test getting analytics data."""
        # Add some test receipts
        receipts = [
            ReceiptCreate(vendor="Starbucks", transaction_date=date(2024, 1, 1), amount=Decimal("5.00"), 
                         category="Food & Dining", source_file="test1.pdf"),
            ReceiptCreate(vendor="Target", transaction_date=date(2024, 1, 15), amount=Decimal("25.00"), 
                         category="Shopping", source_file="test2.pdf"),
            ReceiptCreate(vendor="Starbucks", transaction_date=date(2024, 2, 1), amount=Decimal("6.00"), 
                         category="Food & Dining", source_file="test3.pdf")
        ]
        
        for receipt in receipts:
            temp_db.add_receipt(receipt)
        
        analytics = temp_db.get_analytics_data()
        
        assert analytics.total_receipts == 3
        assert analytics.total_amount == Decimal("36.00")
        assert analytics.average_amount == Decimal("12.00")
        
        # Check date range
        assert analytics.date_range["earliest"] == date(2024, 1, 1)
        assert analytics.date_range["latest"] == date(2024, 2, 1)
        
        # Check top vendors
        assert len(analytics.top_vendors) == 2
        top_vendor = analytics.top_vendors[0]
        assert top_vendor["vendor"] == "Target"  # Highest spending
        assert top_vendor["total_spent"] == Decimal("25.00")
        
        # Check category breakdown
        assert "Food & Dining" in analytics.category_breakdown
        assert "Shopping" in analytics.category_breakdown
        assert analytics.category_breakdown["Food & Dining"] == Decimal("11.00")
        assert analytics.category_breakdown["Shopping"] == Decimal("25.00")
    
    def test_database_constraints(self, temp_db):
        """Test database constraints are enforced."""
        # Test amount constraint (should be positive)
        with pytest.raises(Exception):  # Should raise database error
            with temp_db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO receipts (vendor, transaction_date, amount, source_file)
                    VALUES (?, ?, ?, ?)
                """, ("Test", date.today().isoformat(), -5.0, "test.pdf"))
                conn.commit()
    
    def test_concurrent_access(self, temp_db):
        """Test concurrent database access."""
        import threading
        import time
        
        results = []
        errors = []
        
        def add_receipt(vendor_id):
            try:
                receipt = ReceiptCreate(
                    vendor=f"Vendor {vendor_id}",
                    transaction_date=date.today(),
                    amount=Decimal("10.00"),
                    source_file=f"test{vendor_id}.pdf"
                )
                receipt_id = temp_db.add_receipt(receipt)
                results.append(receipt_id)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_receipt, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert len(set(results)) == 5  # All IDs should be unique


if __name__ == "__main__":
    pytest.main([__file__])

