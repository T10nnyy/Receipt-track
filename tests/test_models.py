"""
Unit tests for Pydantic models in the receipt processing application.
Tests data validation, type checking, and model behavior.
"""

import pytest
from datetime import date, datetime
from decimal import Decimal
from pydantic import ValidationError

from core.models import (
    Receipt, ReceiptCreate, ReceiptUpdate, SearchFilters, 
    ProcessingResult, AnalyticsData
)


class TestReceiptModel:
    """Test cases for the Receipt model."""
    
    def test_valid_receipt_creation(self):
        """Test creating a valid receipt."""
        receipt = Receipt(
            id=1,
            vendor="Test Vendor",
            transaction_date=date(2024, 1, 15),
            amount=Decimal("25.99"),
            category="Food & Dining",
            source_file="test.pdf",
            currency="USD",
            description="Test receipt",
            processing_confidence=0.95
        )
        
        assert receipt.id == 1
        assert receipt.vendor == "Test Vendor"
        assert receipt.amount == Decimal("25.99")
        assert receipt.currency == "USD"
        assert receipt.category == "Food & Dining"
    
    def test_receipt_amount_validation(self):
        """Test amount validation rules."""
        # Test positive amount
        with pytest.raises(ValidationError) as exc_info:
            Receipt(
                vendor="Test",
                transaction_date=date.today(),
                amount=Decimal("-5.00"),
                source_file="test.pdf"
            )
        assert "Amount must be positive" in str(exc_info.value)
        
        # Test unreasonably large amount
        with pytest.raises(ValidationError) as exc_info:
            Receipt(
                vendor="Test",
                transaction_date=date.today(),
                amount=Decimal("9999999.99"),
                source_file="test.pdf"
            )
        assert "Amount seems unreasonably large" in str(exc_info.value)
    
    def test_vendor_validation(self):
        """Test vendor name validation."""
        # Test empty vendor
        with pytest.raises(ValidationError) as exc_info:
            Receipt(
                vendor="",
                transaction_date=date.today(),
                amount=Decimal("10.00"),
                source_file="test.pdf"
            )
        assert "Vendor name cannot be empty" in str(exc_info.value)
        
        # Test vendor with only special characters
        with pytest.raises(ValidationError) as exc_info:
            Receipt(
                vendor="@#$%^&*()",
                transaction_date=date.today(),
                amount=Decimal("10.00"),
                source_file="test.pdf"
            )
        assert "Vendor name must contain valid characters" in str(exc_info.value)
    
    def test_currency_validation(self):
        """Test currency code validation."""
        with pytest.raises(ValidationError) as exc_info:
            Receipt(
                vendor="Test",
                transaction_date=date.today(),
                amount=Decimal("10.00"),
                source_file="test.pdf",
                currency="INVALID"
            )
        assert "Currency must be a 3-letter code" in str(exc_info.value)
    
    def test_category_validation(self):
        """Test category validation and normalization."""
        receipt = Receipt(
            vendor="Test",
            transaction_date=date.today(),
            amount=Decimal("10.00"),
            source_file="test.pdf",
            category="food & dining"  # lowercase
        )
        assert receipt.category == "Food & Dining"
        
        # Test invalid category defaults to "Other"
        receipt = Receipt(
            vendor="Test",
            transaction_date=date.today(),
            amount=Decimal("10.00"),
            source_file="test.pdf",
            category="Invalid Category"
        )
        assert receipt.category == "Other"
    
    def test_receipt_json_serialization(self):
        """Test JSON serialization of receipt."""
        receipt = Receipt(
            id=1,
            vendor="Test Vendor",
            transaction_date=date(2024, 1, 15),
            amount=Decimal("25.99"),
            source_file="test.pdf"
        )
        
        json_data = receipt.dict()
        assert json_data["amount"] == 25.99  # Decimal converted to float
        assert json_data["transaction_date"] == date(2024, 1, 15)


class TestReceiptCreateModel:
    """Test cases for the ReceiptCreate model."""
    
    def test_valid_receipt_create(self):
        """Test creating a valid ReceiptCreate object."""
        receipt_create = ReceiptCreate(
            vendor="Test Store",
            transaction_date=date.today(),
            amount=Decimal("15.50"),
            category="Shopping",
            source_file="receipt.jpg",
            currency="USD",
            processing_confidence=0.85
        )
        
        assert receipt_create.vendor == "Test Store"
        assert receipt_create.amount == Decimal("15.50")
        assert receipt_create.category == "Shopping"
    
    def test_receipt_create_validation(self):
        """Test validation rules are applied from Receipt model."""
        with pytest.raises(ValidationError):
            ReceiptCreate(
                vendor="",
                transaction_date=date.today(),
                amount=Decimal("15.50"),
                source_file="test.pdf"
            )


class TestReceiptUpdateModel:
    """Test cases for the ReceiptUpdate model."""
    
    def test_partial_update(self):
        """Test partial updates with ReceiptUpdate."""
        update = ReceiptUpdate(
            vendor="Updated Vendor",
            amount=Decimal("20.00")
        )
        
        # Only specified fields should be set
        assert update.vendor == "Updated Vendor"
        assert update.amount == Decimal("20.00")
        assert update.transaction_date is None
        assert update.category is None
    
    def test_empty_update(self):
        """Test creating empty update object."""
        update = ReceiptUpdate()
        
        # All fields should be None
        update_dict = update.dict(exclude_unset=True)
        assert len(update_dict) == 0


class TestSearchFiltersModel:
    """Test cases for the SearchFilters model."""
    
    def test_valid_search_filters(self):
        """Test creating valid search filters."""
        filters = SearchFilters(
            vendor_search="Starbucks",
            min_amount=Decimal("5.00"),
            max_amount=Decimal("50.00"),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            category_filter="Food & Dining"
        )
        
        assert filters.vendor_search == "Starbucks"
        assert filters.min_amount == Decimal("5.00")
        assert filters.max_amount == Decimal("50.00")
    
    def test_amount_range_validation(self):
        """Test amount range validation."""
        with pytest.raises(ValidationError) as exc_info:
            SearchFilters(
                min_amount=Decimal("50.00"),
                max_amount=Decimal("25.00")  # Max less than min
            )
        assert "Maximum amount must be greater than minimum amount" in str(exc_info.value)
    
    def test_date_range_validation(self):
        """Test date range validation."""
        with pytest.raises(ValidationError) as exc_info:
            SearchFilters(
                start_date=date(2024, 12, 31),
                end_date=date(2024, 1, 1)  # End before start
            )
        assert "End date must be after start date" in str(exc_info.value)


class TestProcessingResultModel:
    """Test cases for the ProcessingResult model."""
    
    def test_successful_processing_result(self):
        """Test creating successful processing result."""
        receipt_create = ReceiptCreate(
            vendor="Test",
            transaction_date=date.today(),
            amount=Decimal("10.00"),
            source_file="test.pdf"
        )
        
        result = ProcessingResult(
            success=True,
            extracted_data={"vendor": "Test", "amount": "10.00"},
            receipt=receipt_create,
            errors=[],
            warnings=["Date format ambiguous"],
            processing_time=2.5,
            confidence_score=0.95
        )
        
        assert result.success is True
        assert result.receipt is not None
        assert len(result.warnings) == 1
        assert result.confidence_score == 0.95
    
    def test_failed_processing_result(self):
        """Test creating failed processing result."""
        result = ProcessingResult(
            success=False,
            errors=["Could not extract text", "Invalid file format"],
            warnings=[],
            processing_time=1.0
        )
        
        assert result.success is False
        assert len(result.errors) == 2
        assert result.receipt is None


class TestAnalyticsDataModel:
    """Test cases for the AnalyticsData model."""
    
    def test_analytics_data_creation(self):
        """Test creating analytics data."""
        analytics = AnalyticsData(
            total_receipts=100,
            total_amount=Decimal("1500.00"),
            average_amount=Decimal("15.00"),
            date_range={"earliest": date(2024, 1, 1), "latest": date(2024, 12, 31)},
            top_vendors=[
                {"vendor": "Starbucks", "total_spent": Decimal("200.00"), "receipt_count": 20}
            ],
            category_breakdown={"Food & Dining": Decimal("500.00"), "Shopping": Decimal("300.00")},
            monthly_trends=[
                {"month": "2024-01", "total_amount": Decimal("150.00"), "receipt_count": 10}
            ]
        )
        
        assert analytics.total_receipts == 100
        assert analytics.total_amount == Decimal("1500.00")
        assert len(analytics.top_vendors) == 1
        assert len(analytics.category_breakdown) == 2
    
    def test_analytics_json_serialization(self):
        """Test JSON serialization of analytics data."""
        analytics = AnalyticsData(
            total_receipts=10,
            total_amount=Decimal("100.00"),
            average_amount=Decimal("10.00"),
            date_range={"earliest": date(2024, 1, 1), "latest": None},
            top_vendors=[],
            category_breakdown={},
            monthly_trends=[]
        )
        
        json_data = analytics.dict()
        assert json_data["total_amount"] == 100.0  # Decimal converted to float
        assert json_data["date_range"]["earliest"] == "2024-01-01"
        assert json_data["date_range"]["latest"] is None


# Test fixtures and utilities
@pytest.fixture
def sample_receipt():
    """Fixture providing a sample receipt for testing."""
    return Receipt(
        id=1,
        vendor="Sample Store",
        transaction_date=date(2024, 1, 15),
        amount=Decimal("25.99"),
        category="Shopping",
        source_file="sample.pdf",
        currency="USD",
        description="Sample receipt for testing",
        processing_confidence=0.90
    )


@pytest.fixture
def sample_receipt_create():
    """Fixture providing a sample ReceiptCreate for testing."""
    return ReceiptCreate(
        vendor="New Store",
        transaction_date=date.today(),
        amount=Decimal("15.75"),
        source_file="new_receipt.jpg"
    )


if __name__ == "__main__":
    pytest.main([__file__])

