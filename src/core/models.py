"""
Data models using Pydantic for receipt processing application.
Provides validation and type checking for receipt data.
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator
import re

class Receipt(BaseModel):
    """Main receipt model with complete data structure."""
    
    id: Optional[int] = Field(None, description="Database primary key")
    vendor: str = Field(..., min_length=1, max_length=200, description="Vendor/merchant name")
    transaction_date: date = Field(..., description="Date of transaction")
    amount: Decimal = Field(..., gt=0, description="Transaction amount (must be positive)")
    category: Optional[str] = Field("Other", max_length=100, description="Expense category")
    source_file: str = Field(..., description="Original filename")
    currency: str = Field("USD", max_length=3, description="Currency code")
    description: Optional[str] = Field(None, max_length=500, description="Additional notes")
    processing_confidence: Optional[float] = Field(None, ge=0, le=1, description="OCR confidence score")
    created_at: Optional[datetime] = Field(None, description="Record creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        """Validate amount is positive and reasonable."""
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > Decimal('999999.99'):
            raise ValueError('Amount seems unreasonably large')
        return v
    
    @field_validator('vendor')
    @classmethod
    def validate_vendor(cls, v):
        """Clean and validate vendor name."""
        if not v or not v.strip():
            raise ValueError('Vendor name cannot be empty')
        # Clean vendor name
        cleaned = re.sub(r'[^\w\s\-\.]', '', v.strip())
        if not cleaned:
            raise ValueError('Vendor name must contain valid characters')
        return cleaned
    
    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v):
        """Validate currency code format."""
        if not re.match(r'^[A-Z]{3}$', v):
            raise ValueError('Currency must be a 3-letter code (e.g., USD, EUR)')
        return v
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        """Validate and normalize category."""
        if v is None:
            return "Other"
        valid_categories = [
            "Food & Dining", "Gas & Fuel", "Shopping", "Entertainment",
            "Healthcare", "Travel", "Business", "Education", "Other"
        ]
        # Try to match to valid categories (case-insensitive)
        for category in valid_categories:
            if v.lower() == category.lower():
                return category
        return "Other"
    
    model_config = {
        "json_encoders": {
            Decimal: lambda v: float(v),
            date: lambda v: v.isoformat(),
            datetime: lambda v: v.isoformat()
        },
        "json_schema_extra": {
            "example": {
                "vendor": "Starbucks Coffee",
                "transaction_date": "2024-01-15",
                "amount": 4.85,
                "category": "Food & Dining",
                "source_file": "receipt_001.pdf",
                "currency": "USD",
                "description": "Morning coffee",
                "processing_confidence": 0.95
            }
        }
    }

class ReceiptCreate(BaseModel):
    """Model for creating new receipts."""
    
    vendor: str = Field(..., min_length=1, max_length=200)
    transaction_date: date = Field(...)
    amount: Decimal = Field(..., gt=0)
    category: Optional[str] = Field("Other", max_length=100)
    source_file: str = Field(...)
    currency: str = Field("USD", max_length=3)
    description: Optional[str] = Field(None, max_length=500)
    processing_confidence: Optional[float] = Field(None, ge=0, le=1)

class ReceiptUpdate(BaseModel):
    """Model for updating existing receipts."""
    
    vendor: Optional[str] = Field(None, min_length=1, max_length=200)
    transaction_date: Optional[date] = Field(None)
    amount: Optional[Decimal] = Field(None, gt=0)
    category: Optional[str] = Field(None, max_length=100)
    currency: Optional[str] = Field(None, max_length=3)
    description: Optional[str] = Field(None, max_length=500)
    processing_confidence: Optional[float] = Field(None, ge=0, le=1)

class SearchFilters(BaseModel):
    """Model for search and filter operations."""
    
    vendor_search: Optional[str] = Field(None, description="Search by vendor name")
    category_filter: Optional[str] = Field(None, description="Filter by category")
    min_amount: Optional[Decimal] = Field(None, ge=0, description="Minimum amount")
    max_amount: Optional[Decimal] = Field(None, gt=0, description="Maximum amount")
    start_date: Optional[date] = Field(None, description="Start date range")
    end_date: Optional[date] = Field(None, description="End date range")
    currency_filter: Optional[str] = Field(None, max_length=3, description="Currency filter")
    
    # Note: Model-level validation for field relationships would need model_validator in V2
    # For now, we'll handle this validation in the business logic layer

class ProcessingResult(BaseModel):
    """Model for file processing results."""
    
    success: bool = Field(..., description="Whether processing was successful")
    extracted_data: Optional[Dict[str, Any]] = Field(None, description="Raw extracted data")
    receipt: Optional[ReceiptCreate] = Field(None, description="Validated receipt model")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Overall confidence")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "extracted_data": {
                    "vendor": "Target Store",
                    "amount": "29.99",
                    "date": "01/15/2024"
                },
                "receipt": {
                    "vendor": "Target Store",
                    "transaction_date": "2024-01-15",
                    "amount": 29.99,
                    "category": "Shopping",
                    "source_file": "receipt_001.jpg"
                },
                "errors": [],
                "warnings": ["Date format was ambiguous"],
                "processing_time": 2.45,
                "confidence_score": 0.92
            }
        }

class AnalyticsData(BaseModel):
    """Model for analytics data."""
    
    total_receipts: int = Field(..., ge=0, description="Total number of receipts")
    total_amount: Decimal = Field(..., ge=0, description="Total spending amount")
    average_amount: Decimal = Field(..., ge=0, description="Average transaction amount")
    date_range: Dict[str, Optional[date]] = Field(..., description="Date range of data")
    top_vendors: List[Dict[str, Any]] = Field(default_factory=list, description="Top vendors by spending")
    category_breakdown: Dict[str, Decimal] = Field(default_factory=dict, description="Spending by category")
    monthly_trends: List[Dict[str, Any]] = Field(default_factory=list, description="Monthly spending trends")
    
    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
            date: lambda v: v.isoformat() if v else None
        }
