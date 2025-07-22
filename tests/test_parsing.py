"""
Unit tests for file processing and data extraction functionality.
Tests OCR, text extraction, and data parsing capabilities.
"""

import pytest
import io
import os
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from core.parsing import FileProcessor
from core.models import ProcessingResult, ReceiptCreate


class TestFileProcessor:
    """Test cases for FileProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create FileProcessor instance for testing."""
        return FileProcessor()
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content as bytes."""
        # This would normally be actual PDF bytes
        # For testing, we'll create mock content
        return b"PDF-1.4 mock content"
    
    @pytest.fixture
    def sample_image_content(self):
        """Sample image content as bytes."""
        # This would normally be actual image bytes
        # For testing, we'll create mock content
        return b"\x89PNG\r\n\x1a\n mock image content"
    
    @pytest.fixture
    def sample_receipt_text(self):
        """Sample receipt text for testing extraction."""
        return """
        STARBUCKS STORE #12345
        123 Main Street
        Seattle, WA 98101
        
        Date: 01/15/2024
        Time: 10:30 AM
        
        1 Grande Latte          $4.85
        1 Blueberry Muffin      $2.95
        
        Subtotal:               $7.80
        Tax:                    $0.85
        Total:                  $8.65
        
        Payment: VISA ****1234
        
        Thank you for visiting!
        """
    
    def test_processor_initialization(self, processor):
        """Test FileProcessor initialization."""
        assert isinstance(processor, FileProcessor)
        assert hasattr(processor, 'SUPPORTED_EXTENSIONS')
        assert hasattr(processor, 'DATE_PATTERNS')
        assert hasattr(processor, 'AMOUNT_PATTERNS')
    
    def test_supported_extensions(self, processor):
        """Test supported file extensions."""
        expected_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        assert processor.SUPPORTED_EXTENSIONS == expected_extensions
    
    def test_validate_file_valid_extensions(self, processor):
        """Test file validation with valid extensions."""
        valid_files = [
            "receipt.pdf",
            "image.png", 
            "scan.jpg",
            "document.jpeg",
            "photo.bmp",
            "file.tiff",
            "img.tif"
        ]
        
        for filename in valid_files:
            assert processor._validate_file(filename) is True
    
    def test_validate_file_invalid_extensions(self, processor):
        """Test file validation with invalid extensions."""
        invalid_files = [
            "document.doc",
            "spreadsheet.xlsx",
            "text.txt",
            "video.mp4",
            "audio.mp3",
            "archive.zip"
        ]
        
        for filename in invalid_files:
            assert processor._validate_file(filename) is False
    
    def test_extract_vendor_basic(self, processor):
        """Test basic vendor extraction."""
        lines = [
            "STARBUCKS COFFEE",
            "123 Main Street",
            "Seattle, WA 98101"
        ]
        
        vendor = processor._extract_vendor(lines)
        assert vendor == "STARBUCKS COFFEE"
    
    def test_extract_vendor_with_noise(self, processor):
        """Test vendor extraction with noisy data."""
        lines = [
            "123456789",  # Should skip numeric line
            "WALMART SUPERCENTER",
            "4567 Broadway Ave"
        ]
        
        vendor = processor._extract_vendor(lines)
        assert vendor == "WALMART SUPERCENTER"
    
    def test_extract_vendor_cleaned(self, processor):
        """Test vendor extraction with cleaning."""
        lines = [
            "TARGET@#$% STORE!!!",
            "Additional info"
        ]
        
        vendor = processor._extract_vendor(lines)
        assert "TARGET" in vendor
        # Should clean special characters
        assert "@#$%" not in vendor
    
    def test_extract_date_formats(self, processor):
        """Test date extraction with various formats."""
        test_cases = [
            ("Date: 01/15/2024", "2024-01-15"),
            ("2024-01-15 10:30", "2024-01-15"), 
            ("Jan 15, 2024", "2024-01-15"),
            ("15 January 2024", "2024-01-15"),
            ("Transaction Date: 12/25/2023", "2023-12-25")
        ]
        
        for text, expected in test_cases:
            result = processor._extract_date(text)
            assert result == expected, f"Failed for text: {text}"
    
    def test_extract_date_invalid_formats(self, processor):
        """Test date extraction with invalid formats."""
        invalid_texts = [
            "No date here",
            "13/45/2024",  # Invalid day
            "25/13/2024",  # Invalid month
            "Random text with numbers 123456"
        ]
        
        for text in invalid_texts:
            result = processor._extract_date(text)
            assert result is None, f"Should not extract date from: {text}"
    
    def test_extract_amount_basic(self, processor):
        """Test basic amount extraction."""
        test_cases = [
            ("Total: $25.99", "25.99"),
            ("Amount: €15.50", "15.50"),
            ("£12.75 paid", "12.75"),
            ("¥1000", "1000"),
            ("Grand Total $123.45", "123.45")
        ]
        
        for text, expected in test_cases:
            result = processor._extract_amount(text)
            assert result == expected, f"Failed for text: {text}"
    
    def test_extract_amount_with_commas(self, processor):
        """Test amount extraction with comma separators."""
        test_cases = [
            ("Total: $1,234.56", "1234.56"),
            ("Amount: $12,345.00", "12345.00"),
            ("€2,500.99", "2500.99")
        ]
        
        for text, expected in test_cases:
            result = processor._extract_amount(text)
            assert result == expected, f"Failed for text: {text}"
    
    def test_extract_amount_multiple_values(self, processor):
        """Test amount extraction chooses largest value."""
        text = """
        Item 1: $5.99
        Item 2: $3.50
        Tax: $0.85
        Total: $10.34
        """
        
        result = processor._extract_amount(text)
        assert result == "10.34"  # Should pick the largest amount
    
    def test_extract_amount_invalid(self, processor):
        """Test amount extraction with invalid amounts."""
        invalid_texts = [
            "No amounts here",
            "Price: free",
            "Cost: zero dollars",
            "Random numbers 123ABC"
        ]
        
        for text in invalid_texts:
            result = processor._extract_amount(text)
            assert result is None, f"Should not extract amount from: {text}"
    
    def test_extract_data_from_text_complete(self, processor, sample_receipt_text):
        """Test complete data extraction from receipt text."""
        extracted = processor._extract_data_from_text(sample_receipt_text)
        
        assert extracted['vendor'] is not None
        assert "STARBUCKS" in extracted['vendor'].upper()
        assert extracted['date'] == "2024-01-15"
        assert extracted['amount'] == "8.65"
        assert extracted['currency'] == "USD"
    
    def test_extract_data_currency_detection(self, processor):
        """Test currency detection from text."""
        test_cases = [
            ("Total: €25.99", "EUR"),
            ("Amount: £15.50", "GBP"), 
            ("Cost: ¥1000", "JPY"),
            ("Price: $10.00", "USD")  # Default case
        ]
        
        for text, expected_currency in test_cases:
            extracted = processor._extract_data_from_text(text)
            assert extracted['currency'] == expected_currency
    
    def test_create_receipt_from_data_valid(self, processor):
        """Test creating receipt from valid extracted data."""
        extracted_data = {
            'vendor': 'Test Store',
            'amount': '25.99',
            'date': '2024-01-15',
            'currency': 'USD',
            'confidence': 0.95
        }
        
        receipt = processor._create_receipt_from_data(extracted_data, "test.pdf")
        
        assert receipt is not None
        assert isinstance(receipt, ReceiptCreate)
        assert receipt.vendor == 'Test Store'
        assert receipt.amount == Decimal('25.99')
        assert receipt.transaction_date == date(2024, 1, 15)
        assert receipt.currency == 'USD'
        assert receipt.source_file == "test.pdf"
    
    def test_create_receipt_from_data_missing_vendor(self, processor):
        """Test creating receipt with missing vendor."""
        extracted_data = {
            'vendor': None,
            'amount': '25.99',
            'date': '2024-01-15'
        }
        
        receipt = processor._create_receipt_from_data(extracted_data, "test.pdf")
        assert receipt is None
    
    def test_create_receipt_from_data_missing_amount(self, processor):
        """Test creating receipt with missing amount."""
        extracted_data = {
            'vendor': 'Test Store',
            'amount': None,
            'date': '2024-01-15'
        }
        
        receipt = processor._create_receipt_from_data(extracted_data, "test.pdf")
        assert receipt is None
    
    def test_create_receipt_from_data_invalid_amount(self, processor):
        """Test creating receipt with invalid amount."""
        extracted_data = {
            'vendor': 'Test Store',
            'amount': 'invalid',
            'date': '2024-01-15'
        }
        
        receipt = processor._create_receipt_from_data(extracted_data, "test.pdf")
        assert receipt is None
    
    def test_create_receipt_from_data_fallback_date(self, processor):
        """Test creating receipt with fallback date."""
        extracted_data = {
            'vendor': 'Test Store',
            'amount': '25.99',
            'date': None  # Missing date
        }
        
        receipt = processor._create_receipt_from_data(extracted_data, "test.pdf")
        
        assert receipt is not None
        assert receipt.transaction_date == date.today()  # Should use current date
    
    def test_calculate_confidence_high(self, processor):
        """Test confidence calculation with complete data."""
        extracted_data = {
            'vendor': 'Test Store',
            'amount': '25.99',
            'date': '2024-01-15'
        }
        
        raw_text = "This is a good quality receipt text with sufficient content for analysis."
        
        confidence = processor._calculate_confidence(extracted_data, raw_text)
        assert confidence >= 0.8  # Should be high confidence
    
    def test_calculate_confidence_low(self, processor):
        """Test confidence calculation with incomplete data."""
        extracted_data = {
            'vendor': None,
            'amount': '25.99',
            'date': None
        }
        
        raw_text = "???"  # Poor quality text
        
        confidence = processor._calculate_confidence(extracted_data, raw_text)
        assert confidence < 0.5  # Should be low confidence
    
    def test_calculate_confidence_with_artifacts(self, processor):
        """Test confidence calculation with OCR artifacts."""
        extracted_data = {
            'vendor': 'Test Store',
            'amount': '25.99',
            'date': '2024-01-15'
        }
        
        raw_text = "Good text but with ||| artifacts ~~~ and ??? noise"
        
        confidence = processor._calculate_confidence(extracted_data, raw_text)
        # Should be penalized for artifacts but still reasonable
        assert 0.3 <= confidence <= 0.8
    
    @patch('fitz.open')
    def test_process_pdf_success(self, mock_fitz_open, processor, sample_receipt_text):
        """Test successful PDF processing."""
        # Mock PyMuPDF objects
        mock_page = Mock()
        mock_page.get_text.return_value = sample_receipt_text
        
        mock_doc = Mock()
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.close = Mock()
        
        mock_fitz_open.return_value = mock_doc
        
        result = processor._process_pdf(b"mock pdf content", "test.pdf")
        
        assert result.success is True
        assert result.receipt is not None
        assert result.confidence_score == 0.9  # High confidence for direct text
    
    @patch('fitz.open')
    def test_process_pdf_empty(self, mock_fitz_open, processor):
        """Test PDF processing with empty document."""
        mock_doc = Mock()
        mock_doc.page_count = 0
        mock_fitz_open.return_value = mock_doc
        
        result = processor._process_pdf(b"mock pdf content", "test.pdf")
        
        assert result.success is False
        assert "PDF file is empty" in result.errors
    
    @patch('fitz.open')
    def test_process_pdf_no_text_fallback(self, mock_fitz_open, processor):
        """Test PDF processing fallback to OCR when no text found."""
        # Mock PyMuPDF objects with no text
        mock_page = Mock()
        mock_page.get_text.return_value = ""  # No text found
        mock_page.get_pixmap.return_value.tobytes.return_value = b"mock image data"
        
        mock_doc = Mock()
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.close = Mock()
        
        mock_fitz_open.return_value = mock_doc
        
        with patch.object(processor, '_process_image') as mock_process_image:
            mock_process_image.return_value = ProcessingResult(success=True, errors=[])
            
            result = processor._process_pdf(b"mock pdf content", "test.pdf")
            
            # Should call image processing as fallback
            mock_process_image.assert_called_once()
    
    @patch('pytesseract.image_to_string')
    @patch('cv2.cvtColor')
    @patch('PIL.Image.open')
    def test_process_image_success(self, mock_image_open, mock_cv2, mock_tesseract, 
                                 processor, sample_receipt_text):
        """Test successful image processing with OCR."""
        # Skip if tesseract not available
        if not processor.tesseract_available:
            pytest.skip("Tesseract not available")
        
        # Mock PIL Image
        mock_img = Mock()
        mock_image_open.return_value = mock_img
        
        # Mock CV2 operations
        mock_cv2.return_value = Mock()
        
        # Mock Tesseract OCR
        mock_tesseract.return_value = sample_receipt_text
        
        with patch.object(processor, '_preprocess_image') as mock_preprocess:
            mock_preprocess.return_value = Mock()
            
            result = processor._process_image(b"mock image content", "test.jpg")
            
            assert result.success is True
            assert result.receipt is not None
            assert result.confidence_score is not None
    
    def test_process_image_no_tesseract(self, processor):
        """Test image processing when Tesseract is not available."""
        # Force tesseract unavailable
        processor.tesseract_available = False
        
        result = processor._process_image(b"mock image content", "test.jpg")
        
        assert result.success is False
        assert "OCR not available" in result.errors[0]
    
    @patch('pytesseract.image_to_string')
    @patch('cv2.cvtColor')
    @patch('PIL.Image.open')
    def test_process_image_no_text_extracted(self, mock_image_open, mock_cv2, 
                                           mock_tesseract, processor):
        """Test image processing when no text is extracted."""
        if not processor.tesseract_available:
            pytest.skip("Tesseract not available")
        
        # Mock empty OCR result
        mock_tesseract.return_value = ""
        
        mock_img = Mock()
        mock_image_open.return_value = mock_img
        mock_cv2.return_value = Mock()
        
        with patch.object(processor, '_preprocess_image') as mock_preprocess:
            mock_preprocess.return_value = Mock()
            
            result = processor._process_image(b"mock image content", "test.jpg")
            
            assert result.success is False
            assert "No text could be extracted" in result.errors[0]
    
    def test_process_file_unsupported_format(self, processor):
        """Test processing unsupported file format."""
        result = processor.process_file(b"content", "document.docx")
        
        assert result.success is False
        assert "Unsupported file type" in result.errors[0]
    
    def test_process_file_pdf_routing(self, processor):
        """Test file processing routes PDF to PDF processor."""
        with patch.object(processor, '_process_pdf') as mock_process_pdf:
            mock_process_pdf.return_value = ProcessingResult(success=True, errors=[])
            
            processor.process_file(b"pdf content", "test.pdf")
            
            mock_process_pdf.assert_called_once_with(b"pdf content", "test.pdf")
    
    def test_process_file_image_routing(self, processor):
        """Test file processing routes images to image processor."""
        with patch.object(processor, '_process_image') as mock_process_image:
            mock_process_image.return_value = ProcessingResult(success=True, errors=[])
            
            processor.process_file(b"image content", "test.jpg")
            
            mock_process_image.assert_called_once_with(b"image content", "test.jpg")
    
    def test_process_file_timing(self, processor):
        """Test that processing time is recorded."""
        with patch.object(processor, '_process_pdf') as mock_process_pdf:
            mock_result = ProcessingResult(success=True, errors=[])
            mock_process_pdf.return_value = mock_result
            
            result = processor.process_file(b"content", "test.pdf")
            
            assert result.processing_time is not None
            assert result.processing_time > 0
    
    def test_preprocess_image_basic(self, processor):
        """Test basic image preprocessing."""
        import numpy as np
        
        # Create mock image array
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with patch('cv2.cvtColor') as mock_cvtColor, \
             patch('cv2.GaussianBlur') as mock_blur, \
             patch('cv2.adaptiveThreshold') as mock_threshold, \
             patch('cv2.morphologyEx') as mock_morph:
            
            # Configure mocks to return appropriate arrays
            mock_cvtColor.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            mock_blur.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            mock_threshold.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            mock_morph.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            
            result = processor._preprocess_image(mock_image)
            
            # Verify preprocessing steps were called
            mock_cvtColor.assert_called()
            mock_blur.assert_called()
            mock_threshold.assert_called()
            mock_morph.assert_called()
            
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])

