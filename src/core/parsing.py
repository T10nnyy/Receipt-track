"""
File processing and data extraction for receipt processing application.
Optimized for Railway deployment with proper Tesseract configuration.
"""

import os
import io
import re
import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import logging
import shutil
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from PIL import Image
import tempfile

logger = logging.getLogger(__name__)

class FileProcessor:
    """Handles file processing and data extraction from receipts."""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Common date patterns for extraction
    DATE_PATTERNS = [
        r'\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b',  # MM/DD/YYYY or MM-DD-YYYY
        r'\b(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\b',  # YYYY/MM/DD or YYYY-MM-DD
        r'\b(\d{1,2})[\s\-\/](\w{3,9})[\s\-\/](\d{4})\b',  # DD Month YYYY
        r'\b(\w{3,9})[\s\-\/](\d{1,2})[\s\-\/](\d{4})\b',  # Month DD YYYY
    ]
    
    # Enhanced amount patterns with more currency support
    AMOUNT_PATTERNS = [
        r'(?:total|amount|sum|subtotal|grand\s*total)[\s:]*[\$€£¥₹]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'[\$€£¥₹]\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # Currency prefix
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)[\s]*[\$€£¥₹]',  # Currency suffix
        r'\b(\d{1,3}(?:,\d{3})*\.\d{2})\b',  # Simple decimal pattern
    ]
    
    # Vendor extraction patterns
    VENDOR_PATTERNS = [
        r'^([A-Za-z][A-Za-z0-9\s&\.\-\']{2,50})',
        r'([A-Z][A-Z\s&\.]{3,30})',
        r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
    ]
    
    def __init__(self):
        """Initialize the file processor with Railway-optimized configuration."""
        self.logger = logger
        self.tesseract_available = False
        
        # Configure Tesseract for Railway deployment
        self._configure_tesseract_railway()
        
        # Initialize currency and language detection (with fallbacks)
        self._initialize_processors()
    
    def _configure_tesseract_railway(self):
        """Configure Tesseract OCR specifically for Railway deployment."""
        try:
            # Railway typically installs tesseract in standard Linux paths
            tesseract_paths = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                shutil.which("tesseract")  # Check PATH
            ]
            
            for path in tesseract_paths:
                if path and os.path.isfile(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
            
            # Test Tesseract
            version = pytesseract.get_tesseract_version()
            self.tesseract_available = True
            self.logger.info(f"Tesseract OCR configured successfully. Version: {version}")
            
            # Set additional OCR configuration for better accuracy
            os.environ['OMP_THREAD_LIMIT'] = '1'  # Prevent OpenMP issues in containers
            
        except Exception as e:
            self.tesseract_available = False
            self.logger.error(f"Tesseract OCR configuration failed: {str(e)}")
            self.logger.error("Make sure Tesseract is installed in your Dockerfile")
    
    def _initialize_processors(self):
        """Initialize currency and language processors with fallbacks."""
        try:
            # Try to import advanced processors
            from .currency_detector import CurrencyDetector, MultiLanguageProcessor
            self.currency_detector = CurrencyDetector()
            self.language_processor = MultiLanguageProcessor()
            self.logger.info("Advanced processing components initialized")
        except ImportError as e:
            self.logger.warning(f"Advanced processors not available, using fallbacks: {e}")
            self.currency_detector = None
            self.language_processor = None
    
    def process_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process uploaded file and extract receipt data.
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        start_time = datetime.now()
        
        try:
            # Validate file
            if not self._validate_file(filename):
                return {
                    'success': False,
                    'errors': [f"Unsupported file type: {Path(filename).suffix}"],
                    'processing_time': 0.0
                }
            
            # Route to appropriate processor based on file type
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.pdf':
                result = self._process_pdf(file_content, filename)
            else:
                result = self._process_image(file_content, filename)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result['processing_time'] = processing_time
            
            self.logger.info(f"Processed {filename} in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process file {filename}: {str(e)}")
            return {
                'success': False,
                'errors': [f"Processing failed: {str(e)}"],
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _validate_file(self, filename: str) -> bool:
        """Validate if file is supported."""
        return Path(filename).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def _process_pdf(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process PDF file using PyMuPDF."""
        try:
            # Open PDF from memory
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            if pdf_document.page_count == 0:
                return {
                    'success': False,
                    'errors': ["PDF file is empty"]
                }
            
            # Extract text from first page
            page = pdf_document[0]
            text = page.get_text()
            
            if not text.strip():
                # Try OCR fallback
                if not self.tesseract_available:
                    pdf_document.close()
                    return {
                        'success': False,
                        'errors': ["No text in PDF and OCR not available"]
                    }
                
                self.logger.info("No text in PDF, attempting OCR")
                
                # Convert PDF page to image for OCR
                mat = fitz.Matrix(2.0, 2.0)  # Higher resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                pdf_document.close()
                return self._process_image(img_data, filename, is_pdf_fallback=True)
            
            pdf_document.close()
            
            # Extract data from text
            extracted_data = self._extract_data_from_text(text)
            
            return {
                'success': True,
                'extracted_data': extracted_data,
                'confidence_score': 0.9,  # High confidence for direct text
                'raw_text': text
            }
            
        except Exception as e:
            self.logger.error(f"PDF processing error: {str(e)}")
            return {
                'success': False,
                'errors': [f"PDF processing failed: {str(e)}"]
            }
    
    def _process_image(self, file_content: bytes, filename: str, is_pdf_fallback: bool = False) -> Dict[str, Any]:
        """Process image file using OCR."""
        if not self.tesseract_available:
            return {
                'success': False,
                'errors': ["OCR not available - Tesseract not properly installed"]
            }
        
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess for better OCR
            processed_image = self._preprocess_image(cv_image)
            
            # Perform OCR with optimized settings
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/$€£¥₹-: '
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            if not text.strip():
                return {
                    'success': False,
                    'errors': ["No text could be extracted from image"]
                }
            
            # Extract data from OCR text
            extracted_data = self._extract_data_from_text(text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(extracted_data, text)
            
            return {
                'success': True,
                'extracted_data': extracted_data,
                'confidence_score': confidence,
                'raw_text': text
            }
            
        except Exception as e:
            self.logger.error(f"Image processing error: {str(e)}")
            return {
                'success': False,
                'errors': [f"Image processing failed: {str(e)}"]
            }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize if too small
            height, width = gray.shape
            if height < 300 or width < 300:
                scale_factor = max(300 / height, 300 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {str(e)}")
            # Return original grayscale as fallback
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
    
    def _extract_data_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured data from raw text."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        extracted = {
            'raw_text': text,
            'vendor': None,
            'date': None,
            'amount': None,
            'currency': 'USD',
            'category': 'Other',
            'confidence_factors': {}
        }
        
        # Extract vendor (first meaningful line)
        extracted['vendor'] = self._extract_vendor(lines[:5])
        
        # Extract date
        extracted['date'] = self._extract_date(text)
        
        # Extract amount
        extracted['amount'] = self._extract_amount(text)
        
        # Detect currency if advanced processor available
        if self.currency_detector:
            try:
                currency, confidence = self.currency_detector.detect_currency(text)
                extracted['currency'] = currency
                extracted['confidence_factors']['currency'] = confidence
            except:
                pass
        
        # Simple category detection
        extracted['category'] = self._extract_category(extracted['vendor'] or '')
        
        return extracted
    
    def _extract_vendor(self, lines: List[str]) -> Optional[str]:
        """Extract vendor name from first few lines."""
        for line in lines:
            # Skip lines with too many digits (likely addresses/phones)
            if len([c for c in line if c.isdigit()]) > len(line) // 2:
                continue
            
            # Look for business name patterns
            for pattern in self.VENDOR_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    vendor = match.group(1).strip()
                    if len(vendor) > 2 and not vendor.isdigit():
                        return vendor
            
            # Fallback: use first substantial line
            if len(line) > 3 and not line.isdigit():
                cleaned = re.sub(r'[^\w\s&\.\-\']', '', line).strip()
                if cleaned and len(cleaned) > 2:
                    return cleaned
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract transaction date from text."""
        month_names = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
            'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
            'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
            'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
            'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }
        
        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 3:
                        # Handle different date formats
                        if match[2].isdigit() and len(match[2]) == 4:  # Year last
                            year = int(match[2])
                            
                            if match[0].isdigit() and match[1].isdigit():
                                # Numeric month/day
                                month, day = int(match[0]), int(match[1])
                                if 1 <= month <= 12 and 1 <= day <= 31:
                                    return f"{year:04d}-{month:02d}-{day:02d}"
                            else:
                                # Month name
                                month_str = match[1].lower() if match[1].lower() in month_names else match[0].lower()
                                day_str = match[0] if match[1].lower() in month_names else match[1]
                                
                                if month_str in month_names and day_str.isdigit():
                                    month = month_names[month_str]
                                    day = int(day_str)
                                    if 1 <= day <= 31:
                                        return f"{year:04d}-{month:02d}-{day:02d}"
                        
                        elif match[0].isdigit() and len(match[0]) == 4:  # Year first
                            year, month, day = int(match[0]), int(match[1]), int(match[2])
                            if 1 <= month <= 12 and 1 <= day <= 31:
                                return f"{year:04d}-{month:02d}-{day:02d}"
                
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_amount(self, text: str) -> Optional[str]:
        """Extract transaction amount from text."""
        amounts = []
        
        # Look for total/amount patterns first (more likely to be final amount)
        for pattern in self.AMOUNT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean and parse amount
                    amount_str = match.replace(',', '').replace(' ', '')
                    amount = float(amount_str)
                    
                    # Filter reasonable amounts
                    if 0.01 <= amount <= 999999.99:
                        amounts.append(amount)
                        
                except (ValueError, TypeError):
                    continue
        
        if amounts:
            # Return the largest amount (likely the total)
            return str(max(amounts))
        
        return None
    
    def _extract_category(self, vendor: str) -> str:
        """Simple category detection based on vendor name."""
        if not vendor:
            return "Other"
        
        vendor_lower = vendor.lower()
        
        # Simple keyword-based categorization
        categories = {
            'Food & Dining': ['restaurant', 'cafe', 'coffee', 'pizza', 'burger', 'kitchen', 'grill', 'food', 'bar', 'pub'],
            'Gas & Fuel': ['gas', 'fuel', 'station', 'shell', 'exxon', 'bp', 'chevron', 'petrol'],
            'Shopping': ['store', 'shop', 'market', 'mall', 'walmart', 'target', 'amazon'],
            'Healthcare': ['hospital', 'clinic', 'pharmacy', 'medical', 'dental', 'health'],
            'Entertainment': ['theater', 'cinema', 'movie', 'game', 'entertainment'],
            'Travel': ['hotel', 'airline', 'airport', 'taxi', 'uber', 'booking']
        }
        
        for category, keywords in categories.items():
            if any(keyword in vendor_lower for keyword in keywords):
                return category
        
        return "Other"
    
    def _calculate_confidence(self, extracted_data: Dict[str, Any], raw_text: str) -> float:
        """Calculate processing confidence score."""
        confidence = 0.0
        
        # Base confidence from data completeness
        if extracted_data.get('vendor'):
            confidence += 0.3
        if extracted_data.get('amount'):
            confidence += 0.4
        if extracted_data.get('date'):
            confidence += 0.2
        
        # Text quality factors
        text_length = len(raw_text.strip())
        if text_length > 50:
            confidence += 0.05
        if text_length > 200:
            confidence += 0.05
        
        # Penalty for OCR artifacts
        artifacts = ['|||', '~~~', 'lll', '???', '***']
        artifact_count = sum(raw_text.count(artifact) for artifact in artifacts)
        confidence -= min(0.1, artifact_count * 0.02)
        
        return max(0.0, min(1.0, confidence))
