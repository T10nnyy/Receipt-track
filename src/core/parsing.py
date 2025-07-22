"""
File processing and data extraction for receipt processing application.
Handles multiple file formats with OCR and text extraction capabilities.
"""

import os
import io
import re
import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import logging
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from PIL import Image
import tempfile

from .models import ReceiptCreate, ProcessingResult
from .currency_detector import CurrencyDetector, MultiLanguageProcessor

logger = logging.getLogger(__name__)

class FileProcessor:
    """Handles file processing and data extraction from receipts."""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Common date patterns for extraction
    DATE_PATTERNS = [
        r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b',  # MM/DD/YYYY or MM-DD-YYYY
        r'\b(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})\b',  # YYYY/MM/DD or YYYY-MM-DD
        r'\b(\d{1,2})[\s\-\/](\w{3,9})[\s\-\/](\d{4})\b',  # DD Month YYYY
        r'\b(\w{3,9})[\s\-\/](\d{1,2})[\s\-\/](\d{4})\b',  # Month DD YYYY
    ]
    
    # Amount patterns with various currency symbols
    AMOUNT_PATTERNS = [
        r'[\$€£¥]\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $10.99, €15,50
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)[\s]*[\$€£¥]',  # 10.99$
        r'\btotal[\s:]*[\$€£¥]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # Total: $10.99
        r'\bamount[\s:]*[\$€£¥]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # Amount: 10.99
        r'(\d+\.\d{2})',  # Simple decimal pattern
    ]
    
    # Vendor extraction - usually found in first few lines
    VENDOR_PATTERNS = [
        r'^([A-Za-z][A-Za-z0-9\s&\.\-]{2,50})',  # First line with business name characteristics
        r'([A-Z][A-Z\s&\.]{5,30})',  # All caps business name
    ]
    
    def __init__(self):
        """Initialize the file processor."""
        self.logger = logger
        
        # Initialize advanced processing components
        self.currency_detector = CurrencyDetector()
        self.language_processor = MultiLanguageProcessor()
        
        # Configure Tesseract if available
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            self.logger.info("Tesseract OCR is available")
        except Exception as e:
            self.tesseract_available = False
            self.logger.warning(f"Tesseract OCR not available: {e}")
    
    def process_file(self, file_content: bytes, filename: str) -> ProcessingResult:
        """Process uploaded file and extract receipt data.
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename
            
        Returns:
            ProcessingResult with extraction results
        """
        start_time = datetime.now()
        
        try:
            # Validate file
            if not self._validate_file(filename):
                return ProcessingResult(
                    success=False,
                    errors=[f"Unsupported file type: {Path(filename).suffix}"]
                )
            
            # Route to appropriate processor based on file type
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.pdf':
                result = self._process_pdf(file_content, filename)
            else:
                result = self._process_image(file_content, filename)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            self.logger.info(f"Processed {filename} in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process file {filename}: {str(e)}")
            return ProcessingResult(
                success=False,
                errors=[f"Processing failed: {str(e)}"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_file(self, filename: str) -> bool:
        """Validate if file is supported.
        
        Args:
            filename: File name to validate
            
        Returns:
            True if supported, False otherwise
        """
        return Path(filename).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def _process_pdf(self, file_content: bytes, filename: str) -> ProcessingResult:
        """Process PDF file using PyMuPDF.
        
        Args:
            file_content: PDF file content
            filename: Original filename
            
        Returns:
            ProcessingResult with extraction results
        """
        try:
            # Open PDF from memory
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            if pdf_document.page_count == 0:
                return ProcessingResult(
                    success=False,
                    errors=["PDF file is empty"]
                )
            
            # Extract text from first page (receipts are usually single page)
            page = pdf_document[0]
            text = page.get_text()
            
            if not text.strip():
                # If no text found, try OCR on PDF page as image
                self.logger.info("No text found in PDF, attempting OCR")
                
                # Convert PDF page to image
                mat = fitz.Matrix(2.0, 2.0)  # Increase resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Process as image
                return self._process_image(img_data, filename, is_pdf_fallback=True)
            
            pdf_document.close()
            
            # Extract data from text
            extracted_data = self._extract_data_from_text(text)
            
            # Validate and create receipt
            receipt = self._create_receipt_from_data(extracted_data, filename)
            
            if receipt:
                return ProcessingResult(
                    success=True,
                    extracted_data=extracted_data,
                    receipt=receipt,
                    confidence_score=0.9  # High confidence for direct text extraction
                )
            else:
                return ProcessingResult(
                    success=False,
                    extracted_data=extracted_data,
                    errors=["Could not extract required data from PDF"]
                )
                
        except Exception as e:
            self.logger.error(f"PDF processing error: {str(e)}")
            return ProcessingResult(
                success=False,
                errors=[f"PDF processing failed: {str(e)}"]
            )
    
    def _process_image(self, file_content: bytes, filename: str, is_pdf_fallback: bool = False) -> ProcessingResult:
        """Process image file using OCR.
        
        Args:
            file_content: Image file content
            filename: Original filename
            is_pdf_fallback: Whether this is a fallback from PDF processing
            
        Returns:
            ProcessingResult with extraction results
        """
        if not self.tesseract_available:
            return ProcessingResult(
                success=False,
                errors=["OCR not available - Tesseract not installed"]
            )
        
        try:
            # Load image
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(cv_image)
            
            # Perform OCR
            text = pytesseract.image_to_string(processed_image, lang='eng')
            
            if not text.strip():
                return ProcessingResult(
                    success=False,
                    errors=["No text could be extracted from image"]
                )
            
            # Extract data from OCR text
            extracted_data = self._extract_data_from_text(text)
            
            # Create receipt
            receipt = self._create_receipt_from_data(extracted_data, filename)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(extracted_data, text)
            
            if receipt:
                return ProcessingResult(
                    success=True,
                    extracted_data=extracted_data,
                    receipt=receipt,
                    confidence_score=confidence
                )
            else:
                return ProcessingResult(
                    success=False,
                    extracted_data=extracted_data,
                    errors=["Could not extract required data from image"],
                    confidence_score=confidence
                )
                
        except Exception as e:
            self.logger.error(f"Image processing error: {str(e)}")
            return ProcessingResult(
                success=False,
                errors=[f"Image processing failed: {str(e)}"]
            )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed, using original: {str(e)}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def _extract_data_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured data from raw text with advanced language and currency detection.
        
        Args:
            text: Raw text from document
            
        Returns:
            Dictionary with extracted data
        """
        extracted = {
            'raw_text': text,
            'vendor': None,
            'date': None,
            'amount': None,
            'currency': 'USD',
            'language': 'en',
            'confidence_factors': {}
        }
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Detect language first
        detected_lang, lang_confidence = self.language_processor.detect_language(text)
        extracted['language'] = detected_lang
        extracted['confidence_factors']['language'] = lang_confidence
        
        # Detect currency with language context
        detected_currency, currency_confidence = self.currency_detector.detect_currency(
            text, context={'language': detected_lang}
        )
        extracted['currency'] = detected_currency
        extracted['confidence_factors']['currency'] = currency_confidence
        
        # Extract using language-specific patterns
        lang_extracts = self.language_processor.extract_with_language_context(text, detected_lang)
        
        # Extract vendor (usually in first few lines)
        extracted['vendor'] = self._extract_vendor(lines[:5])
        
        # Extract date with language context
        extracted['date'] = self._extract_date_enhanced(text, detected_lang)
        
        # Extract amount with currency normalization
        extracted['amount'] = self._extract_amount_enhanced(text, detected_currency)
        
        # Try to extract category based on vendor patterns
        extracted['category'] = self._extract_category(extracted['vendor'] or '', detected_lang)
        
        return extracted
    
    def _extract_vendor(self, lines: List[str]) -> Optional[str]:
        """Extract vendor name from text lines.
        
        Args:
            lines: List of text lines (usually first few)
            
        Returns:
            Extracted vendor name or None
        """
        for line in lines:
            # Skip lines that look like addresses or phone numbers
            if re.search(r'\d{3,}', line) and len(re.findall(r'\d', line)) > 3:
                continue
                
            # Look for business name patterns
            for pattern in self.VENDOR_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    vendor = match.group(1).strip()
                    if len(vendor) > 2 and not vendor.isdigit():
                        return vendor
            
            # If no pattern matches, use first substantial line
            if len(line) > 5 and not line.isdigit():
                # Clean up common OCR artifacts
                cleaned = re.sub(r'[^\w\s&\.\-]', '', line).strip()
                if cleaned and len(cleaned) > 2:
                    return cleaned
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract transaction date from text.
        
        Args:
            text: Raw text content
            
        Returns:
            Extracted date string or None
        """
        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Parse different date formats
                    if len(match) == 3:
                        if match[2].isdigit() and len(match[2]) == 4:  # Year is 4 digits
                            if match[0].isdigit() and match[1].isdigit():
                                # MM/DD/YYYY or DD/MM/YYYY
                                month, day, year = int(match[0]), int(match[1]), int(match[2])
                                if month <= 12 and day <= 31:
                                    return f"{year:04d}-{month:02d}-{day:02d}"
                                elif day <= 12 and month <= 31:
                                    # Swap if DD/MM format
                                    return f"{year:04d}-{day:02d}-{month:02d}"
                            else:
                                # Handle month names
                                month_names = {
                                    'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
                                    'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
                                    'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
                                    'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                                    'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
                                    'dec': 12, 'december': 12
                                }
                                
                                if match[1].lower() in month_names:
                                    month = month_names[match[1].lower()]
                                    day = int(match[0])
                                    year = int(match[2])
                                elif match[0].lower() in month_names:
                                    month = month_names[match[0].lower()]
                                    day = int(match[1])
                                    year = int(match[2])
                                else:
                                    continue
                                
                                if 1 <= month <= 12 and 1 <= day <= 31:
                                    return f"{year:04d}-{month:02d}-{day:02d}"
                        
                        elif match[0].isdigit() and len(match[0]) == 4:  # YYYY/MM/DD
                            year, month, day = int(match[0]), int(match[1]), int(match[2])
                            if 1 <= month <= 12 and 1 <= day <= 31:
                                return f"{year:04d}-{month:02d}-{day:02d}"
                                
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_amount_enhanced(self, text: str, currency: str) -> Optional[str]:
        """Extract and normalize transaction amount from text.
        
        Args:
            text: Raw text content
            currency: Detected currency code
            
        Returns:
            Extracted and normalized amount string or None
        """
        # Try enhanced currency-aware extraction first
        for pattern in self.AMOUNT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    amount_str = match[0]
                else:
                    amount_str = match
                
                # Use currency detector to normalize amount
                normalized = self.currency_detector.normalize_amount(amount_str, currency)
                if normalized and normalized > Decimal('0'):
                    return str(normalized)
        
        # Fallback to basic extraction
        return self._extract_amount(text)
    
    def _extract_date_enhanced(self, text: str, language: str) -> Optional[str]:
        """Extract transaction date with language context.
        
        Args:
            text: Raw text content
            language: Detected language code
            
        Returns:
            Extracted date string or None
        """
        # First try the standard extraction
        date_result = self._extract_date(text)
        if date_result:
            return date_result
        
        # Try language-specific patterns
        if language == 'es':  # Spanish
            spanish_months = {
                'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
                'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
                'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
            }
            for month_name, month_num in spanish_months.items():
                pattern = rf'\b(\d{{1,2}})\s+de\s+{month_name}\s+de\s+(\d{{4}})\b'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    day, year = match.groups()
                    return f"{year}-{month_num}-{int(day):02d}"
        
        elif language == 'fr':  # French
            french_months = {
                'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04',
                'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08',
                'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
            }
            for month_name, month_num in french_months.items():
                pattern = rf'\b(\d{{1,2}})\s+{month_name}\s+(\d{{4}})\b'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    day, year = match.groups()
                    return f"{year}-{month_num}-{int(day):02d}"
        
        elif language == 'de':  # German
            german_months = {
                'januar': '01', 'februar': '02', 'märz': '03', 'april': '04',
                'mai': '05', 'juni': '06', 'juli': '07', 'august': '08',
                'september': '09', 'oktober': '10', 'november': '11', 'dezember': '12'
            }
            for month_name, month_num in german_months.items():
                pattern = rf'\b(\d{{1,2}})\.\s*{month_name}\s+(\d{{4}})\b'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    day, year = match.groups()
                    return f"{year}-{month_num}-{int(day):02d}"
        
        return None
    
    def _extract_category(self, vendor: str, language: str) -> Optional[str]:
        """Extract or predict category based on vendor name and language.
        
        Args:
            vendor: Vendor name
            language: Detected language
            
        Returns:
            Predicted category or None
        """
        if not vendor:
            return None
        
        vendor_lower = vendor.lower()
        
        # Common category keywords by language
        category_keywords = {
            'en': {
                'Food & Dining': ['restaurant', 'cafe', 'coffee', 'pizza', 'burger', 'deli', 'bakery', 'bar', 'pub', 'bistro', 'kitchen', 'grill', 'food', 'dining', 'starbucks', 'mcdonald', 'subway'],
                'Gas & Fuel': ['gas', 'fuel', 'station', 'shell', 'exxon', 'bp', 'chevron', 'mobil', 'petrol'],
                'Shopping': ['store', 'shop', 'market', 'mall', 'outlet', 'walmart', 'target', 'costco', 'amazon', 'retail'],
                'Healthcare': ['hospital', 'clinic', 'pharmacy', 'medical', 'dental', 'doctor', 'health'],
                'Entertainment': ['theater', 'cinema', 'movie', 'concert', 'game', 'entertainment', 'netflix', 'spotify'],
                'Travel': ['hotel', 'airline', 'airport', 'taxi', 'uber', 'lyft', 'rental', 'booking'],
                'Business': ['office', 'supplies', 'service', 'consulting', 'software', 'tech']
            },
            'es': {
                'Food & Dining': ['restaurante', 'café', 'pizzería', 'bar', 'cocina', 'comida'],
                'Gas & Fuel': ['gasolina', 'combustible', 'estación'],
                'Shopping': ['tienda', 'mercado', 'centro comercial'],
                'Healthcare': ['hospital', 'clínica', 'farmacia', 'médico', 'salud'],
                'Entertainment': ['teatro', 'cine', 'concierto', 'entretenimiento'],
                'Travel': ['hotel', 'aerolínea', 'aeropuerto', 'taxi', 'viaje']
            },
            'fr': {
                'Food & Dining': ['restaurant', 'café', 'pizzeria', 'bar', 'cuisine', 'nourriture'],
                'Gas & Fuel': ['essence', 'carburant', 'station'],
                'Shopping': ['magasin', 'marché', 'centre commercial'],
                'Healthcare': ['hôpital', 'clinique', 'pharmacie', 'médecin', 'santé'],
                'Entertainment': ['théâtre', 'cinéma', 'concert', 'divertissement'],
                'Travel': ['hôtel', 'compagnie aérienne', 'aéroport', 'taxi', 'voyage']
            }
        }
        
        lang_categories = category_keywords.get(language, category_keywords['en'])
        
        for category, keywords in lang_categories.items():
            for keyword in keywords:
                if keyword in vendor_lower:
                    return category
        
        return "Other"
    
    def _extract_amount(self, text: str) -> Optional[str]:
        """Extract transaction amount from text.
        
        Args:
            text: Raw text content
            
        Returns:
            Extracted amount string or None
        """
        amounts = []
        
        for pattern in self.AMOUNT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean amount string
                    amount_str = match.replace(',', '').replace(' ', '')
                    amount = float(amount_str)
                    
                    # Filter reasonable amounts (not too small or large)
                    if 0.01 <= amount <= 99999.99:
                        amounts.append(amount)
                        
                except (ValueError, TypeError):
                    continue
        
        if amounts:
            # Return the largest amount found (likely the total)
            return str(max(amounts))
        
        return None
    
    def _create_receipt_from_data(self, extracted_data: Dict[str, Any], filename: str) -> Optional[ReceiptCreate]:
        """Create ReceiptCreate object from extracted data.
        
        Args:
            extracted_data: Dictionary with extracted data
            filename: Original filename
            
        Returns:
            ReceiptCreate object or None if validation fails
        """
        try:
            # Validate required fields
            if not extracted_data.get('vendor'):
                self.logger.warning("No vendor found in extracted data")
                return None
            
            if not extracted_data.get('amount'):
                self.logger.warning("No amount found in extracted data")
                return None
            
            # Parse date
            transaction_date = None
            if extracted_data.get('date'):
                try:
                    transaction_date = datetime.fromisoformat(extracted_data['date']).date()
                except ValueError:
                    self.logger.warning(f"Invalid date format: {extracted_data['date']}")
            
            if not transaction_date:
                # Use current date as fallback
                transaction_date = date.today()
                self.logger.info("Using current date as fallback")
            
            # Parse amount
            try:
                amount = Decimal(str(extracted_data['amount']))
            except (InvalidOperation, ValueError):
                self.logger.warning(f"Invalid amount: {extracted_data['amount']}")
                return None
            
            # Create receipt
            receipt = ReceiptCreate(
                vendor=extracted_data['vendor'],
                transaction_date=transaction_date,
                amount=amount,
                category="Other",  # Default category
                source_file=filename,
                currency=extracted_data.get('currency', 'USD'),
                processing_confidence=extracted_data.get('confidence', 0.7)
            )
            
            return receipt
            
        except Exception as e:
            self.logger.error(f"Failed to create receipt from data: {str(e)}")
            return None
    
    def _calculate_confidence(self, extracted_data: Dict[str, Any], raw_text: str) -> float:
        """Calculate processing confidence score.
        
        Args:
            extracted_data: Extracted data dictionary
            raw_text: Original text from OCR
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.0
        
        # Base confidence from data completeness
        if extracted_data.get('vendor'):
            confidence += 0.3
        if extracted_data.get('amount'):
            confidence += 0.4
        if extracted_data.get('date'):
            confidence += 0.2
        
        # Bonus for text quality
        text_length = len(raw_text.strip())
        if text_length > 50:
            confidence += 0.05
        if text_length > 200:
            confidence += 0.05
        
        # Penalty for OCR artifacts
        artifacts = ['|||', '~~~', '...', '???']
        for artifact in artifacts:
            if artifact in raw_text:
                confidence -= 0.05
        
        return max(0.0, min(1.0, confidence))
