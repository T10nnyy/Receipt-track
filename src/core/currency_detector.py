"""
Currency detection and multi-language support for receipt processing.
Provides automatic currency detection and multi-language text processing capabilities.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from decimal import Decimal

logger = logging.getLogger(__name__)

class CurrencyDetector:
    """Detects currencies from receipt text and provides currency-specific processing."""
    
    # Currency symbols and their codes
    CURRENCY_SYMBOLS = {
        '$': ['USD', 'CAD', 'AUD', 'NZD', 'SGD', 'HKD'],
        '€': ['EUR'],
        '£': ['GBP'],
        '¥': ['JPY', 'CNY'],
        '₹': ['INR'],
        '₽': ['RUB'],
        '₩': ['KRW'],
        '₪': ['ILS'],
        '₦': ['NGN'],
        '₨': ['PKR', 'LKR'],
        '₫': ['VND'],
        '₡': ['CRC'],
        '₱': ['PHP'],
        '₺': ['TRY'],
        'R': ['ZAR'],
        'kr': ['SEK', 'NOK', 'DKK'],
        'zł': ['PLN'],
        'Kč': ['CZK'],
        'Ft': ['HUF'],
        'lei': ['RON'],
        'BGN': ['BGN'],
        'CHF': ['CHF']
    }
    
    # Currency keywords in different languages
    CURRENCY_KEYWORDS = {
        'USD': ['dollar', 'dollars', 'usd', 'us dollar', 'american dollar'],
        'EUR': ['euro', 'euros', 'eur', 'european'],
        'GBP': ['pound', 'pounds', 'gbp', 'sterling', 'british pound'],
        'JPY': ['yen', 'jpy', 'japanese yen'],
        'CNY': ['yuan', 'rmb', 'renminbi', 'chinese yuan'],
        'INR': ['rupee', 'rupees', 'inr', 'indian rupee'],
        'CAD': ['canadian dollar', 'cad', 'canadian'],
        'AUD': ['australian dollar', 'aud', 'australian'],
        'CHF': ['franc', 'francs', 'chf', 'swiss franc'],
        'SEK': ['krona', 'kronor', 'sek', 'swedish krona'],
        'NOK': ['krone', 'kroner', 'nok', 'norwegian krone'],
        'DKK': ['krone', 'kroner', 'dkk', 'danish krone']
    }
    
    # Regional patterns for number formatting
    REGIONAL_NUMBER_FORMATS = {
        'US': {'decimal': '.', 'thousands': ','},  # 1,234.56
        'EU': {'decimal': ',', 'thousands': '.'},  # 1.234,56
        'CH': {'decimal': '.', 'thousands': "'"},  # 1'234.56
        'IN': {'decimal': '.', 'thousands': ','},  # 1,23,456.78 (special Indian format)
    }
    
    def __init__(self):
        """Initialize the currency detector."""
        self.logger = logger
        
        # Build reverse lookup for symbols
        self._symbol_to_currencies = {}
        for symbol, currencies in self.CURRENCY_SYMBOLS.items():
            self._symbol_to_currencies[symbol] = currencies
    
    def detect_currency(self, text: str, context: Optional[Dict] = None) -> Tuple[str, float]:
        """Detect currency from text content.
        
        Args:
            text: Text content to analyze
            context: Additional context (location, language, etc.)
            
        Returns:
            Tuple of (currency_code, confidence_score)
        """
        text_lower = text.lower()
        detected_currencies = {}
        
        # 1. Look for currency symbols
        symbol_matches = self._detect_by_symbols(text)
        for currency, confidence in symbol_matches.items():
            detected_currencies[currency] = detected_currencies.get(currency, 0) + confidence
        
        # 2. Look for currency keywords
        keyword_matches = self._detect_by_keywords(text_lower)
        for currency, confidence in keyword_matches.items():
            detected_currencies[currency] = detected_currencies.get(currency, 0) + confidence
        
        # 3. Look for currency codes (ISO 4217)
        code_matches = self._detect_by_codes(text_lower)
        for currency, confidence in code_matches.items():
            detected_currencies[currency] = detected_currencies.get(currency, 0) + confidence
        
        # 4. Apply context-based adjustments
        if context:
            detected_currencies = self._apply_context_adjustments(detected_currencies, context)
        
        # 5. Determine most likely currency
        if not detected_currencies:
            return "USD", 0.1  # Default fallback with low confidence
        
        # Sort by confidence and return top match
        best_currency = max(detected_currencies.items(), key=lambda x: x[1])
        
        # Normalize confidence to 0-1 range
        max_confidence = min(best_currency[1], 1.0)
        
        self.logger.info(f"Detected currency: {best_currency[0]} with confidence {max_confidence:.2f}")
        return best_currency[0], max_confidence
    
    def _detect_by_symbols(self, text: str) -> Dict[str, float]:
        """Detect currency by symbols."""
        detected = {}
        
        for symbol, currencies in self.CURRENCY_SYMBOLS.items():
            if symbol in text:
                # Count occurrences for confidence
                count = text.count(symbol)
                confidence = min(0.8 * count, 0.9)  # Max 0.9 confidence from symbols
                
                # If multiple currencies use same symbol, distribute confidence
                for currency in currencies:
                    detected[currency] = confidence / len(currencies)
        
        return detected
    
    def _detect_by_keywords(self, text_lower: str) -> Dict[str, float]:
        """Detect currency by keywords."""
        detected = {}
        
        for currency, keywords in self.CURRENCY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Exact match gets higher confidence
                    if f" {keyword} " in f" {text_lower} ":
                        confidence = 0.7
                    else:
                        confidence = 0.5
                    
                    detected[currency] = max(detected.get(currency, 0), confidence)
        
        return detected
    
    def _detect_by_codes(self, text_lower: str) -> Dict[str, float]:
        """Detect currency by ISO codes."""
        detected = {}
        
        # Look for 3-letter currency codes
        iso_pattern = r'\b([A-Z]{3})\b'
        matches = re.findall(iso_pattern, text_lower.upper())
        
        known_codes = set()
        for currencies in self.CURRENCY_SYMBOLS.values():
            known_codes.update(currencies)
        known_codes.update(self.CURRENCY_KEYWORDS.keys())
        
        for match in matches:
            if match in known_codes:
                detected[match] = 0.8  # High confidence for explicit codes
        
        return detected
    
    def _apply_context_adjustments(self, detected: Dict[str, float], 
                                 context: Dict) -> Dict[str, float]:
        """Apply context-based adjustments to detection confidence."""
        adjusted = detected.copy()
        
        # Country/location context
        if 'country' in context:
            country_currencies = {
                'US': 'USD', 'USA': 'USD', 'United States': 'USD',
                'GB': 'GBP', 'UK': 'GBP', 'United Kingdom': 'GBP',
                'DE': 'EUR', 'FR': 'EUR', 'IT': 'EUR', 'ES': 'EUR',
                'JP': 'JPY', 'Japan': 'JPY',
                'CN': 'CNY', 'China': 'CNY',
                'IN': 'INR', 'India': 'INR',
                'CA': 'CAD', 'Canada': 'CAD',
                'AU': 'AUD', 'Australia': 'AUD',
                'CH': 'CHF', 'Switzerland': 'CHF'
            }
            
            country = context['country'].upper()
            if country in country_currencies:
                currency = country_currencies[country]
                adjusted[currency] = adjusted.get(currency, 0) + 0.3
        
        # Language context
        if 'language' in context:
            language_currencies = {
                'en': ['USD', 'GBP', 'CAD', 'AUD'],
                'de': ['EUR', 'CHF'],
                'fr': ['EUR', 'CHF', 'CAD'],
                'es': ['EUR'],
                'it': ['EUR'],
                'ja': ['JPY'],
                'zh': ['CNY'],
                'hi': ['INR'],
                'sv': ['SEK'],
                'no': ['NOK'],
                'da': ['DKK']
            }
            
            lang = context['language'].lower()
            if lang in language_currencies:
                for currency in language_currencies[lang]:
                    adjusted[currency] = adjusted.get(currency, 0) + 0.2
        
        return adjusted
    
    def normalize_amount(self, amount_str: str, currency: str, 
                        detected_format: Optional[str] = None) -> Optional[Decimal]:
        """Normalize amount string based on currency and regional format.
        
        Args:
            amount_str: Raw amount string
            currency: Detected currency code
            detected_format: Detected regional format
            
        Returns:
            Normalized Decimal amount or None if parsing fails
        """
        try:
            # Clean the amount string
            cleaned = re.sub(r'[^\d.,\'-]', '', amount_str.strip())
            
            if not cleaned:
                return None
            
            # Determine format based on currency if not provided
            if not detected_format:
                detected_format = self._get_default_format_for_currency(currency)
            
            # Parse based on detected format
            if detected_format == 'EU':
                # European format: 1.234,56
                if ',' in cleaned and '.' in cleaned:
                    # Both separators present
                    last_comma = cleaned.rfind(',')
                    last_dot = cleaned.rfind('.')
                    
                    if last_comma > last_dot:
                        # Comma is decimal separator
                        cleaned = cleaned.replace('.', '').replace(',', '.')
                    else:
                        # Dot is decimal separator
                        cleaned = cleaned.replace(',', '')
                elif ',' in cleaned:
                    # Only comma - could be thousands or decimal
                    parts = cleaned.split(',')
                    if len(parts) == 2 and len(parts[1]) <= 2:
                        # Likely decimal separator
                        cleaned = cleaned.replace(',', '.')
                    else:
                        # Likely thousands separator
                        cleaned = cleaned.replace(',', '')
            
            elif detected_format == 'CH':
                # Swiss format: 1'234.56
                cleaned = cleaned.replace("'", "")
            
            elif detected_format == 'IN':
                # Indian format: 1,23,456.78
                # Remove all commas except the last one before decimal
                if '.' in cleaned:
                    parts = cleaned.split('.')
                    parts[0] = parts[0].replace(',', '')
                    cleaned = '.'.join(parts)
                else:
                    cleaned = cleaned.replace(',', '')
            
            else:
                # US format: 1,234.56 (default)
                if ',' in cleaned and '.' in cleaned:
                    # Remove thousands separators (commas)
                    parts = cleaned.split('.')
                    parts[0] = parts[0].replace(',', '')
                    cleaned = '.'.join(parts)
                elif ',' in cleaned:
                    # Only comma - check if it's thousands or decimal
                    parts = cleaned.split(',')
                    if len(parts) == 2 and len(parts[1]) <= 2:
                        # Likely decimal separator (non-US format)
                        cleaned = cleaned.replace(',', '.')
                    else:
                        # Thousands separator
                        cleaned = cleaned.replace(',', '')
            
            return Decimal(cleaned)
            
        except (InvalidOperation, ValueError) as e:
            self.logger.warning(f"Failed to normalize amount '{amount_str}': {e}")
            return None
    
    def _get_default_format_for_currency(self, currency: str) -> str:
        """Get default regional format for currency."""
        format_map = {
            'USD': 'US', 'CAD': 'US', 'GBP': 'US', 'AUD': 'US',
            'EUR': 'EU', 'SEK': 'EU', 'NOK': 'EU', 'DKK': 'EU',
            'CHF': 'CH',
            'INR': 'IN', 'PKR': 'IN', 'LKR': 'IN'
        }
        return format_map.get(currency, 'US')
    
    def get_currency_info(self, currency_code: str) -> Dict[str, str]:
        """Get detailed information about a currency.
        
        Args:
            currency_code: 3-letter currency code
            
        Returns:
            Dictionary with currency information
        """
        currency_info = {
            'USD': {'name': 'US Dollar', 'symbol': '$', 'decimal_places': 2},
            'EUR': {'name': 'Euro', 'symbol': '€', 'decimal_places': 2},
            'GBP': {'name': 'British Pound', 'symbol': '£', 'decimal_places': 2},
            'JPY': {'name': 'Japanese Yen', 'symbol': '¥', 'decimal_places': 0},
            'CNY': {'name': 'Chinese Yuan', 'symbol': '¥', 'decimal_places': 2},
            'INR': {'name': 'Indian Rupee', 'symbol': '₹', 'decimal_places': 2},
            'CAD': {'name': 'Canadian Dollar', 'symbol': '$', 'decimal_places': 2},
            'AUD': {'name': 'Australian Dollar', 'symbol': '$', 'decimal_places': 2},
            'CHF': {'name': 'Swiss Franc', 'symbol': 'CHF', 'decimal_places': 2},
            'SEK': {'name': 'Swedish Krona', 'symbol': 'kr', 'decimal_places': 2},
            'NOK': {'name': 'Norwegian Krone', 'symbol': 'kr', 'decimal_places': 2},
            'DKK': {'name': 'Danish Krone', 'symbol': 'kr', 'decimal_places': 2}
        }
        
        return currency_info.get(currency_code, {
            'name': currency_code,
            'symbol': currency_code,
            'decimal_places': 2
        })


class MultiLanguageProcessor:
    """Handles multi-language text processing for receipts."""
    
    # Language-specific patterns and keywords
    LANGUAGE_PATTERNS = {
        'en': {
            'total_keywords': ['total', 'amount', 'sum', 'balance', 'charge'],
            'date_keywords': ['date', 'time', 'day'],
            'tax_keywords': ['tax', 'vat', 'gst', 'sales tax']
        },
        'es': {
            'total_keywords': ['total', 'suma', 'cantidad', 'importe'],
            'date_keywords': ['fecha', 'día'],
            'tax_keywords': ['impuesto', 'iva', 'impuestos']
        },
        'fr': {
            'total_keywords': ['total', 'montant', 'somme'],
            'date_keywords': ['date', 'jour'],
            'tax_keywords': ['taxe', 'tva', 'impôt']
        },
        'de': {
            'total_keywords': ['gesamt', 'summe', 'betrag'],
            'date_keywords': ['datum', 'tag'],
            'tax_keywords': ['steuer', 'mwst', 'mehrwertsteuer']
        },
        'it': {
            'total_keywords': ['totale', 'somma', 'importo'],
            'date_keywords': ['data', 'giorno'],
            'tax_keywords': ['tassa', 'iva', 'imposta']
        },
        'zh': {
            'total_keywords': ['总计', '合计', '总额', '金额'],
            'date_keywords': ['日期', '时间', '日'],
            'tax_keywords': ['税', '增值税', '税费']
        },
        'ja': {
            'total_keywords': ['合計', '総額', '金額'],
            'date_keywords': ['日付', '日', '時間'],
            'tax_keywords': ['税', '消費税', '税金']
        }
    }
    
    def __init__(self):
        """Initialize the multi-language processor."""
        self.logger = logger
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect the language of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        text_lower = text.lower()
        language_scores = {}
        
        for lang_code, patterns in self.LANGUAGE_PATTERNS.items():
            score = 0
            total_keywords = 0
            
            for category, keywords in patterns.items():
                for keyword in keywords:
                    total_keywords += 1
                    if keyword in text_lower:
                        score += 1
            
            if total_keywords > 0:
                language_scores[lang_code] = score / total_keywords
        
        if not language_scores:
            return 'en', 0.1  # Default to English with low confidence
        
        best_lang = max(language_scores.items(), key=lambda x: x[1])
        return best_lang[0], best_lang[1]
    
    def extract_with_language_context(self, text: str, language: str) -> Dict[str, List[str]]:
        """Extract information using language-specific patterns.
        
        Args:
            text: Text to process
            language: Detected language code
            
        Returns:
            Dictionary with extracted information by category
        """
        if language not in self.LANGUAGE_PATTERNS:
            language = 'en'  # Fallback to English
        
        patterns = self.LANGUAGE_PATTERNS[language]
        results = {}
        
        text_lower = text.lower()
        lines = text.split('\n')
        
        for category, keywords in patterns.items():
            results[category] = []
            
            for line in lines:
                line_lower = line.lower().strip()
                if not line_lower:
                    continue
                
                for keyword in keywords:
                    if keyword in line_lower:
                        # Extract the relevant part of the line
                        results[category].append(line.strip())
                        break
        
        return results