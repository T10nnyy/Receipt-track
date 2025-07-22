# Receipt Processing Application

## Overview

This is a comprehensive receipt processing application built with Streamlit that extracts structured data from various file formats (PDFs and images) using OCR technology. The application provides analytical insights into spending patterns with a complete CRUD interface for managing receipt data.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Directory Structure
The application follows a professional "src layout" structure:
- `src/` - Main source code directory
- `src/core/` - Backend business logic (models, database, parsing, algorithms)
- `src/pages/` - Streamlit multi-page components
- `src/ui/` - Reusable UI components
- `tests/` - Comprehensive test suite

### Technology Stack
- **Frontend**: Streamlit with multi-page architecture
- **Backend**: Python with modular component design
- **Database**: SQLite with ACID compliance and proper indexing
- **OCR/Text Processing**: Hybrid approach using PyMuPDF for text-based PDFs and Tesseract+OpenCV for images
- **Data Validation**: Pydantic models with strict type checking
- **Analytics**: Plotly for interactive visualizations

## Key Components

### Core Models (`src/core/models.py`)
- **Receipt Model**: Complete data structure with Pydantic validation
- **Validation Rules**: Positive amounts, required fields, proper date formats
- **Supporting Models**: ReceiptCreate, ReceiptUpdate, SearchFilters, ProcessingResult, AnalyticsData

### File Processing Engine (`src/core/parsing.py`)
- **Multi-format Support**: Intelligent routing for PDFs, images (PNG, JPG, JPEG, BMP, TIFF)
- **Image Preprocessing**: OpenCV pipeline with perspective correction, noise removal, binarization
- **Text Extraction**: Direct text parsing for machine-readable PDFs, OCR for images and scanned PDFs
- **Data Extraction**: Regex-based extraction for vendor names, dates, amounts, categories

### Database Layer (`src/core/database.py`)
- **SQLite Implementation**: Context-managed connections with automatic cleanup
- **CRUD Operations**: Complete Create, Read, Update, Delete functionality
- **Advanced Queries**: Search, filtering, pagination, sorting
- **Data Integrity**: Transaction management and error handling

### Analytics Engine (`src/core/algorithms.py`)
- **Search Functionality**: Fuzzy search with similarity scoring
- **Statistical Analysis**: Spending trends, averages, medians
- **Pattern Recognition**: Monthly/daily spending analysis, vendor insights
- **Aggregation Functions**: Category breakdowns, time-based groupings

## Data Flow

### File Processing Workflow
1. **File Upload**: Multi-format file acceptance through Streamlit interface
2. **Format Detection**: Automatic identification of file type
3. **Preprocessing**: Image optimization for OCR (if needed)
4. **Text Extraction**: PyMuPDF for PDFs, Tesseract OCR for images
5. **Data Parsing**: Regex-based extraction of structured data
6. **Validation**: Pydantic model validation and confidence scoring
7. **Database Storage**: SQLite persistence with proper indexing

### User Interface Flow
1. **Main Page**: File upload and processing interface
2. **Data Explorer**: Interactive table with search, filter, and edit capabilities
3. **Analytics Dashboard**: Comprehensive visualizations and insights
4. **Sidebar**: Global controls and quick statistics

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **PyMuPDF (fitz)**: PDF text extraction
- **Pytesseract**: OCR engine integration
- **OpenCV**: Image preprocessing
- **Plotly**: Interactive visualizations
- **Pydantic**: Data validation and serialization
- **Pandas**: Data manipulation and analysis

### System Requirements
- **Tesseract OCR**: Must be installed system-wide for OCR functionality
- **SQLite**: Built into Python standard library
- **PIL/Pillow**: Image processing support

## Deployment Strategy

### Local Development
- **Database**: SQLite file-based storage (`receipts.db`)
- **Logging**: File-based logging with console output
- **Configuration**: Environment-based settings

### Production Considerations
- **Database Migration**: Current SQLite setup can be extended to PostgreSQL
- **File Storage**: Local file processing with potential cloud storage integration
- **Scalability**: Modular architecture supports horizontal scaling
- **Monitoring**: Comprehensive logging infrastructure in place

### Error Handling
- **Database Errors**: Transaction rollback and connection cleanup
- **File Processing Errors**: Graceful fallback and user feedback
- **Validation Errors**: Detailed error messages with correction suggestions
- **OCR Confidence**: Quality scoring and manual review workflows

The application is designed with production readiness in mind, featuring proper separation of concerns, comprehensive error handling, and extensible architecture for future enhancements.