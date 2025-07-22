# Receipt Processing Application

A comprehensive receipt processing application built with Streamlit that extracts structured data from various file formats using OCR technology and provides analytical insights into spending patterns.

## 🚀 Features

### 📄 Multi-Format Support
- **PDF Processing**: Direct text extraction from machine-readable PDFs using PyMuPDF
- **Image Processing**: OCR capabilities for PNG, JPG, JPEG, BMP, and TIFF formats
- **Intelligent Routing**: Automatic format detection and optimal processing method selection
- **Hybrid Approach**: Fallback to OCR for scanned PDFs when direct text extraction fails

### 🔍 Smart Data Extraction
- **Automated Parsing**: Extract vendor names, transaction dates, amounts, and categories
- **Multiple Date Formats**: Support for various date formats (MM/DD/YYYY, DD/MM/YYYY, Month DD YYYY, etc.)
- **Currency Detection**: Automatic detection of USD, EUR, GBP, and JPY currencies
- **Amount Recognition**: Robust parsing of monetary values with various currency symbols
- **Vendor Identification**: Intelligent vendor name extraction from receipt headers

### 🗄️ Database Management
- **SQLite Storage**: ACID-compliant local database with proper indexing
- **CRUD Operations**: Complete Create, Read, Update, Delete functionality
- **Search & Filter**: Advanced search capabilities with multiple filter criteria
- **Data Validation**: Comprehensive validation using Pydantic models
- **Backup Support**: Easy database backup and restore capabilities

### 📊 Analytics & Insights
- **Interactive Dashboard**: Real-time analytics with interactive visualizations
- **Spending Trends**: Monthly and daily spending pattern analysis
- **Vendor Analysis**: Top vendors and spending distribution insights
- **Category Breakdown**: Expense categorization with automatic classification
- **Statistical Reports**: Comprehensive statistics including averages, medians, and trends
- **Anomaly Detection**: Identification of unusual spending patterns

### 🛠️ Data Management
- **Manual Correction**: Inline editing interface for receipt data
- **Bulk Operations**: Update or delete multiple receipts simultaneously
- **Export Functionality**: CSV and JSON export with custom formatting
- **Data Import**: Support for importing existing receipt data
- **Quality Control**: Confidence scoring and manual review for low-confidence extractions

## 🏗️ Architecture

### Project Structure
