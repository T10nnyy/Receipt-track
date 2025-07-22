"""
Database operations for receipt processing application.
Handles SQLite database initialization, CRUD operations, and queries.
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, date
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from .models import Receipt, ReceiptCreate, ReceiptUpdate, SearchFilters, AnalyticsData

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database operations for receipt data."""
    
    def __init__(self, db_path: str = "receipts.db"):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logger
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def initialize_database(self) -> None:
        """Initialize database with proper schema and indexes."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create receipts table with comprehensive schema
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS receipts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        vendor TEXT NOT NULL,
                        transaction_date DATE NOT NULL,
                        amount DECIMAL(10,2) NOT NULL CHECK (amount > 0),
                        category TEXT DEFAULT 'Other',
                        source_file TEXT NOT NULL,
                        currency TEXT DEFAULT 'USD' CHECK (length(currency) = 3),
                        description TEXT,
                        processing_confidence REAL CHECK (processing_confidence >= 0 AND processing_confidence <= 1),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better query performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_vendor ON receipts(vendor)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_transaction_date ON receipts(transaction_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_amount ON receipts(amount)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON receipts(category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_file ON receipts(source_file)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON receipts(created_at)")
                
                # Create compound indexes for common queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_vendor_date ON receipts(vendor, transaction_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_category_amount ON receipts(category, amount)")
                
                # Create trigger to update updated_at timestamp
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_receipts_updated_at
                    AFTER UPDATE ON receipts
                    BEGIN
                        UPDATE receipts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                    END
                """)
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def add_receipt(self, receipt: ReceiptCreate) -> int:
        """Add a new receipt to the database.
        
        Args:
            receipt: Receipt data to add
            
        Returns:
            ID of the newly created receipt
            
        Raises:
            Exception: If database operation fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO receipts (
                        vendor, transaction_date, amount, category, source_file,
                        currency, description, processing_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    receipt.vendor,
                    receipt.transaction_date.isoformat(),
                    float(receipt.amount),
                    receipt.category,
                    receipt.source_file,
                    receipt.currency,
                    receipt.description,
                    receipt.processing_confidence
                ))
                
                receipt_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Added receipt with ID: {receipt_id}")
                return receipt_id
                
        except Exception as e:
            self.logger.error(f"Failed to add receipt: {str(e)}")
            raise
    
    def get_receipt(self, receipt_id: int) -> Optional[Receipt]:
        """Get a receipt by ID.
        
        Args:
            receipt_id: ID of the receipt to retrieve
            
        Returns:
            Receipt object if found, None otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM receipts WHERE id = ?", (receipt_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_receipt(row)
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get receipt {receipt_id}: {str(e)}")
            raise
    
    def get_all_receipts(self, limit: Optional[int] = None, offset: int = 0) -> List[Receipt]:
        """Get all receipts with optional pagination.
        
        Args:
            limit: Maximum number of receipts to return
            offset: Number of receipts to skip
            
        Returns:
            List of Receipt objects
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM receipts ORDER BY transaction_date DESC, created_at DESC"
                params = []
                
                if limit is not None:
                    query += " LIMIT ? OFFSET ?"
                    params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_receipt(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get all receipts: {str(e)}")
            raise
    
    def update_receipt(self, receipt_id: int, updates: ReceiptUpdate) -> bool:
        """Update an existing receipt.
        
        Args:
            receipt_id: ID of the receipt to update
            updates: Fields to update
            
        Returns:
            True if update was successful, False if receipt not found
        """
        try:
            # Build dynamic update query based on provided fields
            update_fields = []
            update_values = []
            
            update_dict = updates.dict(exclude_unset=True)
            if not update_dict:
                return True  # No updates to make
            
            for field, value in update_dict.items():
                if field == 'transaction_date' and isinstance(value, date):
                    update_fields.append(f"{field} = ?")
                    update_values.append(value.isoformat())
                elif field == 'amount' and isinstance(value, Decimal):
                    update_fields.append(f"{field} = ?")
                    update_values.append(float(value))
                else:
                    update_fields.append(f"{field} = ?")
                    update_values.append(value)
            
            update_values.append(receipt_id)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = f"""
                    UPDATE receipts 
                    SET {', '.join(update_fields)}
                    WHERE id = ?
                """
                
                cursor.execute(query, update_values)
                rows_affected = cursor.rowcount
                conn.commit()
                
                if rows_affected > 0:
                    self.logger.info(f"Updated receipt {receipt_id}")
                    return True
                else:
                    self.logger.warning(f"Receipt {receipt_id} not found for update")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to update receipt {receipt_id}: {str(e)}")
            raise
    
    def delete_receipt(self, receipt_id: int) -> bool:
        """Delete a receipt by ID.
        
        Args:
            receipt_id: ID of the receipt to delete
            
        Returns:
            True if deletion was successful, False if receipt not found
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))
                rows_affected = cursor.rowcount
                conn.commit()
                
                if rows_affected > 0:
                    self.logger.info(f"Deleted receipt {receipt_id}")
                    return True
                else:
                    self.logger.warning(f"Receipt {receipt_id} not found for deletion")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to delete receipt {receipt_id}: {str(e)}")
            raise
    
    def search_receipts(self, filters: SearchFilters, limit: Optional[int] = None, offset: int = 0) -> List[Receipt]:
        """Search receipts based on filters.
        
        Args:
            filters: Search criteria
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching Receipt objects
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build dynamic query
                conditions = []
                params = []
                
                if filters.vendor_search:
                    conditions.append("vendor LIKE ?")
                    params.append(f"%{filters.vendor_search}%")
                
                if filters.category_filter:
                    conditions.append("category = ?")
                    params.append(filters.category_filter)
                
                if filters.min_amount is not None:
                    conditions.append("amount >= ?")
                    params.append(float(filters.min_amount))
                
                if filters.max_amount is not None:
                    conditions.append("amount <= ?")
                    params.append(float(filters.max_amount))
                
                if filters.start_date:
                    conditions.append("transaction_date >= ?")
                    params.append(filters.start_date.isoformat())
                
                if filters.end_date:
                    conditions.append("transaction_date <= ?")
                    params.append(filters.end_date.isoformat())
                
                if filters.currency_filter:
                    conditions.append("currency = ?")
                    params.append(filters.currency_filter)
                
                # Build final query
                query = "SELECT * FROM receipts"
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                query += " ORDER BY transaction_date DESC, created_at DESC"
                
                if limit is not None:
                    query += " LIMIT ? OFFSET ?"
                    params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_receipt(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to search receipts: {str(e)}")
            raise
    
    def get_analytics_data(self) -> AnalyticsData:
        """Get comprehensive analytics data.
        
        Returns:
            AnalyticsData object with computed statistics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get basic statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_receipts,
                        SUM(amount) as total_amount,
                        AVG(amount) as average_amount,
                        MIN(transaction_date) as earliest_date,
                        MAX(transaction_date) as latest_date
                    FROM receipts
                """)
                stats = cursor.fetchone()
                
                # Get top vendors
                cursor.execute("""
                    SELECT vendor, SUM(amount) as total_spent, COUNT(*) as receipt_count
                    FROM receipts
                    GROUP BY vendor
                    ORDER BY total_spent DESC
                    LIMIT 10
                """)
                top_vendors = [
                    {
                        "vendor": row["vendor"],
                        "total_spent": Decimal(str(row["total_spent"])),
                        "receipt_count": row["receipt_count"]
                    }
                    for row in cursor.fetchall()
                ]
                
                # Get category breakdown
                cursor.execute("""
                    SELECT category, SUM(amount) as total_amount
                    FROM receipts
                    GROUP BY category
                    ORDER BY total_amount DESC
                """)
                category_breakdown = {
                    row["category"]: Decimal(str(row["total_amount"]))
                    for row in cursor.fetchall()
                }
                
                # Get monthly trends
                cursor.execute("""
                    SELECT 
                        strftime('%Y-%m', transaction_date) as month,
                        SUM(amount) as total_amount,
                        COUNT(*) as receipt_count
                    FROM receipts
                    GROUP BY strftime('%Y-%m', transaction_date)
                    ORDER BY month
                """)
                monthly_trends = [
                    {
                        "month": row["month"],
                        "total_amount": Decimal(str(row["total_amount"])),
                        "receipt_count": row["receipt_count"]
                    }
                    for row in cursor.fetchall()
                ]
                
                return AnalyticsData(
                    total_receipts=stats["total_receipts"] or 0,
                    total_amount=Decimal(str(stats["total_amount"] or 0)),
                    average_amount=Decimal(str(stats["average_amount"] or 0)),
                    date_range={
                        "earliest": datetime.fromisoformat(stats["earliest_date"]).date() if stats["earliest_date"] else None,
                        "latest": datetime.fromisoformat(stats["latest_date"]).date() if stats["latest_date"] else None
                    },
                    top_vendors=top_vendors,
                    category_breakdown=category_breakdown,
                    monthly_trends=monthly_trends
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get analytics data: {str(e)}")
            raise
    
    def get_receipt_count(self, filters: Optional[SearchFilters] = None) -> int:
        """Get total count of receipts matching filters.
        
        Args:
            filters: Optional search criteria
            
        Returns:
            Number of matching receipts
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if filters is None:
                    cursor.execute("SELECT COUNT(*) FROM receipts")
                    return cursor.fetchone()[0]
                
                # Build filtered count query (similar to search_receipts)
                conditions = []
                params = []
                
                if filters.vendor_search:
                    conditions.append("vendor LIKE ?")
                    params.append(f"%{filters.vendor_search}%")
                
                if filters.category_filter:
                    conditions.append("category = ?")
                    params.append(filters.category_filter)
                
                if filters.min_amount is not None:
                    conditions.append("amount >= ?")
                    params.append(float(filters.min_amount))
                
                if filters.max_amount is not None:
                    conditions.append("amount <= ?")
                    params.append(float(filters.max_amount))
                
                if filters.start_date:
                    conditions.append("transaction_date >= ?")
                    params.append(filters.start_date.isoformat())
                
                if filters.end_date:
                    conditions.append("transaction_date <= ?")
                    params.append(filters.end_date.isoformat())
                
                if filters.currency_filter:
                    conditions.append("currency = ?")
                    params.append(filters.currency_filter)
                
                query = "SELECT COUNT(*) FROM receipts"
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                cursor.execute(query, params)
                return cursor.fetchone()[0]
                
        except Exception as e:
            self.logger.error(f"Failed to get receipt count: {str(e)}")
            raise
    
    def _row_to_receipt(self, row: sqlite3.Row) -> Receipt:
        """Convert database row to Receipt object.
        
        Args:
            row: SQLite row object
            
        Returns:
            Receipt object
        """
        return Receipt(
            id=row["id"],
            vendor=row["vendor"],
            transaction_date=datetime.fromisoformat(row["transaction_date"]).date(),
            amount=Decimal(str(row["amount"])),
            category=row["category"],
            source_file=row["source_file"],
            currency=row["currency"],
            description=row["description"],
            processing_confidence=row["processing_confidence"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
        )
