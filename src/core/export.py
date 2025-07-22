"""
Export functionality for receipt data.
Provides CSV, JSON export capabilities and data formatting utilities.
"""

import json
import csv
import io
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Any, Optional, Union
import pandas as pd

from .models import Receipt

logger = logging.getLogger(__name__)

class DataExporter:
    """Handles data export functionality for receipt data."""
    
    def __init__(self):
        """Initialize the data exporter."""
        self.logger = logger
    
    def export_to_csv(self, receipts: List[Receipt], include_metadata: bool = True) -> str:
        """Export receipts to CSV format.
        
        Args:
            receipts: List of receipts to export
            include_metadata: Whether to include processing metadata
            
        Returns:
            CSV content as string
        """
        try:
            if not receipts:
                return ""
            
            # Prepare data for CSV
            csv_data = []
            
            for receipt in receipts:
                row = {
                    "ID": receipt.id,
                    "Vendor": receipt.vendor,
                    "Date": receipt.transaction_date.isoformat(),
                    "Amount": float(receipt.amount),
                    "Currency": receipt.currency,
                    "Category": receipt.category or "Other",
                    "Description": receipt.description or "",
                    "Source_File": receipt.source_file
                }
                
                if include_metadata:
                    row.update({
                        "Processing_Confidence": receipt.processing_confidence,
                        "Created_At": receipt.created_at.isoformat() if receipt.created_at else "",
                        "Updated_At": receipt.updated_at.isoformat() if receipt.updated_at else ""
                    })
                
                csv_data.append(row)
            
            # Create CSV string
            output = io.StringIO()
            if csv_data:
                fieldnames = csv_data[0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            csv_content = output.getvalue()
            output.close()
            
            self.logger.info(f"Exported {len(receipts)} receipts to CSV")
            return csv_content
            
        except Exception as e:
            self.logger.error(f"CSV export failed: {str(e)}")
            raise
    
    def export_to_json(self, receipts: List[Receipt], include_metadata: bool = True, 
                      format_type: str = "detailed") -> str:
        """Export receipts to JSON format.
        
        Args:
            receipts: List of receipts to export
            include_metadata: Whether to include processing metadata
            format_type: 'detailed', 'summary', or 'analytics'
            
        Returns:
            JSON content as string
        """
        try:
            if format_type == "summary":
                return self._export_summary_json(receipts)
            elif format_type == "analytics":
                return self._export_analytics_json(receipts)
            else:
                return self._export_detailed_json(receipts, include_metadata)
            
        except Exception as e:
            self.logger.error(f"JSON export failed: {str(e)}")
            raise
    
    def _export_detailed_json(self, receipts: List[Receipt], include_metadata: bool) -> str:
        """Export detailed JSON format."""
        json_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_receipts": len(receipts),
                "format": "detailed",
                "includes_metadata": include_metadata
            },
            "receipts": []
        }
        
        for receipt in receipts:
            receipt_data = {
                "id": receipt.id,
                "vendor": receipt.vendor,
                "transaction_date": receipt.transaction_date.isoformat(),
                "amount": float(receipt.amount),
                "currency": receipt.currency,
                "category": receipt.category or "Other",
                "description": receipt.description,
                "source_file": receipt.source_file
            }
            
            if include_metadata:
                receipt_data.update({
                    "processing_confidence": receipt.processing_confidence,
                    "created_at": receipt.created_at.isoformat() if receipt.created_at else None,
                    "updated_at": receipt.updated_at.isoformat() if receipt.updated_at else None
                })
            
            json_data["receipts"].append(receipt_data)
        
        return json.dumps(json_data, indent=2, ensure_ascii=False)
    
    def _export_summary_json(self, receipts: List[Receipt]) -> str:
        """Export summary JSON format with aggregated data."""
        from .algorithms import AnalyticsEngine
        from .database import DatabaseManager
        
        # Create temporary analytics engine for calculations
        analytics = AnalyticsEngine(DatabaseManager())
        stats = analytics.calculate_statistics(receipts)
        patterns = analytics.analyze_spending_patterns(receipts)
        
        json_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "format": "summary",
                "total_receipts": len(receipts)
            },
            "statistics": {
                "count": stats["count"],
                "total_amount": float(stats["total_amount"]),
                "average_amount": float(stats["average_amount"]),
                "median_amount": float(stats["median_amount"]),
                "mode_amount": float(stats.get("mode_amount", 0)),
                "min_amount": float(stats["min_amount"]),
                "max_amount": float(stats["max_amount"]),
                "standard_deviation": stats["std_deviation"]
            },
            "spending_patterns": {
                "top_vendors": [(vendor, float(amount)) for vendor, amount in patterns.get("top_vendors", [])],
                "category_breakdown": {k: float(v) for k, v in patterns.get("category_breakdown", {}).items()},
                "monthly_trends": {
                    month: {
                        "total": float(data["total"]),
                        "count": data["count"],
                        "average": float(data["average"])
                    }
                    for month, data in patterns.get("monthly_trends", {}).items()
                }
            }
        }
        
        return json.dumps(json_data, indent=2, ensure_ascii=False)
    
    def _export_analytics_json(self, receipts: List[Receipt]) -> str:
        """Export analytics-focused JSON format."""
        # Group data by different dimensions for analysis
        by_category = {}
        by_vendor = {}
        by_month = {}
        
        for receipt in receipts:
            category = receipt.category or "Other"
            month = receipt.transaction_date.strftime("%Y-%m")
            vendor = receipt.vendor
            
            # Category aggregation
            if category not in by_category:
                by_category[category] = {"count": 0, "total": 0, "amounts": []}
            by_category[category]["count"] += 1
            by_category[category]["total"] += float(receipt.amount)
            by_category[category]["amounts"].append(float(receipt.amount))
            
            # Vendor aggregation
            if vendor not in by_vendor:
                by_vendor[vendor] = {"count": 0, "total": 0, "amounts": []}
            by_vendor[vendor]["count"] += 1
            by_vendor[vendor]["total"] += float(receipt.amount)
            by_vendor[vendor]["amounts"].append(float(receipt.amount))
            
            # Monthly aggregation
            if month not in by_month:
                by_month[month] = {"count": 0, "total": 0, "amounts": []}
            by_month[month]["count"] += 1
            by_month[month]["total"] += float(receipt.amount)
            by_month[month]["amounts"].append(float(receipt.amount))
        
        json_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "format": "analytics",
                "total_receipts": len(receipts)
            },
            "analytics": {
                "by_category": by_category,
                "by_vendor": by_vendor,
                "by_month": by_month
            }
        }
        
        return json.dumps(json_data, indent=2, ensure_ascii=False)
    
    def export_filtered_data(self, receipts: List[Receipt], filters: Dict[str, Any],
                           format_type: str = "csv", include_metadata: bool = True) -> str:
        """Export filtered receipt data with applied filters info.
        
        Args:
            receipts: List of filtered receipts
            filters: Applied filters information
            format_type: 'csv' or 'json'
            include_metadata: Whether to include metadata
            
        Returns:
            Exported data as string
        """
        try:
            if format_type.lower() == "json":
                # Add filter information to JSON export
                json_content = self.export_to_json(receipts, include_metadata, "detailed")
                data = json.loads(json_content)
                data["export_info"]["applied_filters"] = filters
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                # CSV export remains the same
                return self.export_to_csv(receipts, include_metadata)
                
        except Exception as e:
            self.logger.error(f"Filtered export failed: {str(e)}")
            raise
    
    def get_export_filename(self, format_type: str, export_scope: str = "all") -> str:
        """Generate appropriate filename for export.
        
        Args:
            format_type: 'csv' or 'json'
            export_scope: 'all', 'filtered', 'summary', or 'analytics'
            
        Returns:
            Generated filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"receipts_{export_scope}_{timestamp}.{format_type.lower()}"