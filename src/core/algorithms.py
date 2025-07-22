"""
Search and analytics algorithms for receipt processing application.
Implements search mechanisms, sorting algorithms, and aggregation functions.
"""

import re
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
import statistics

from .models import Receipt, SearchFilters, AnalyticsData
from .database import DatabaseManager

logger = logging.getLogger(__name__)

class SearchEngine:
    """Advanced search functionality for receipts."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize search engine.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.logger = logger
    
    def fuzzy_search(self, query: str, receipts: List[Receipt], threshold: float = 0.6) -> List[Receipt]:
        """Perform fuzzy search on receipts.
        
        Args:
            query: Search query string
            receipts: List of receipts to search
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of matching receipts with similarity scores
        """
        if not query.strip():
            return receipts
        
        query_lower = query.lower()
        results = []
        
        for receipt in receipts:
            # Calculate similarity for different fields
            vendor_similarity = self._calculate_similarity(query_lower, receipt.vendor.lower())
            description_similarity = 0.0
            
            if receipt.description:
                description_similarity = self._calculate_similarity(query_lower, receipt.description.lower())
            
            # Use highest similarity
            max_similarity = max(vendor_similarity, description_similarity)
            
            if max_similarity >= threshold:
                results.append(receipt)
        
        # Sort by relevance (similarity)
        return self._sort_by_relevance(results, query_lower)
    
    def advanced_search(self, filters: SearchFilters, sort_by: str = "date", sort_desc: bool = True) -> List[Receipt]:
        """Perform advanced search with multiple filters and sorting.
        
        Args:
            filters: Search filters object
            sort_by: Field to sort by ("date", "amount", "vendor", "category")
            sort_desc: Sort in descending order
            
        Returns:
            List of filtered and sorted receipts
        """
        try:
            # Get filtered receipts from database
            receipts = self.db_manager.search_receipts(filters)
            
            # Apply in-memory sorting if needed
            return self._sort_receipts(receipts, sort_by, sort_desc)
            
        except Exception as e:
            self.logger.error(f"Advanced search failed: {str(e)}")
            raise
    
    def search_by_pattern(self, pattern: str, receipts: List[Receipt], field: str = "vendor") -> List[Receipt]:
        """Search receipts using regex patterns.
        
        Args:
            pattern: Regex pattern to match
            receipts: List of receipts to search
            field: Field to search in ("vendor", "description", "category")
            
        Returns:
            List of matching receipts
        """
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            results = []
            
            for receipt in receipts:
                search_text = ""
                
                if field == "vendor":
                    search_text = receipt.vendor
                elif field == "description" and receipt.description:
                    search_text = receipt.description
                elif field == "category":
                    search_text = receipt.category or ""
                
                if compiled_pattern.search(search_text):
                    results.append(receipt)
            
            return results
            
        except re.error as e:
            self.logger.error(f"Invalid regex pattern '{pattern}': {str(e)}")
            return []
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Jaccard similarity.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to sets of words
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _sort_by_relevance(self, receipts: List[Receipt], query: str) -> List[Receipt]:
        """Sort receipts by relevance to search query.
        
        Args:
            receipts: List of receipts
            query: Search query
            
        Returns:
            Receipts sorted by relevance
        """
        def relevance_score(receipt: Receipt) -> float:
            vendor_sim = self._calculate_similarity(query, receipt.vendor.lower())
            desc_sim = 0.0
            
            if receipt.description:
                desc_sim = self._calculate_similarity(query, receipt.description.lower())
            
            return max(vendor_sim, desc_sim)
        
        return sorted(receipts, key=relevance_score, reverse=True)
    
    def _sort_receipts(self, receipts: List[Receipt], sort_by: str, sort_desc: bool) -> List[Receipt]:
        """Sort receipts by specified field.
        
        Args:
            receipts: List of receipts to sort
            sort_by: Field to sort by
            sort_desc: Sort in descending order
            
        Returns:
            Sorted list of receipts
        """
        sort_key_map = {
            "date": lambda r: r.transaction_date,
            "amount": lambda r: r.amount,
            "vendor": lambda r: r.vendor.lower(),
            "category": lambda r: r.category.lower() if r.category else "",
            "created_at": lambda r: r.created_at or datetime.min,
        }
        
        if sort_by not in sort_key_map:
            sort_by = "date"
        
        try:
            return sorted(receipts, key=sort_key_map[sort_by], reverse=sort_desc)
        except Exception as e:
            self.logger.error(f"Sorting failed: {str(e)}")
            return receipts

class AnalyticsEngine:
    """Analytics and aggregation functions for receipt data."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize analytics engine.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.logger = logger
    
    def calculate_statistics(self, receipts: List[Receipt]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for receipts.
        
        Args:
            receipts: List of receipts
            
        Returns:
            Dictionary with statistical data
        """
        if not receipts:
            return {
                "count": 0,
                "total_amount": Decimal("0"),
                "average_amount": Decimal("0"),
                "median_amount": Decimal("0"),
                "min_amount": Decimal("0"),
                "max_amount": Decimal("0"),
                "std_deviation": 0.0
            }
        
        amounts = [float(receipt.amount) for receipt in receipts]
        
        return {
            "count": len(receipts),
            "total_amount": sum(receipt.amount for receipt in receipts),
            "average_amount": Decimal(str(statistics.mean(amounts))),
            "median_amount": Decimal(str(statistics.median(amounts))),
            "min_amount": min(receipt.amount for receipt in receipts),
            "max_amount": max(receipt.amount for receipt in receipts),
            "std_deviation": statistics.stdev(amounts) if len(amounts) > 1 else 0.0
        }
    
    def analyze_spending_patterns(self, receipts: List[Receipt]) -> Dict[str, Any]:
        """Analyze spending patterns and trends.
        
        Args:
            receipts: List of receipts
            
        Returns:
            Dictionary with spending pattern analysis
        """
        if not receipts:
            return {}
        
        # Group by various dimensions
        by_vendor = defaultdict(list)
        by_category = defaultdict(list)
        by_month = defaultdict(list)
        by_day_of_week = defaultdict(list)
        
        for receipt in receipts:
            by_vendor[receipt.vendor].append(receipt)
            by_category[receipt.category or "Other"].append(receipt)
            
            month_key = receipt.transaction_date.strftime("%Y-%m")
            by_month[month_key].append(receipt)
            
            day_name = receipt.transaction_date.strftime("%A")
            by_day_of_week[day_name].append(receipt)
        
        # Calculate top spending categories
        top_vendors = sorted(
            [(vendor, sum(r.amount for r in receipts_list)) 
             for vendor, receipts_list in by_vendor.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        category_totals = {
            category: sum(r.amount for r in receipts_list)
            for category, receipts_list in by_category.items()
        }
        
        # Monthly trends
        monthly_trends = {}
        for month, month_receipts in by_month.items():
            monthly_trends[month] = {
                "total": sum(r.amount for r in month_receipts),
                "count": len(month_receipts),
                "average": sum(r.amount for r in month_receipts) / len(month_receipts)
            }
        
        # Day of week analysis
        day_patterns = {}
        for day, day_receipts in by_day_of_week.items():
            day_patterns[day] = {
                "total": sum(r.amount for r in day_receipts),
                "count": len(day_receipts),
                "average": sum(r.amount for r in day_receipts) / len(day_receipts)
            }
        
        return {
            "top_vendors": top_vendors,
            "category_breakdown": category_totals,
            "monthly_trends": monthly_trends,
            "day_of_week_patterns": day_patterns,
            "spending_frequency": self._calculate_spending_frequency(receipts)
        }
    
    def detect_anomalies(self, receipts: List[Receipt], threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalous transactions based on statistical analysis.
        
        Args:
            receipts: List of receipts
            threshold: Standard deviation threshold for anomaly detection
            
        Returns:
            List of anomalous transactions with reasons
        """
        if len(receipts) < 10:  # Need sufficient data
            return []
        
        amounts = [float(receipt.amount) for receipt in receipts]
        mean_amount = statistics.mean(amounts)
        std_amount = statistics.stdev(amounts) if len(amounts) > 1 else 0
        
        anomalies = []
        
        for receipt in receipts:
            reasons = []
            
            # Amount-based anomalies
            if std_amount > 0:
                z_score = abs((float(receipt.amount) - mean_amount) / std_amount)
                if z_score > threshold:
                    reasons.append(f"Unusual amount (z-score: {z_score:.2f})")
            
            # Frequency-based anomalies (vendor appears very rarely)
            vendor_count = sum(1 for r in receipts if r.vendor == receipt.vendor)
            if vendor_count == 1 and len(receipts) > 20:
                reasons.append("One-time vendor")
            
            # Date-based anomalies (weekend vs weekday spending patterns)
            is_weekend = receipt.transaction_date.weekday() >= 5
            weekend_receipts = [r for r in receipts if r.transaction_date.weekday() >= 5]
            weekday_receipts = [r for r in receipts if r.transaction_date.weekday() < 5]
            
            if weekend_receipts and weekday_receipts:
                weekend_avg = sum(r.amount for r in weekend_receipts) / len(weekend_receipts)
                weekday_avg = sum(r.amount for r in weekday_receipts) / len(weekday_receipts)
                
                if is_weekend and float(receipt.amount) > float(weekend_avg) * 2:
                    reasons.append("High weekend spending")
                elif not is_weekend and float(receipt.amount) > float(weekday_avg) * 2:
                    reasons.append("High weekday spending")
            
            if reasons:
                anomalies.append({
                    "receipt_id": receipt.id,
                    "vendor": receipt.vendor,
                    "amount": receipt.amount,
                    "date": receipt.transaction_date,
                    "reasons": reasons,
                    "confidence": min(len(reasons) * 0.3, 1.0)
                })
        
        # Sort by confidence (most anomalous first)
        return sorted(anomalies, key=lambda x: x["confidence"], reverse=True)
    
    def generate_insights(self, receipts: List[Receipt]) -> List[Dict[str, Any]]:
        """Generate actionable insights from receipt data.
        
        Args:
            receipts: List of receipts
            
        Returns:
            List of insight objects
        """
        insights = []
        
        if not receipts:
            return insights
        
        # Spending trend insight
        if len(receipts) >= 6:  # Need at least 6 receipts
            monthly_totals = defaultdict(Decimal)
            for receipt in receipts:
                month_key = receipt.transaction_date.strftime("%Y-%m")
                monthly_totals[month_key] += receipt.amount
            
            if len(monthly_totals) >= 2:
                sorted_months = sorted(monthly_totals.items())
                recent_months = sorted_months[-2:]
                
                if len(recent_months) == 2:
                    prev_total = recent_months[0][1]
                    curr_total = recent_months[1][1]
                    
                    if curr_total > prev_total * Decimal("1.2"):
                        insights.append({
                            "type": "spending_increase",
                            "title": "Spending Increased",
                            "description": f"Your spending increased by {((curr_total/prev_total - 1) * 100):.1f}% this month",
                            "severity": "warning",
                            "data": {"previous": prev_total, "current": curr_total}
                        })
                    elif curr_total < prev_total * Decimal("0.8"):
                        insights.append({
                            "type": "spending_decrease",
                            "title": "Spending Decreased", 
                            "description": f"Your spending decreased by {((1 - curr_total/prev_total) * 100):.1f}% this month",
                            "severity": "info",
                            "data": {"previous": prev_total, "current": curr_total}
                        })
        
        # Top vendor insight
        vendor_totals = defaultdict(Decimal)
        for receipt in receipts:
            vendor_totals[receipt.vendor] += receipt.amount
        
        if vendor_totals:
            top_vendor, top_amount = max(vendor_totals.items(), key=lambda x: x[1])
            total_spending = sum(vendor_totals.values())
            percentage = (top_amount / total_spending * 100) if total_spending > 0 else 0
            
            if percentage > 30:
                insights.append({
                    "type": "top_vendor",
                    "title": "High Vendor Concentration",
                    "description": f"{top_vendor} accounts for {percentage:.1f}% of your spending",
                    "severity": "info",
                    "data": {"vendor": top_vendor, "amount": top_amount, "percentage": percentage}
                })
        
        # Category distribution insight
        category_totals = defaultdict(Decimal)
        for receipt in receipts:
            category_totals[receipt.category or "Other"] += receipt.amount
        
        if len(category_totals) > 1:
            sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
            top_category, top_amount = sorted_categories[0]
            total_spending = sum(category_totals.values())
            percentage = (top_amount / total_spending * 100) if total_spending > 0 else 0
            
            if percentage > 50:
                insights.append({
                    "type": "category_concentration",
                    "title": "Category Concentration",
                    "description": f"{top_category} represents {percentage:.1f}% of your spending",
                    "severity": "info",
                    "data": {"category": top_category, "amount": top_amount, "percentage": percentage}
                })
        
        return insights
    
    def _calculate_spending_frequency(self, receipts: List[Receipt]) -> Dict[str, Any]:
        """Calculate spending frequency patterns.
        
        Args:
            receipts: List of receipts
            
        Returns:
            Dictionary with frequency analysis
        """
        if not receipts:
            return {}
        
        # Sort receipts by date
        sorted_receipts = sorted(receipts, key=lambda r: r.transaction_date)
        
        # Calculate intervals between transactions
        intervals = []
        for i in range(1, len(sorted_receipts)):
            delta = (sorted_receipts[i].transaction_date - sorted_receipts[i-1].transaction_date).days
            if delta > 0:
                intervals.append(delta)
        
        if not intervals:
            return {"average_days_between": 0, "frequency_pattern": "insufficient_data"}
        
        avg_interval = statistics.mean(intervals)
        
        # Determine frequency pattern
        if avg_interval <= 1:
            pattern = "daily"
        elif avg_interval <= 3:
            pattern = "frequent"
        elif avg_interval <= 7:
            pattern = "weekly"
        elif avg_interval <= 30:
            pattern = "monthly"
        else:
            pattern = "infrequent"
        
        return {
            "average_days_between": avg_interval,
            "frequency_pattern": pattern,
            "total_intervals": len(intervals),
            "min_interval": min(intervals),
            "max_interval": max(intervals)
        }
