"""
Analytics Dashboard page for receipt processing application.
Provides comprehensive analytics, visualizations, and insights.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from collections import defaultdict
import calendar
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from core.algorithms import AnalyticsEngine
from core.models import SearchFilters

logger = logging.getLogger(__name__)

def main():
    """Main function for Analytics Dashboard page."""
    st.set_page_config(
        page_title="Analytics Dashboard - Receipt Processing",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Analytics Dashboard")
    st.markdown("Comprehensive insights into your spending patterns")
    
    # Check if database manager is initialized
    if 'db_manager' not in st.session_state:
        st.error("Database not initialized. Please return to the main page.")
        return
    
    db_manager = st.session_state.db_manager
    
    # Initialize analytics engine
    if 'analytics_engine' not in st.session_state:
        st.session_state.analytics_engine = AnalyticsEngine(db_manager)
    
    analytics_engine = st.session_state.analytics_engine
    
    try:
        # Load data
        all_receipts = db_manager.get_all_receipts()
        
        if not all_receipts:
            st.info("📝 No receipt data available yet. Upload some receipts to see analytics!")
            st.markdown("""
            ### Get Started:
            1. Go back to the main page
            2. Upload receipt files (PDFs or images)
            3. Return here to view your analytics
            """)
            return
        
        # Sidebar filters for analytics
        with st.sidebar:
            st.header("📅 Analytics Filters")
            
            # Date range selector
            min_date = min(r.transaction_date for r in all_receipts)
            max_date = max(r.transaction_date for r in all_receipts)
            
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Select date range for analytics"
            )
            
            # Category filter
            categories = sorted(list(set(r.category or "Other" for r in all_receipts)))
            selected_categories = st.multiselect(
                "Categories",
                options=categories,
                default=categories,
                help="Select categories to include in analysis"
            )
            
            # Vendor filter (top 10)
            vendor_totals = defaultdict(Decimal)
            for receipt in all_receipts:
                vendor_totals[receipt.vendor] += receipt.amount
            
            top_vendors = sorted(vendor_totals.items(), key=lambda x: x[1], reverse=True)[:10]
            vendor_options = [v[0] for v in top_vendors]
            
            selected_vendors = st.multiselect(
                "Top Vendors",
                options=vendor_options,
                default=vendor_options,
                help="Select vendors to include in analysis"
            )
            
            # Currency filter
            currencies = sorted(list(set(r.currency for r in all_receipts)))
            selected_currencies = st.multiselect(
                "Currencies",
                options=currencies,
                default=currencies
            )
        
        # Filter receipts based on sidebar selections
        filtered_receipts = []
        for receipt in all_receipts:
            # Date filter
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                if not (start_date <= receipt.transaction_date <= end_date):
                    continue
            elif isinstance(date_range, list) and len(date_range) == 1:
                if receipt.transaction_date != date_range[0]:
                    continue
            
            # Category filter
            if (receipt.category or "Other") not in selected_categories:
                continue
            
            # Vendor filter (if any selected)
            if selected_vendors and receipt.vendor not in selected_vendors:
                continue
            
            # Currency filter
            if receipt.currency not in selected_currencies:
                continue
            
            filtered_receipts.append(receipt)
        
        if not filtered_receipts:
            st.warning("No data matches the selected filters. Please adjust your filters.")
            return
        
        # Calculate statistics
        stats = analytics_engine.calculate_statistics(filtered_receipts)
        patterns = analytics_engine.analyze_spending_patterns(filtered_receipts)
        insights = analytics_engine.generate_insights(filtered_receipts)
        
        # Key Metrics Row
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Receipts",
                stats['count'],
                help="Number of receipts in selected period"
            )
        
        with col2:
            st.metric(
                "Total Spending",
                f"${stats['total_amount']:,.2f}",
                help="Total amount spent in selected period"
            )
        
        with col3:
            st.metric(
                "Average Amount", 
                f"${stats['average_amount']:.2f}",
                help="Average spending per receipt"
            )
        
        with col4:
            # Calculate spending vs previous period (if applicable)
            if len(filtered_receipts) > 1:
                sorted_receipts = sorted(filtered_receipts, key=lambda r: r.transaction_date)
                mid_point = len(sorted_receipts) // 2
                
                first_half_total = sum(r.amount for r in sorted_receipts[:mid_point])
                second_half_total = sum(r.amount for r in sorted_receipts[mid_point:])
                
                if first_half_total > 0:
                    change = ((second_half_total - first_half_total) / first_half_total) * 100
                    st.metric(
                        "Period Change",
                        f"{change:+.1f}%",
                        help="Change compared to previous period"
                    )
                else:
                    st.metric("Period Change", "N/A")
            else:
                st.metric("Period Change", "N/A")
        
        # Secondary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Median Amount", f"${stats['median_amount']:.2f}")
        with col2:
            st.metric("Highest Receipt", f"${stats['max_amount']:.2f}")
        with col3:
            st.metric("Lowest Receipt", f"${stats['min_amount']:.2f}")
        with col4:
            st.metric("Std Deviation", f"${stats['std_deviation']:.2f}")
        
        st.markdown("---")
        
        # Insights Section
        if insights:
            st.subheader("💡 Key Insights")
            
            for insight in insights[:3]:  # Show top 3 insights
                if insight['severity'] == 'warning':
                    st.warning(f"⚠️ **{insight['title']}**: {insight['description']}")
                elif insight['severity'] == 'info':
                    st.info(f"ℹ️ **{insight['title']}**: {insight['description']}")
                else:
                    st.success(f"✅ **{insight['title']}**: {insight['description']}")
            
            st.markdown("---")
        
        # Visualizations
        st.subheader("📊 Spending Analysis")
        
        # Row 1: Category breakdown and vendor analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Category pie chart
            if patterns.get('category_breakdown'):
                categories_df = pd.DataFrame([
                    {"Category": cat, "Amount": float(amount)} 
                    for cat, amount in patterns['category_breakdown'].items()
                ])
                
                fig_pie = px.pie(
                    categories_df, 
                    values="Amount", 
                    names="Category",
                    title="Spending by Category"
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No category data available")
        
        with col2:
            # Top vendors bar chart
            if patterns.get('top_vendors'):
                vendors_df = pd.DataFrame([
                    {"Vendor": vendor, "Amount": float(amount)} 
                    for vendor, amount in patterns['top_vendors'][:10]
                ])
                
                fig_bar = px.bar(
                    vendors_df,
                    x="Amount",
                    y="Vendor",
                    orientation="h",
                    title="Top 10 Vendors by Spending"
                )
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No vendor data available")
        
        # Row 2: Time-based analysis
        st.subheader("📅 Time-Based Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly spending trend
            if patterns.get('monthly_trends'):
                monthly_df = pd.DataFrame([
                    {
                        "Month": month,
                        "Total": float(data['total']),
                        "Count": data['count'],
                        "Average": float(data['average'])
                    }
                    for month, data in patterns['monthly_trends'].items()
                ])
                monthly_df = monthly_df.sort_values('Month')
                
                fig_line = px.line(
                    monthly_df,
                    x="Month",
                    y="Total",
                    title="Monthly Spending Trend",
                    markers=True
                )
                fig_line.update_layout(xaxis_title="Month", yaxis_title="Total Spending ($)")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Insufficient data for monthly trends")
        
        with col2:
            # Day of week pattern
            if patterns.get('day_of_week_patterns'):
                days_df = pd.DataFrame([
                    {
                        "Day": day,
                        "Total": float(data['total']),
                        "Count": data['count'],
                        "Average": float(data['average'])
                    }
                    for day, data in patterns['day_of_week_patterns'].items()
                ])
                
                # Order days properly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                days_df['Day'] = pd.Categorical(days_df['Day'], categories=day_order, ordered=True)
                days_df = days_df.sort_values('Day')
                
                fig_bar_days = px.bar(
                    days_df,
                    x="Day",
                    y="Total",
                    title="Spending by Day of Week"
                )
                st.plotly_chart(fig_bar_days, use_container_width=True)
            else:
                st.info("Insufficient data for day-of-week analysis")
        
        # Row 3: Advanced analytics
        st.subheader("🔍 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Spending distribution histogram
            amounts = [float(r.amount) for r in filtered_receipts]
            fig_hist = px.histogram(
                x=amounts,
                title="Spending Amount Distribution",
                labels={'x': 'Amount ($)', 'y': 'Frequency'},
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Cumulative spending over time
            receipts_by_date = sorted(filtered_receipts, key=lambda r: r.transaction_date)
            cumulative_data = []
            running_total = Decimal('0')
            
            for receipt in receipts_by_date:
                running_total += receipt.amount
                cumulative_data.append({
                    'Date': receipt.transaction_date,
                    'Cumulative': float(running_total)
                })
            
            if cumulative_data:
                cumulative_df = pd.DataFrame(cumulative_data)
                fig_cumulative = px.line(
                    cumulative_df,
                    x='Date',
                    y='Cumulative',
                    title='Cumulative Spending Over Time'
                )
                st.plotly_chart(fig_cumulative, use_container_width=True)
            else:
                st.info("No data for cumulative analysis")
        
        # Detailed Tables Section
        st.markdown("---")
        st.subheader("📋 Detailed Breakdown")
        
        tab1, tab2, tab3 = st.tabs(["📊 Category Summary", "🏪 Vendor Summary", "📅 Monthly Summary"])
        
        with tab1:
            if patterns.get('category_breakdown'):
                category_summary = []
                for category, total in patterns['category_breakdown'].items():
                    count = sum(1 for r in filtered_receipts if (r.category or "Other") == category)
                    avg = total / count if count > 0 else Decimal('0')
                    category_summary.append({
                        "Category": category,
                        "Total Spent": f"${total:.2f}",
                        "Receipt Count": count,
                        "Average per Receipt": f"${avg:.2f}",
                        "% of Total": f"{(total / stats['total_amount'] * 100):.1f}%"
                    })
                
                category_df = pd.DataFrame(category_summary)
                st.dataframe(category_df, use_container_width=True)
            else:
                st.info("No category data available")
        
        with tab2:
            if patterns.get('top_vendors'):
                vendor_summary = []
                for vendor, total in patterns['top_vendors']:
                    vendor_receipts = [r for r in filtered_receipts if r.vendor == vendor]
                    count = len(vendor_receipts)
                    avg = total / count if count > 0 else Decimal('0')
                    vendor_summary.append({
                        "Vendor": vendor,
                        "Total Spent": f"${total:.2f}",
                        "Visit Count": count,
                        "Average per Visit": f"${avg:.2f}",
                        "% of Total": f"{(total / stats['total_amount'] * 100):.1f}%"
                    })
                
                vendor_df = pd.DataFrame(vendor_summary)
                st.dataframe(vendor_df, use_container_width=True)
            else:
                st.info("No vendor data available")
        
        with tab3:
            if patterns.get('monthly_trends'):
                monthly_summary = []
                for month, data in sorted(patterns['monthly_trends'].items()):
                    monthly_summary.append({
                        "Month": month,
                        "Total Spent": f"${data['total']:.2f}",
                        "Receipt Count": data['count'],
                        "Average per Receipt": f"${data['average']:.2f}"
                    })
                
                monthly_df = pd.DataFrame(monthly_summary)
                st.dataframe(monthly_df, use_container_width=True)
            else:
                st.info("No monthly data available")
        
        # Export Analytics Data
        st.markdown("---")
        st.subheader("📥 Export Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Summary Report"):
                report_data = {
                    "summary": {
                        "total_receipts": stats['count'],
                        "total_spending": float(stats['total_amount']),
                        "average_amount": float(stats['average_amount']),
                        "date_range": f"{min(r.transaction_date for r in filtered_receipts)} to {max(r.transaction_date for r in filtered_receipts)}"
                    },
                    "category_breakdown": {k: float(v) for k, v in patterns.get('category_breakdown', {}).items()},
                    "top_vendors": [(v, float(a)) for v, a in patterns.get('top_vendors', [])],
                    "insights": insights
                }
                
                import json
                json_report = json.dumps(report_data, indent=2, default=str)
                st.download_button(
                    label="Download Summary Report",
                    data=json_report,
                    file_name=f"analytics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📈 Export Chart Data"):
                chart_data = {
                    "monthly_trends": patterns.get('monthly_trends', {}),
                    "category_breakdown": {k: float(v) for k, v in patterns.get('category_breakdown', {}).items()},
                    "day_patterns": {k: dict(v) for k, v in patterns.get('day_of_week_patterns', {}).items()}
                }
                
                import json
                chart_json = json.dumps(chart_data, indent=2, default=str)
                st.download_button(
                    label="Download Chart Data",
                    data=chart_json,
                    file_name=f"chart_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📋 Export Raw Analytics Data"):
                analytics_data = []
                for receipt in filtered_receipts:
                    analytics_data.append({
                        "id": receipt.id,
                        "vendor": receipt.vendor,
                        "date": receipt.transaction_date.isoformat(),
                        "amount": float(receipt.amount),
                        "category": receipt.category,
                        "currency": receipt.currency,
                        "day_of_week": receipt.transaction_date.strftime("%A"),
                        "month": receipt.transaction_date.strftime("%Y-%m")
                    })
                
                analytics_df = pd.DataFrame(analytics_data)
                csv_data = analytics_df.to_csv(index=False)
                st.download_button(
                    label="Download Raw Data CSV",
                    data=csv_data,
                    file_name=f"analytics_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        logger.error(f"Error in Analytics Dashboard: {str(e)}")
        st.error(f"An error occurred while generating analytics: {str(e)}")
        st.info("Please try refreshing the page or check if there's sufficient data.")

if __name__ == "__main__":
    main()
