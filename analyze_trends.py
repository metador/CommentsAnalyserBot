#!/usr/bin/env python3
"""
Engagement Comments Analysis Script
Analyzes social media comments from engagements.csv with metrics-driven trend reporting
"""

import sys
import os
sys.path.append('./src')

from socialMediaInsightsBot import analyze_social_media_comments, SocialMediaInsightsBot
import json
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*70 + f"\n {title}\n" + "="*70)

def print_metrics_summary(metrics_report):
    """Print a formatted summary of key metrics"""
    print_section_header("METRICS OVERVIEW")
    
    metadata = metrics_report.get("report_metadata", {})
    key_metrics = metrics_report.get("key_metrics", {})
    
    print(f"Report Generated: {metadata.get('generated_at', 'N/A')}")
    print(f"Analysis Period: {metadata.get('analysis_period', {}).get('start', 'N/A')} to {metadata.get('analysis_period', {}).get('end', 'N/A')}")
    print(f"Total Comments: {metadata.get('total_comments', 0):,}")
    
    print(f"\nKEY PERFORMANCE INDICATORS")
    print(f"   • Sentiment Score: {key_metrics.get('sentiment_score', 0)}/100")
    print(f"   • Engagement Score: {key_metrics.get('engagement_score', 0)}/100")
    print(f"   • Topic Diversity: {key_metrics.get('topic_diversity', 0)} unique themes")

def print_temporal_trends(metrics_report):
    """Print temporal trend analysis"""
    print_section_header("TEMPORAL TRENDS ANALYSIS")
    
    engagement_metrics = metrics_report.get("engagement_metrics", {})
    temporal_analysis = engagement_metrics.get("temporal_analysis", {})
    
    if temporal_analysis.get("error"):
        print("No temporal data available for analysis")
        return
    
    volume_trends = temporal_analysis.get("volume_trends", {})
    patterns = temporal_analysis.get("patterns", {})
    
    print(f"VOLUME STATISTICS")
    print(f"   • Daily Average: {volume_trends.get('daily_avg', 0):.1f} comments/day")
    print(f"   • Peak Activity: {volume_trends.get('peak_day', {}).get('count', 0)} comments on {volume_trends.get('peak_day', {}).get('date', 'N/A')}")
    print(f"   • Lowest Activity: {volume_trends.get('lowest_day', {}).get('count', 0)} comments on {volume_trends.get('lowest_day', {}).get('date', 'N/A')}")
    
    print(f"\nENGAGEMENT PATTERNS")
    print(f"   • Busiest Day: {patterns.get('busiest_day_of_week', 'N/A')}")
    print(f"   • Peak Hour: {patterns.get('busiest_hour', 'N/A')}:00")

def print_weekly_trends(metrics_report):
    """Print weekly trend breakdown"""
    print_section_header("WEEKLY TRENDS")
    
    weekly_trends = metrics_report.get("engagement_metrics", {}).get("weekly_trends", [])
    
    if not weekly_trends:
        print("No weekly trend data available")
        return
    
    print(f"Recent Weekly Activity (Last {len(weekly_trends)} weeks):\n")
    
    for i, week in enumerate(weekly_trends[-8:], 1):  # Show last 8 weeks
        print(f"   Week {i}: {week['week']}")
        print(f"      • Comments: {week['comment_count']:,}")
        #print(f"      • Avg Length: {week['avg_comment_length']:.1f} characters")

def print_topic_trends(metrics_report):
    """Print topic trend analysis"""
    print_section_header("TOPIC TRENDS OVER TIME")
    
    topics_over_time = metrics_report.get("content_insights", {}).get("topic_trends_over_time", [])
    
    if not topics_over_time:
        print(" No topic trend data available")
        return
    
    print("Topic Evolution (Recent periods):\n")
    
    # Show last few periods
    for period_data in topics_over_time[-4:]:  # Last 4 periods
        print(f"   Period: {period_data['period']} ({period_data['total_comments']} comments)")
        topics = period_data['topics']
        
        # Sort topics by frequency
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        
        for topic, count in sorted_topics[:3]:  # Top 3 topics
            if count > 0:
                percentage = (count / period_data['total_comments']) * 100
                print(f"      • {topic.capitalize()}: {count} mentions ({percentage:.1f}%)")

def main():
    """Main analysis function for engagements.csv"""
    print_section_header("SOCIAL MEDIA ENGAGEMENT ANALYSIS")
    
    csv_path = './documents/csv/engagements.csv'
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        print("Please make sure the engagements.csv file exists in the documents/csv/ directory")
        return
    
    print(f" Analyzing comments from: {csv_path}")
    
    try:
        # Load and display basic file info
        df = pd.read_csv(csv_path)
        print(f"File contains {len(df):,} records")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Check for required columns
        required_columns = ['timestamp', 'media_id', 'media_caption', 'comment_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return
        
        print(f"All required columns present: {required_columns}")
        
        # Perform comprehensive analysis
        print("\nStarting comprehensive analysis...")
        
        results, bot = analyze_social_media_comments(
            data_source=csv_path,
            source_type='csv',
            create_vectorstore=False,
            save_path='./engagements_analysis_results.json',
            generate_trends=True,
            create_visualizations=True,
            limit=100,
            apply_filtering=True
        )
        
        # Display results
        print_section_header("ANALYSIS RESULTS")
        
        # Basic statistics
        sentiment = results.get('sentiment_analysis', {})
        theme_analysis = results.get('theme_analysis', {})
        
        print(f"SENTIMENT ANALYSIS")
        print(f"   Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A').upper()}")
        
        sentiment_pct = sentiment.get('sentiment_percentages', {})
        print(f"   • Positive: {sentiment_pct.get('positive', 0):.1f}%")
        print(f"   • Negative: {sentiment_pct.get('negative', 0):.1f}%")
        print(f"   • Neutral: {sentiment_pct.get('neutral', 0):.1f}%")
        
        print(f"\n TOP THEMES")
        top_themes = theme_analysis.get('top_themes', [])[:5]
        for i, theme in enumerate(top_themes, 1):
            print(f"   {i}. {theme.get('theme', 'N/A')} (frequency: {theme.get('frequency', 0)})")
        
        print(f"\n TRENDING TOPICS")
        trending_topics = theme_analysis.get('trending_topics', [])[:5]
        for i, topic in enumerate(trending_topics, 1):
            print(f"   {i}. {topic.get('topic', 'N/A')} ({topic.get('mentions', 0)} mentions)")
        
        # Display metrics report if available
        if 'metrics_report' in results:
            metrics_report = results['metrics_report']
            print_metrics_summary(metrics_report)
            print_temporal_trends(metrics_report)
            print_weekly_trends(metrics_report)
            print_topic_trends(metrics_report)
        
        # Executive summary
        print_section_header("EXECUTIVE SUMMARY")
        print(results.get('executive_summary', 'No executive summary available'))
        
        # Interactive query demonstration
        print_section_header("SAMPLE INSIGHTS QUERIES")
        
        if bot and bot.vectorstore:
            sample_queries = [
                "What are customers saying about product quality?",
                "What complaints are most common?",
                "What do people love most about the products?",
                "Are there any delivery or shipping issues mentioned?"
            ]
            
            for query in sample_queries:
                response = bot.query_insights(query, top_k=3)
                print(f"\n Query: {query}")

        print_section_header("ANALYSIS COMPLETE")
        print("Generated Files:")
        print("   • engagements_analysis_results.json - Complete analysis results")
        print("   • ./reports/engagement_trends.png - Temporal trend visualizations")
        print("   • ./reports/topic_trends_over_time.png - Topic trends line chart with trending highlights")
        print("   • ./reports/comment_wordcloud.png - Word cloud visualization")
        print("   • ./vectorstore/chroma_social_comments/ - Vector database for queries")
        
        print(f"\n Analysis completed successfully!")
        print(f"Processed {len(df):,} comments with temporal trend analysis")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    