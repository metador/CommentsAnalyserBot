import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union
import asyncio
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import yaml
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
from charts import create_trend_visualizations, create_topic_trends_chart
from utils import NumpyJSONEncoder, convert_numpy_types, clean_json_response, filter_comments




# Load environment variables
#load_dotenv()

class SocialMediaInsightsBot:
    """
    A comprehensive social media comments analysis tool using LLMs
    """
    
    def __init__(self, model_name: str = 'gpt-4o-mini', temperature: float = 0.1):
        """
        Initialize the Social Media Insights Bot
        
        Args:
            model_name: The LLM model to use
            temperature: Temperature for LLM responses
        """
        self.model = init_chat_model(model=model_name, temperature=temperature)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.batch_size = 50  # Process comments in batches
        self.vectorstore = None

        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
    
    def create_trend_visualizations(self, comments: List[Dict], topics: List[Dict], output_dir: str = "./reports"):
        topics = self._analyze_topic_trends(comments)
        create_trend_visualizations(comments, topics, output_dir)

    
    def _load_prompt_templates(self) -> Dict:
        """Load prompt templates from YAML file"""
        try:
            with open('./prompts/socialMediaInsightsBot.yaml', 'r') as file:
                print('Loaded prompt templates  from ./prompts/socialMediaInsightsBot.yaml')
                loaded_data = yaml.safe_load(file)
                
                return loaded_data
        except FileNotFoundError:
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict:
        """Default prompt templates for social media analysis"""
        return { 'sentiment_analysis': """
            You are an expert social media analyst. Analyze the sentiment of the following comments.
            Classify each comment as: POSITIVE, NEGATIVE, NEUTRAL
            Also provide a confidence score (0-1) for each classification.
            
            Comments:
            {comments}
            
            Return your analysis in JSON format:
            {
                "sentiment_summary": {
                    "positive_count": float,
                    "negative_count": float,
                    "neutral_count": float,
                    "overall_sentiment": "positive/negative/neutral"
                },
                "detailed_analysis": [
                    {
                        "comment": "text",
                        "sentiment": "positive/negative/neutral",
                        "confidence": float,
                        "reasoning": "brief explanation"
                    }
                ]
            }
            """,
            
            'theme_extraction': """
            You are an expert in content analysis. Identify the main themes and topics from these social media comments.
            Look for recurring patterns, concerns, interests, and discussion topics.
            
            Comments:
            {comments}
            
            Return your analysis in JSON format:
            {
                "main_themes": [
                    {
                        "theme": "theme name",
                        "frequency": int,
                        "description": "brief description",
                        "sample_comments": ["comment1", "comment2"]
                    }
                ],
                "trending_topics": ["topic1", "topic2", "topic3"],
                "key_insights": ["insight1", "insight2", "insight3"]
            }
            """,
            
            'insights_summary': """
            You are a senior data analyst specializing in social media insights. 
            Based on the provided analysis data, create a comprehensive executive summary.
            
            Analysis Data:
            {analysis_data}
            
            Provide a structured summary including:
            1. Overall sentiment distribution
            2. Key themes and topics
            3. Notable trends or patterns
            4. Actionable insights for stakeholders
            5. Recommendations for next steps
            
            Format your response as a clear, executive-level report.
            """
        }
    
    def load_comments(self, data_source: Union[str, List[str]], source_type: str = 'auto', 
                     apply_filtering: bool = True) -> List[Dict]:
        """
        Load social media comments from various sources
        
        Args:
            data_source: File path, list of comments, or data
            source_type: 'csv', 'json', 'list', 'auto'
            apply_filtering: Whether to filter out unwanted comments (tags, names only, numbers only)
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        
        if source_type == 'auto':
            if isinstance(data_source, str):
                if data_source.endswith('.csv'):
                    source_type = 'csv'
                elif data_source.endswith('.json'):
                    source_type = 'json'
            elif isinstance(data_source, list):
                source_type = 'list'
        
        if source_type == 'csv':
            df = pd.read_csv(data_source)
            # Handle specific columns for engagements.csv or general comment columns
            text_columns = ['comment_text', 'comment', 'text', 'content', 'message', 'post']
            text_col = None
            for col in text_columns:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col:
                for idx, row in df.iterrows():
                    # Parse timestamp if available
                    timestamp = None
                    if 'timestamp' in row and pd.notna(row['timestamp']):
                        try:
                            timestamp = pd.to_datetime(row['timestamp'])
                        except:
                            timestamp = None
                    
                    comments.append({
                        'id': idx,
                        'text': str(row[text_col]),
                        'timestamp': timestamp,
                        'metadata': {k: v for k, v in row.items() if k != text_col}
                    })
        
        elif source_type == 'json':
            with open(data_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, str):
                        comments.append({'id': idx, 'text': item, 'metadata': {}})
                    elif isinstance(item, dict):
                        text = item.get('text', item.get('comment', item.get('content', '')))
                        comments.append({
                            'id': idx,
                            'text': str(text),
                            'metadata': {k: v for k, v in item.items() if k not in ['text', 'comment', 'content']}
                        })
        
        elif source_type == 'list':
            for idx, comment in enumerate(data_source):
                if isinstance(comment, str):
                    comments.append({'id': idx, 'text': comment, 'metadata': {}})
                elif isinstance(comment, dict):
                    text = comment.get('text', comment.get('comment', str(comment)))
                    comments.append({
                        'id': idx,
                        'text': str(text),
                        'metadata': {k: v for k, v in comment.items() if k not in ['text', 'comment']}
                    })
        
        print(f"Loaded {len(comments)} comments from {data_source}")
        
        # Apply filtering if enabled
        if apply_filtering:
            print("Applying comment filtering (removing tags, names only, numbers only)...")
            comments = filter_comments(comments)
        
        return comments
    
    def create_vectorstore(self, comments: List[Dict], persist_directory: str = './vectorstore/chroma_social_comments'):
        """
        Create a vectorstore from comments for RAG-based analysis
        
        Args:
            comments: List of comment dictionaries
            persist_directory: Directory to persist the vectorstore
        """
        # Convert comments to documents
        documents = []
        for comment in comments:
            doc = Document(
                page_content=comment['text'],
                metadata={
                    'comment_id': comment['id'],
                    **comment.get('metadata', {})
                }
            )
            documents.append(doc)
        
        # Split documents if they're too long
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        # Create vectorstore
        os.makedirs(persist_directory, exist_ok=True)
        self.vectorstore = Chroma(
            collection_name='social_media_comments',
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # Add documents in batches to avoid memory issues
        batch_size = 100
        for i in tqdm(range(0, len(docs), batch_size), desc="Creating vectorstore"):
            batch_docs = docs[i:i+batch_size]
            self.vectorstore.add_documents(batch_docs)
        
        print(f"Created vectorstore with {len(docs)} document chunks")
    
    def analyze_sentiment_batch(self, comments_batch: List[Dict]) -> Dict:
        """
        Analyze sentiment for a batch of comments
        
        Args:
            comments_batch: Batch of comments to analyze
            
        Returns:
            Sentiment analysis results
        """
        comments_text = "\n".join([f"{i+1}. {comment['text'][:200]}..." for i, comment in enumerate(comments_batch)])

        
        prompt = self.prompt_templates['sentiment_analysis'].format(comments=comments_text) + self.prompt_templates['sentiment_output']
        
        messages = [
                SystemMessage(content="You are an expert social media sentiment analyst."),
                HumanMessage(content=prompt)
            ]
        
        try:
            response = self.model.invoke(messages)
            
            # Clean and parse JSON response
            cleaned_content = clean_json_response(response.content)
            result = json.loads(cleaned_content)
            
            return result
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error in sentiment analysis: {e}")
            print(f"Raw response: {response.content[:500] if 'response' in locals() else 'No response'}")
            # Fallback to simple analysis
            return {
                "sentiment_summary": {
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": len(comments_batch),
                    "overall_sentiment": "neutral"
                },
                "detailed_analysis": []
            }
    
    def extract_themes_batch(self, comments_batch: List[Dict]) -> Dict:
        """
        Extract themes from a batch of comments
        
        Args:
            comments_batch: Batch of comments to analyze
            
        Returns:
            Theme extraction results
        """
        comments_text = "\n".join([f"{i+1}. {comment['text']}" for i, comment in enumerate(comments_batch)])
        
        prompt = self.prompt_templates['theme_extraction'].format(comments=comments_text) + self.prompt_templates['theme_extraction_output']
        
        try:
            response = self.model.invoke([
                SystemMessage(content="You are an expert in social media content analysis and theme extraction."),
                HumanMessage(content=prompt)
            ])
            
            # Clean and parse JSON response
            cleaned_content = clean_json_response(response.content)
            result = json.loads(cleaned_content)
            return result
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error in theme extraction: {e}")
            print(f"Raw response: {response.content[:500] if 'response' in locals() else 'No response'}")
            return {
                "main_themes": [],
                "trending_topics": [],
                "key_insights": []
            }
    
    def analyze_comments(self, comments: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive analysis of social media comments
        
        Args:
            comments: List of comment dictionaries
            
        Returns:
            Complete analysis results
        """
        print(f"Starting analysis of {len(comments)} comments...")
        
        # Initialize results
        sentiment_results = {
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "detailed_analyses": []
        }
        
        theme_results = {
            "all_themes": [],
            "all_topics": [],
            "all_insights": []
        }
        
        # Process comments in batches
        num_batches = (len(comments) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, len(comments), self.batch_size), desc="Analyzing comments"):
            batch = comments[i:i+self.batch_size]
            
            # Sentiment analysis
            sentiment_batch = self.analyze_sentiment_batch(batch)
            sentiment_results["positive_count"] += sentiment_batch.get("sentiment_summary", {}).get("positive_count", 0)
            sentiment_results["negative_count"] += sentiment_batch.get("sentiment_summary", {}).get("negative_count", 0)
            sentiment_results["neutral_count"] += sentiment_batch.get("sentiment_summary", {}).get("neutral_count", 0)
            sentiment_results["detailed_analyses"].extend(sentiment_batch.get("detailed_analysis", []))
            
            # Theme extraction
            theme_batch = self.extract_themes_batch(batch)
            theme_results["all_themes"].extend(theme_batch.get("main_themes", []))
            theme_results["all_topics"].extend(theme_batch.get("trending_topics", []))
            theme_results["all_insights"].extend(theme_batch.get("key_insights", []))
        
        # Aggregate results
        total_comments = sentiment_results["positive_count"] + sentiment_results["negative_count"] + sentiment_results["neutral_count"]
        
        # Calculate overall sentiment
        if total_comments > 0:
            pos_pct = sentiment_results["positive_count"] / total_comments
            neg_pct = sentiment_results["negative_count"] / total_comments
            
            if pos_pct > neg_pct and pos_pct > 0.4:
                overall_sentiment = "positive"
            elif neg_pct > pos_pct and neg_pct > 0.4:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
        else:
            overall_sentiment = "neutral"
        
        # Aggregate themes by frequency
        theme_counter = Counter()
        for theme in theme_results["all_themes"]:
            if isinstance(theme, dict) and 'theme' in theme:
                theme_counter[theme['theme']] += theme.get('frequency', 1)
        
        topic_counter = Counter(theme_results["all_topics"])
        
        final_results = {
            "analysis_metadata": {
                "total_comments_analyzed": len(comments),
                "analysis_date": datetime.now().isoformat(),
                "batch_size": self.batch_size,
                "model_used": "gpt-4o-mini"
            },
            "sentiment_analysis": {
                "overall_sentiment": overall_sentiment,
                "sentiment_distribution": {
                    "positive": sentiment_results["positive_count"],
                    "negative": sentiment_results["negative_count"],
                    "neutral": sentiment_results["neutral_count"]
                },
                "sentiment_percentages": {
                    "positive": round(pos_pct * 100, 2) if total_comments > 0 else 0,
                    "negative": round(neg_pct * 100, 2) if total_comments > 0 else 0,
                    "neutral": round((sentiment_results["neutral_count"] / total_comments) * 100, 2) if total_comments > 0 else 0
                }
            },
            "theme_analysis": {
                "top_themes": [{"theme": theme, "frequency": freq} for theme, freq in theme_counter.most_common(10)],
                "trending_topics": [{"topic": topic, "mentions": freq} for topic, freq in topic_counter.most_common(15)],
                "unique_insights": list(set(theme_results["all_insights"]))
            }
        }
        
        return final_results
    
    def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate an executive summary of the analysis
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            Executive summary as a string
        """
        
        
        prompt = self.prompt_templates['insights_summary'].format(
            analysis_data=json.dumps(analysis_results, indent=2, cls=NumpyJSONEncoder)
        ) + self.prompt_templates['insights_summary_output']
        
        response = self.model.invoke([
            SystemMessage(content="You are a senior data analyst creating executive summaries. Return a well-formatted text summary, not JSON."),
            HumanMessage(content=prompt)
        ])
        
        return response.content
    
    def query_insights(self, query: str, top_k: int = 10) -> str:
        """
        Query the vectorstore for specific insights
        
        Args:
            query: Question about the comments
            top_k: Number of relevant comments to retrieve
            
        Returns:
            AI response to the query
        """
        if not self.vectorstore:
            return "Vectorstore not created. Please run create_vectorstore first."
        
        # Retrieve relevant documents
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        relevant_docs = retriever.get_relevant_documents(query)
        
        # Combine context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        sys_prompt = f"""
        You are an expert social media analyst. Answer the user's question based on the following social media comments.
        If the answer is not in the provided context, say so clearly.
        
        Relevant Comments:
        {context}
        """
        
        response = self.model.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=query)
        ])
        
        return response.content
    
    def analyze_temporal_trends(self, comments: List[Dict]) -> Dict[str, Any]:
        """
        Analyze how sentiment and topics change over time
        
        Args:
            comments: List of comment dictionaries with timestamps
            
        Returns:
            Temporal analysis results
        """
        # Filter comments with timestamps
        timestamped_comments = [c for c in comments if c.get('timestamp')]
        
        if not timestamped_comments:
            return {"error": "No comments with valid timestamps found"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(timestamped_comments)
        df['date'] = df['timestamp'].dt.date
        df['week'] = df['timestamp'].dt.to_period('W')
        df['month'] = df['timestamp'].dt.to_period('M')
        
        # Group by time periods
        daily_counts = df.groupby('date').size()
        weekly_counts = df.groupby('week').size()
        monthly_counts = df.groupby('month').size()
        
        # Analyze sentiment trends over time (simplified for this example)
        # In a full implementation, you'd run sentiment analysis on each time period
        time_periods = ['daily', 'weekly', 'monthly']
        
        results = {
            "temporal_overview": {
                "total_comments": len(timestamped_comments),
                "date_range": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat()
                },
                "total_days": (df['timestamp'].max() - df['timestamp'].min()).days + 1
            },
            "volume_trends": {
                "daily_avg": float(daily_counts.mean()),
                "peak_day": {
                    "date": str(daily_counts.idxmax()),
                    "count": int(daily_counts.max())
                },
                "lowest_day": {
                    "date": str(daily_counts.idxmin()),
                    "count": int(daily_counts.min())
                }
            },
            "patterns": {
                "busiest_day_of_week": str(df['timestamp'].dt.day_name().value_counts().index[0]),
                "busiest_hour": int(df['timestamp'].dt.hour.value_counts().index[0])
            }
        }
        
        return results
    
    def generate_metrics_report(self, comments: List[Dict], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics-driven report
        
        Args:
            comments: List of comment dictionaries
            analysis_results: Results from analyze_comments
            
        Returns:
            Comprehensive metrics report
        """
        # Get temporal analysis
        temporal_analysis = self.analyze_temporal_trends(comments)
        
        # Calculate engagement metrics
        timestamped_comments = [c for c in comments if c.get('timestamp')]
        
        if timestamped_comments:
            df = pd.DataFrame(timestamped_comments)
            
            # Weekly trend analysis
            df['week'] = df['timestamp'].dt.to_period('W')
            avg_length = lambda x: x.str.len().mean()
            weekly_stats = df.groupby('week').agg({
                'text': ['count', avg_length]
            }).round(2)
            
            weekly_trends = []
            for week, stats in weekly_stats.iterrows():
                weekly_trends.append({
                    "week": str(week),
                    "comment_count": int(stats[('text', 'count')]),

                })
        else:
            weekly_trends = []
        
        # Extract topics and their frequencies over time
        topics_over_time = self._analyze_topic_trends(comments)
        
        # Create comprehensive metrics report
        metrics_report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "analysis_period": temporal_analysis.get("temporal_overview", {}).get("date_range", {}),
                "total_comments": len(comments)
            },
            "engagement_metrics": {
                "temporal_analysis": temporal_analysis,
                "weekly_trends": weekly_trends[:8],  # Last 8 weeks
                "volume_statistics": {
                    "daily_average": temporal_analysis.get("volume_trends", {}).get("daily_avg", 0),
                    "peak_activity": temporal_analysis.get("volume_trends", {}).get("peak_day", {}),
                    "engagement_patterns": temporal_analysis.get("patterns", {})
                }
            },
            "content_insights": {
                "sentiment_evolution": analysis_results.get("sentiment_analysis", {}),
                "trending_topics": analysis_results.get("theme_analysis", {}).get("trending_topics", []),
                "topic_trends_over_time": topics_over_time
            },
            "key_metrics": {
                "sentiment_score": self._calculate_sentiment_score(analysis_results.get("sentiment_analysis", {})),
                "engagement_score": self._calculate_engagement_score(temporal_analysis),
                "topic_diversity": len(analysis_results.get("theme_analysis", {}).get("top_themes", []))
            }
        }
        
        return metrics_report
    
    def _analyze_topic_trends(self, comments: List[Dict]) -> List[Dict]:
        """Analyze how topics trend over time periods"""
        timestamped_comments = [c for c in comments if c.get('timestamp')]
        
        if not timestamped_comments:
            return []
        
        # Group comments by week and extract basic topic indicators
        df = pd.DataFrame(timestamped_comments)
        df['week'] = df['timestamp'].dt.to_period('W')
        
        # Simple keyword-based topic extraction for trends
        topic_keywords = {
            "quality": ["quality", "good", "great", "excellent", "amazing", "perfect"],
            "price": ["price", "expensive", "cheap", "cost", "money", "value"],
            "service": ["service", "support", "help", "customer", "staff"],
            "delivery": ["delivery", "shipping", "fast", "slow", "arrived"],
            "product": ["product", "item", "love", "hate", "recommend"]
        }
        
        weekly_topics = []
        for week in df['week'].unique():
            week_comments = df[df['week'] == week]['text'].str.lower()
            
            topic_counts = {}
            for topic, keywords in topic_keywords.items():
                count = sum(week_comments.str.contains('|'.join(keywords), na=False))
                topic_counts[topic] = int(count)  # Convert numpy int to Python int
            
            weekly_topics.append({
                "period": str(week),
                "topics": topic_counts,
                "total_comments": int(len(week_comments))  # Ensure it's a Python int
            })
        
        return weekly_topics
    
    def _calculate_sentiment_score(self, sentiment_analysis: Dict) -> float:
        """Calculate overall sentiment score (0-100)"""
        if not sentiment_analysis.get("sentiment_percentages"):
            return 50.0
        
        pos = sentiment_analysis["sentiment_percentages"].get("positive", 0)
        neg = sentiment_analysis["sentiment_percentages"].get("negative", 0)
        
        # Score from 0-100 where 50 is neutral
        score = 50 + (pos - neg) / 2
        return round(max(0, min(100, score)), 2)
    
    def _calculate_engagement_score(self, temporal_analysis: Dict) -> float:
        """Calculate engagement score based on volume and consistency"""
        if temporal_analysis.get("error"):
            return 0.0
        
        daily_avg = temporal_analysis.get("volume_trends", {}).get("daily_avg", 0)
        
        # Simple engagement score (can be enhanced with more metrics)
        if daily_avg > 100:
            return 90.0
        elif daily_avg > 50:
            return 75.0
        elif daily_avg > 20:
            return 60.0
        elif daily_avg > 5:
            return 40.0
        else:
            return 25.0
    

    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save analysis results to a JSON file"""
        # Convert numpy types to Python native types
        clean_results = convert_numpy_types(results)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
        print(f"Results saved to {filepath}")


def analyze_social_media_comments(
    data_source: Union[str, List[str]], 
    source_type: str = 'auto',
    create_vectorstore: bool = True,
    save_path: str = None,
    generate_trends: bool = True,
    create_visualizations: bool = True,
    limit: int = None,
    apply_filtering: bool = True
    
) -> Dict[str, Any]:
    """
    Main function to analyze social media comments with metrics-driven trend analysis
    
    Args:
        data_source: Path to data file or list of comments
        source_type: Type of data source ('csv', 'json', 'list', 'auto')
        create_vectorstore: Whether to create a vectorstore for RAG queries
        save_path: Path to save results (optional)
        generate_trends: Whether to generate temporal trend analysis
        create_visualizations: Whether to create trend visualizations
        limit: Maximum number of comments to analyze (optional)
        apply_filtering: Whether to filter out unwanted comments (tags, names only, numbers only)
        
    Returns:
        Complete analysis results including metrics report
    """
    
    # Initialize the bot
    bot = SocialMediaInsightsBot()
    
    # Load comments
    comments = bot.load_comments(data_source, source_type, apply_filtering)
    if limit:
        comments = comments[:limit]
        
    if len(comments) == 0:
        raise ValueError("No comments loaded. Please check your data source.")
    
    print(f"Loaded {len(comments)} comments for analysis")

    # Create vectorstore for RAG queries
    if create_vectorstore:
        bot.create_vectorstore(comments)
    
    # Perform comprehensive analysis
    print("Performing sentiment and theme analysis...")
    results = bot.analyze_comments(comments)
    
    # Generate metrics-driven report with temporal trends
    if generate_trends:
        print("Generating metrics-driven trend analysis...")
        metrics_report = bot.generate_metrics_report(comments, results)
        results["metrics_report"] = metrics_report
    
    # Generate executive summary
    print("Generating executive summary...")
    executive_summary = bot.generate_executive_summary(results)
    results["executive_summary"] = executive_summary
    
    # Create visualizations
    if create_visualizations:
        print("Creating trend visualizations...")
        bot.create_trend_visualizations(comments, "./reports")

        
    
    # Save results if path provided
    if save_path:
        bot.save_results(results, save_path)
    
    return results, bot


# Example usage
if __name__ == "__main__":
    # Example with sample data
    sample_comments = [
        "I love this new product! It's amazing and works perfectly.",
        "Terrible experience. Would not recommend to anyone.",
        "It's okay, nothing special but does the job.",
        "Best purchase I've made this year! Highly recommend!",
        "Had some issues initially but customer service was helpful.",
        # Add more sample comments here...
    ] * 200  # Simulate 1000 comments
    
    # Analyze the comments
    results, bot = analyze_social_media_comments(
        data_source=sample_comments,
        source_type='list',
        create_vectorstore =False,
        save_path='./analysis_results.json',
        generate_trends=True,
        create_visualizations=True,
        limit=1000
    )
    
   