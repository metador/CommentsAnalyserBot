# Social Media Engagement Analysis with Metrics-Driven Trends

This updated version of the Social Media Insights Bot now includes comprehensive **metrics-driven trend analysis** to track how sentiment, topics, and engagement patterns shift over time.

## Quick Start

### Analyze the Engagements CSV File

```bash
# Install dependencies
pip install -r requirements.txt

# Run the trends analysis
python analyze_trends.py
```

This will automatically process `./documents/csv/engagements.csv` and generate:
- **Temporal trend analysis** showing how engagement changes over time
- **Sentiment evolution** tracking positive/negative patterns
- **Topic trends** revealing which themes gain/lose popularity
- **Interactive visualizations** (saved as PNG files)
- **Comprehensive metrics report** with actionable insights

### Running with Comment Limit

By default, analyze_trends.py processes up to 1000 comments for analysis. You can modify this limit by editing the script:

```python
# In analyze_trends.py, line 142:
results, bot = analyze_social_media_comments(
    data_source=csv_path,
    source_type='csv',
    create_vectorstore=False,
    save_path='./engagements_analysis_results.json',
    generate_trends=True,
    create_visualizations=True,
    limit=1000  # Change this value to process more or fewer comments
)
```

For large datasets, processing all comments may take significant time. The limit helps ensure reasonable processing times while still providing meaningful insights.

## New Features

### 1. Temporal Trend Analysis
- **Daily/Weekly/Monthly** comment volume patterns
- **Peak activity detection** (busiest days and hours)
- **Engagement consistency** metrics
- **Time-based sentiment evolution**

### 2. Metrics-Driven Reporting
- **Sentiment Score** (0-100 scale)
- **Engagement Score** (activity-based scoring)
- **Topic Diversity** (theme variety metrics)
- **Volume Statistics** (averages, peaks, trends)

### 3. Advanced Visualizations
- **Daily comment volume** line charts
- **Day-of-week** activity patterns
- **Hourly engagement** heatmaps
- **Comment length** distributions
- **Word clouds** of popular terms

### 4. Topic Evolution Tracking
- **Weekly topic trends** showing which themes are gaining/losing traction
- **Keyword-based categorization** (quality, price, service, delivery, product)
- **Percentage-based tracking** of topic mentions over time

## File Structure

```
CommentsAnalyserBot/
├── analyze_trends.py               # Main trends analysis script for engagements.csv
├── src/
│   └── socialMediaInsightsBot.py   # Enhanced bot with temporal analysis
├── documents/csv/
│   └── engagements.csv              # Your comment data (timestamp, media_id, media_caption, comment_text)
├── reports/                         # Generated visualizations
│   ├── engagement_trends.png
│   └── comment_wordcloud.png
├── vectorstore/                     # Vector database for queries
└── requirements.txt                 # Updated with visualization dependencies
```

## API Usage

### Basic Analysis with Trends

```python
from src.socialMediaInsightsBot import analyze_social_media_comments

# Analyze with full metrics and trends
results, bot = analyze_social_media_comments(
    data_source='./documents/csv/engagements.csv',
    source_type='csv',
    create_vectorstore=True,
    generate_trends=True,
    create_visualizations=True,
    save_path='./analysis_results.json'
)

# Access metrics report
metrics = results['metrics_report']
print(f"Sentiment Score: {metrics['key_metrics']['sentiment_score']}/100")
print(f"Engagement Score: {metrics['key_metrics']['engagement_score']}/100")
```

### Query Specific Insights

```python
# Interactive querying of the comment database
insights = bot.query_insights("What are the main complaints about delivery?")
print(insights)

# Topic-specific analysis
quality_insights = bot.query_insights("How do customers feel about product quality?")
print(quality_insights)
```

### Custom Time Period Analysis

```python
# Filter comments by date range
import pandas as pd
from datetime import datetime, timedelta

# Load and filter data
df = pd.read_csv('./documents/csv/engagements.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Last 30 days only
recent_date = df['timestamp'].max() - timedelta(days=30)
recent_comments = df[df['timestamp'] >= recent_date]

# Convert to format expected by the bot
comments_list = []
for _, row in recent_comments.iterrows():
    comments_list.append({
        'id': row.name,
        'text': row['comment_text'],
        'timestamp': row['timestamp'],
        'metadata': {
            'media_id': row['media_id'],
            'media_caption': row['media_caption']
        }
    })

# Analyze recent comments only
results, bot = analyze_social_media_comments(
    data_source=comments_list,
    source_type='list',
    generate_trends=True
)
```

## Metrics Report Structure

The enhanced analysis now provides a comprehensive metrics report:

```json
{
  "metrics_report": {
    "report_metadata": {
      "generated_at": "2025-01-27T10:30:00",
      "analysis_period": {
        "start": "2025-03-01T00:13:57",
        "end": "2025-03-15T23:45:30"
      },
      "total_comments": 102806
    },
    "engagement_metrics": {
      "temporal_analysis": {
        "volume_trends": {
          "daily_avg": 75.2,
          "peak_day": {"date": "2025-03-08", "count": 234},
          "lowest_day": {"date": "2025-03-12", "count": 12}
        },
        "patterns": {
          "busiest_day_of_week": "Friday",
          "busiest_hour": 14
        }
      },
      "weekly_trends": [
        {
          "week": "2025-03-03/2025-03-09",
          "comment_count": 523,
          "avg_comment_length": 67.3
        }
      ]
    },
    "content_insights": {
      "sentiment_evolution": "...",
      "topic_trends_over_time": [
        {
          "period": "2025-03-03/2025-03-09",
          "topics": {
            "quality": 45,
            "price": 23,
            "service": 67,
            "delivery": 12,
            "product": 89
          },
          "total_comments": 523
        }
      ]
    },
    "key_metrics": {
      "sentiment_score": 72.5,
      "engagement_score": 85.0,
      "topic_diversity": 15
    }
  }
}
```

## Sample Output

When you run `python analyze_trends.py`, you'll see:

```
======================================================================
 SOCIAL MEDIA ENGAGEMENT ANALYSIS
======================================================================

Analyzing comments from: ./documents/csv/engagements.csv
File contains 102,806 records
Date range: 2025-03-01 00:13:57.153000+00:00 to 2025-03-15 23:45:30.879000+00:00
All required columns present: ['timestamp', 'media_id', 'media_caption', 'comment_text']

Starting comprehensive analysis...
Loaded 102,806 comments for analysis
Performing sentiment and theme analysis...
Generating metrics-driven trend analysis...
Creating trend visualizations...

======================================================================
 METRICS OVERVIEW
======================================================================
Report Generated: 2025-01-27T10:30:00
Analysis Period: 2025-03-01 to 2025-03-15
Total Comments: 102,806

KEY PERFORMANCE INDICATORS
   • Sentiment Score: 72/100
   • Engagement Score: 85/100
   • Topic Diversity: 15 unique themes

======================================================================
 TEMPORAL TRENDS ANALYSIS
======================================================================
VOLUME STATISTICS
   • Daily Average: 75.2 comments/day
   • Peak Activity: 234 comments on 2025-03-08
   • Lowest Activity: 12 comments on 2025-03-12

ENGAGEMENT PATTERNS
   • Busiest Day: Friday
   • Peak Hour: 14:00
```

## Customization

### Add Custom Topic Categories

You can modify the topic analysis in `src/socialMediaInsightsBot.py`:

```python
def _analyze_topic_trends(self, comments: List[Dict]) -> List[Dict]:
    # Add your custom topic keywords
    topic_keywords = {
        "quality": ["quality", "good", "great", "excellent", "amazing", "perfect"],
        "price": ["price", "expensive", "cheap", "cost", "money", "value"],
        "service": ["service", "support", "help", "customer", "staff"],
        "delivery": ["delivery", "shipping", "fast", "slow", "arrived"],
        "product": ["product", "item", "love", "hate", "recommend"],
        # Add your custom categories:
        "sustainability": ["eco", "green", "sustainable", "environment"],
        "packaging": ["package", "box", "wrapped", "packaging"]
    }
```

### Adjust Visualization Styles

Modify the `create_trend_visualizations` method to customize chart styles, colors, and formats.

## Requirements

The updated `requirements.txt` includes all necessary dependencies:

- **langchain**: LLM orchestration
- **pandas**: Data manipulation
- **matplotlib**: Chart creation
- **seaborn**: Statistical visualizations
- **wordcloud**: Text visualization
- **chromadb**: Vector database
- **openai**: Language model access

## Advanced Features

### Comparative Analysis
Compare sentiment and topics across different time periods or media types.

### Real-time Monitoring
Set up the analysis to run periodically and track metric changes over time.

### Custom Insights
Use the interactive query system to ask specific questions about your engagement data.

### Export Capabilities
All results are saved in JSON format for further analysis or integration with other tools.

---

This enhanced version provides comprehensive insights into how your social media engagement evolves over time, helping you identify trends, optimize content strategy, and improve customer satisfaction.