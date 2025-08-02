# Social Media Insights Bot 🤖📊

A comprehensive tool that uses Large Language Models (LLMs) to analyze up to 10,000 social media comments and extract meaningful insights including sentiment analysis, theme extraction, and interactive querying capabilities.

## 🚀 Quick Start

```python
from src.socialMediaInsightsBot import analyze_social_media_comments

# Analyze comments from a CSV file
results, bot = analyze_social_media_comments(
    data_source='path/to/comments.csv',
    source_type='csv',
    create_vectorstore=True,
    save_path='analysis_results.json'
)

# Print summary
print(f"Overall Sentiment: {results['sentiment_analysis']['overall_sentiment']}")
print(f"Top Themes: {results['theme_analysis']['top_themes'][:3]}")
```

## 📋 Features

### 🎯 Core Analysis Capabilities

- **Sentiment Analysis**: Classifies comments as positive, negative, or neutral with confidence scores
- **Theme Extraction**: Identifies recurring topics, concerns, and discussion patterns
- **Trend Analysis**: Detects trending topics and emerging themes
- **Statistical Insights**: Provides comprehensive metrics and distributions

### 🔧 Technical Features

- **Batch Processing**: Efficiently handles up to 10,000 comments
- **Multiple Data Formats**: Supports CSV, JSON, and Python lists
- **RAG Integration**: Creates vectorstore for context-aware querying
- **Executive Reporting**: Generates professional summary reports
- **Persistent Storage**: Saves results and vectorstores for reuse

### 🤖 Interactive Querying

- Ask specific questions about the comments
- Get AI-powered insights based on the actual data
- Use natural language queries
- Retrieval Augmented Generation (RAG) for accurate responses

## 📦 Installation

### Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key** (set in `.env` file)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages include:
- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-huggingface`
- `chromadb`
- `pandas`
- `python-dotenv`
- `tqdm`
- `pyyaml`

### Environment Setup

Create a `.env` file in your project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 📚 Usage Examples

### 1. Analyze CSV Data

```python
from src.socialMediaInsightsBot import analyze_social_media_comments

# Your CSV should have a column with comments (text, comment, content, etc.)
results, bot = analyze_social_media_comments(
    data_source='social_comments.csv',
    source_type='csv'
)

# Access results
sentiment = results['sentiment_analysis']
themes = results['theme_analysis']
summary = results['executive_summary']
```

### 2. Analyze JSON Data

```python
# JSON format: [{"text": "comment1"}, {"text": "comment2"}, ...]
# or simple list: ["comment1", "comment2", ...]
results, bot = analyze_social_media_comments(
    data_source='comments.json',
    source_type='json'
)
```

### 3. Analyze Python List

```python
comments = [
    "Love this product! Amazing quality.",
    "Terrible experience, would not recommend.",
    "Good value for money, satisfied overall.",
    # ... more comments
]

results, bot = analyze_social_media_comments(
    data_source=comments,
    source_type='list'
)
```

### 4. Interactive Querying

```python
# After analysis, use the bot for specific queries
bot.query_insights("What are the main complaints?")
bot.query_insights("What features do users love most?")
bot.query_insights("Are there technical issues mentioned?")
```

### 5. Custom Analysis

```python
from src.socialMediaInsightsBot import SocialMediaInsightsBot

# Initialize with custom settings
bot = SocialMediaInsightsBot(
    model_name='gpt-4o-mini',
    temperature=0.1
)

# Load comments
comments = bot.load_comments('data.csv', 'csv')

# Create vectorstore for RAG
bot.create_vectorstore(comments)

# Run analysis
results = bot.analyze_comments(comments)

# Generate executive summary
summary = bot.generate_executive_summary(results)
```

## 📊 Output Structure

The analysis returns a comprehensive dictionary with the following structure:

```python
{
    "analysis_metadata": {
        "total_comments_analyzed": int,
        "analysis_date": "ISO datetime",
        "batch_size": int,
        "model_used": "model_name"
    },
    "sentiment_analysis": {
        "overall_sentiment": "positive|negative|neutral",
        "sentiment_distribution": {
            "positive": int,
            "negative": int, 
            "neutral": int
        },
        "sentiment_percentages": {
            "positive": float,
            "negative": float,
            "neutral": float
        }
    },
    "theme_analysis": {
        "top_themes": [
            {
                "theme": "theme_name",
                "frequency": int
            }
        ],
        "trending_topics": [
            {
                "topic": "topic_name",
                "mentions": int
            }
        ],
        "unique_insights": ["insight1", "insight2", ...]
    },
    "executive_summary": "AI-generated executive summary text"
}
```

## 🎮 Demo & Examples

### Run the Demo Script

```bash
python demo_social_insights.py
```

This will:
1. Analyze sample CSV data
2. Process a list of sample comments
3. Demonstrate interactive querying
4. Save results to JSON files

### Jupyter Notebook

Open `notebooks/comments.ipynb` for an interactive demonstration with:
- Step-by-step analysis walkthrough
- Visual results display
- Interactive query examples

## ⚙️ Configuration

### Model Settings

```python
# Use different models
bot = SocialMediaInsightsBot(
    model_name='gpt-4',  # or 'gpt-3.5-turbo'
    temperature=0.0      # for more deterministic results
)
```

### Batch Size

```python
# Adjust batch size based on your needs and API limits
bot.batch_size = 25  # smaller batches for API rate limits
bot.batch_size = 100 # larger batches for faster processing
```

### Custom Prompts

The system uses prompts from `prompts/chatbot.yaml`. You can modify these for domain-specific analysis:

```yaml
sentiment_analysis: |
  You are an expert social media analyst...
  [custom prompt for your domain]

theme_extraction: |
  You are an expert in content analysis...
  [custom prompt for your use case]
```

## 📁 File Structure

```
DocumentChatBot/
├── src/
│   ├── socialMediaInsightsBot.py    # Main analysis module
│   └── __init__.py
├── sample_data/
│   └── social_media_comments.csv    # Sample data
├── notebooks/
│   └── comments.ipynb               # Interactive demo
├── prompts/
│   └── chatbot.yaml                 # Prompt templates
├── demo_social_insights.py          # Demo script
├── requirements.txt                 # Dependencies
└── README_Social_Media_Insights.md  # This file
```

## 🔍 Data Format Requirements

### CSV Format
Your CSV should contain at least one column with comment text. Supported column names:
- `text`
- `comment`
- `content`
- `message`
- `post`

Example:
```csv
comment_id,text,timestamp,platform
1,"Love this product!",2024-01-15,twitter
2,"Needs improvement",2024-01-15,facebook
```

### JSON Format
Two supported formats:

1. **Array of objects:**
```json
[
    {"text": "Great product!", "user": "user1"},
    {"text": "Could be better", "user": "user2"}
]
```

2. **Simple array:**
```json
["Great product!", "Could be better", "Amazing quality!"]
```

## ⚡ Performance & Limitations

### Processing Capacity
- **Maximum comments**: 10,000 per analysis
- **Batch processing**: Comments processed in configurable batches (default: 50)
- **Memory efficient**: Uses streaming and batching to handle large datasets

### API Considerations
- **Rate limits**: Automatically handles OpenAI API rate limits
- **Cost optimization**: Uses efficient prompting and batching
- **Error handling**: Graceful fallbacks for API failures

### Processing Time
- **~50 comments**: 1-2 minutes
- **~500 comments**: 10-15 minutes  
- **~1000 comments**: 20-30 minutes
- **~10K comments**: 3-4 hours

*Times vary based on API response times and batch sizes*

## 🛠️ Troubleshooting

### Common Issues

1. **"No comments loaded"**
   - Check file path
   - Verify column names in CSV
   - Ensure proper JSON format

2. **"OpenAI API Error"**
   - Verify API key in `.env` file
   - Check API rate limits
   - Ensure sufficient API credits

3. **"Import Error"**
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+)

4. **"Memory Error"**
   - Reduce batch size: `bot.batch_size = 25`
   - Process fewer comments at once
   - Close other applications

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with debug information
results, bot = analyze_social_media_comments(
    data_source=your_data,
    source_type='auto'
)
```

## 📈 Advanced Usage

### Custom Analysis Pipeline

```python
from src.socialMediaInsightsBot import SocialMediaInsightsBot

class CustomInsightsBot(SocialMediaInsightsBot):
    def custom_analysis(self, comments):
        # Add your custom analysis logic
        pass
    
    def domain_specific_themes(self, comments):
        # Industry-specific theme extraction
        pass

# Use your custom bot
custom_bot = CustomInsightsBot()
```

### Parallel Processing

For very large datasets, consider parallel processing:

```python
# Split comments into chunks
chunk_size = 1000
comment_chunks = [comments[i:i+chunk_size] for i in range(0, len(comments), chunk_size)]

# Process chunks separately and combine results
all_results = []
for chunk in comment_chunks:
    results, _ = analyze_social_media_comments(chunk, 'list')
    all_results.append(results)

# Combine results (implement your aggregation logic)
```

## 📞 Support

For issues, feature requests, or questions:

1. Check the troubleshooting section above
2. Review the demo script and notebook examples
3. Ensure your data format matches the requirements
4. Verify API key and dependencies are properly set up

## 📄 License

This project is part of the DocumentChatBot package. See the main project license for details.