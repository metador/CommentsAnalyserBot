import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import List, Dict





def create_trend_visualizations(comments: List[Dict],topics: List[str], output_dir: str = "./reports"):
    """
    Create visualizations for trend analysis
    
    Args:
        comments: List of comment dictionaries
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamped_comments = [c for c in comments if c.get('timestamp')]
    
    if not timestamped_comments:
        print("No timestamped comments available for visualization")
        return
    
    df = pd.DataFrame(timestamped_comments)
    df['date'] = df['timestamp'].dt.date
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Daily comment volume
    daily_counts = df.groupby('date').size()
    axes[0, 0].plot(daily_counts.index, daily_counts.values, linewidth=2)
    axes[0, 0].set_title('Daily Comment Volume', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Number of Comments')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Comments by day of week
    df['day_of_week'] = df['timestamp'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().reindex(day_order)
    axes[0, 1].bar(day_counts.index, day_counts.values, color='skyblue')
    axes[0, 1].set_title('Comments by Day of Week', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Number of Comments')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Comments by hour of day
    hour_counts = df['timestamp'].dt.hour.value_counts().sort_index()
    axes[1, 0].plot(hour_counts.index, hour_counts.values, marker='o', linewidth=2, markersize=4)
    axes[1, 0].set_title('Comments by Hour of Day', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Number of Comments')
    axes[1, 0].set_xticks(range(0, 24, 2))
    
    # 4. Comment length distribution
    comment_lengths = df['text'].str.len()
    axes[1, 1].hist(comment_lengths, bins=30, color='lightcoral', alpha=0.7)
    axes[1, 1].set_title('Comment Length Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Comment Length (characters)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/engagement_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create word cloud for overall content
    all_text = ' '.join(df['text'].astype(str))
    if all_text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Comments', fontsize=16, fontweight='bold')
        plt.savefig(f"{output_dir}/comment_wordcloud.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create topic trends over time chart
    create_topic_trends_chart(comments, topics, output_dir)
    
    print(f"Visualizations saved to {output_dir}/")

def create_topic_trends_chart(comments: List[Dict], topics: List[Dict], output_dir: str = "./reports"):
    """
    Create a line chart showing topic trends over time with trending topics highlighted
    
    Args:
        comments: List of comment dictionaries
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get topic trends data
    topics_over_time = topics
    
    if not topics_over_time:
        print("No topic trend data available for visualization")
        return
    
    # Prepare data for plotting
    periods = [data['period'] for data in topics_over_time]
    topic_keywords = ["quality", "price", "service", "delivery", "product"]
    
    # Extract topic counts for each period
    topic_data = {topic: [] for topic in topic_keywords}
    for period_data in topics_over_time:
        for topic in topic_keywords:
            topic_data[topic].append(period_data['topics'].get(topic, 0))
    
    # Calculate trend scores (percentage change from first to last period)
    trend_scores = {}
    for topic in topic_keywords:
        values = topic_data[topic]
        if len(values) >= 2 and values[0] > 0:
            trend_scores[topic] = ((values[-1] - values[0]) / values[0]) * 100
        else:
            trend_scores[topic] = 0
    
    # Identify trending topics (top 2 by growth rate)
    trending_topics = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    trending_topic_names = [topic for topic, _ in trending_topics]
    
    # Create the plot
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot lines for each topic
    for i, topic in enumerate(topic_keywords):
        linewidth = 3 if topic in trending_topic_names else 2
        alpha = 1.0 if topic in trending_topic_names else 0.7
        marker = 'o' if topic in trending_topic_names else None
        markersize = 6 if topic in trending_topic_names else 4
        
        ax.plot(periods, topic_data[topic], 
                color=colors[i], 
                linewidth=linewidth, 
                alpha=alpha,
                marker=marker,
                markersize=markersize,
                label=f"{topic.capitalize()}{' (Trending ↗)' if topic in trending_topic_names else ''}")
    
    # Highlight trending topics with annotations
    for topic in trending_topic_names:
        values = topic_data[topic]
        if values:
            # Annotate the highest point
            max_idx = values.index(max(values))
            max_value = max(values)
            ax.annotate(f'Peak: {max_value}', 
                        xy=(periods[max_idx], max_value),
                        xytext=(10, 10), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Customize the plot
    ax.set_title('Topic Trends Over Time\n(Highlighted: Top Trending Topics)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time Period (Weeks)', fontsize=12)
    ax.set_ylabel('Number of Mentions', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add trend summary text box
    trend_summary = f"Top Trending Topics:\n"
    for topic, growth in trending_topics[:2]:
        if growth > 0:
            trend_summary += f"• {topic.capitalize()}: +{growth:.1f}% growth\n"
        elif growth < 0:
            trend_summary += f"• {topic.capitalize()}: {growth:.1f}% decline\n"
        else:
            trend_summary += f"• {topic.capitalize()}: No change\n"
    
    ax.text(0.02, 0.98, trend_summary, 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/topic_trends_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Topic trends chart saved to {output_dir}/topic_trends_over_time.png")
