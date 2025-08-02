import json
import re
import numpy as np
import pandas as pd

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj
def clean_json_response(response_content: str) -> str:
    """
    Clean LLM response to extract valid JSON
    
    Args:
        response_content: Raw response from LLM
        
    Returns:
        Clean JSON string
    """
    # Remove markdown code block formatting
    cleaned = re.sub(r'^```json\s*\n?', '', response_content.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
    
    # Remove any leading/trailing whitespace
    cleaned = cleaned.strip()
    
    # If response starts with ```json, remove it
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:].strip()
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3].strip()
        
    return cleaned


def is_tag_only_comment(text: str) -> bool:
    """
    Check if a comment is just a tag of a person (e.g., @username)
    
    Args:
        text: Comment text to check
        
    Returns:
        True if comment is just a tag, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Clean the text (remove extra spaces, newlines)
    cleaned_text = text.strip()
    
    # Pattern for @mentions: @username (allowing letters, numbers, underscores, dots)
    # Must be the entire comment (start to end)
    tag_pattern = r'^@[\w.]+$'
    
    return bool(re.match(tag_pattern, cleaned_text, re.IGNORECASE))


def is_name_only_comment(text: str) -> bool:
    """
    Check if a comment contains only people's names
    
    Args:
        text: Comment text to check
        
    Returns:
        True if comment appears to be just names, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Clean the text
    cleaned_text = text.strip()
    
    # If text is very short (1-2 characters), likely not a meaningful name
    if len(cleaned_text) <= 2:
        return True
    
    # Split into words and check if all words look like names
    words = cleaned_text.split()
    
    # If no words, return True (empty after cleaning)
    if not words:
        return True
    
    # Pattern for typical names (starts with capital letter, followed by lowercase letters)
    # Allow for common name patterns including hyphenated names and initials
    name_pattern = r'^[A-Z][a-z]*(-[A-Z][a-z]*)*$|^[A-Z]\.$'
    
    # Check if all words match name patterns
    for word in words:
        # Remove common punctuation that might be at the end
        clean_word = word.rstrip('.,!?;:')
        if not re.match(name_pattern, clean_word):
            return False
    
    # Additional check: if it's just 1-2 common title words, filter it out
    common_titles = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'madam'}
    if len(words) <= 2 and all(word.lower().rstrip('.,!?;:') in common_titles for word in words):
        return True
    
    # If we have 1-3 words that all look like names, it's probably just names
    if len(words) <= 3:
        return True
    
    # For longer sequences, be more conservative (might be actual content)
    return False


def is_numbers_only_comment(text: str) -> bool:
    """
    Check if a comment contains only numbers (and basic punctuation/spaces)
    
    Args:
        text: Comment text to check
        
    Returns:
        True if comment is just numbers, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Clean the text
    cleaned_text = text.strip()
    
    # Remove all numbers, spaces, and common number-related punctuation
    # If nothing is left, it was numbers only
    numbers_and_punct_pattern = r'[0-9\s.,+\-()$€£¥%/:]'
    remaining_text = re.sub(numbers_and_punct_pattern, '', cleaned_text)
    
    # If nothing remains after removing numbers and punctuation, it was numbers only
    return len(remaining_text) == 0 and len(cleaned_text) > 0


def should_filter_comment(text: str) -> bool:
    """
    Main function to determine if a comment should be filtered out
    
    Args:
        text: Comment text to check
        
    Returns:
        True if comment should be filtered out, False if it should be kept
    """
    if not text or not isinstance(text, str):
        return True
    
    # Check all filtering conditions
    return (is_tag_only_comment(text) or 
            is_name_only_comment(text) or 
            is_numbers_only_comment(text))


def filter_comments(comments: list) -> list:
    """
    Filter out unwanted comments from a list of comment dictionaries
    
    Args:
        comments: List of comment dictionaries with 'text' field
        
    Returns:
        Filtered list of comments
    """
    filtered_comments = []
    filtered_count = 0
    
    for comment in comments:
        # Handle both string comments and dictionary comments
        if isinstance(comment, str):
            comment_text = comment
        elif isinstance(comment, dict):
            comment_text = comment.get('text', '')
        else:
            continue
        
        # If comment should not be filtered, keep it
        if not should_filter_comment(comment_text):
            filtered_comments.append(comment)
        else:
            filtered_count += 1
    
    print(f"Filtered out {filtered_count} comments (tags, names only, numbers only)")
    print(f"Remaining comments: {len(filtered_comments)}")
    
    return filtered_comments