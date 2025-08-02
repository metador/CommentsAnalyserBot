#!/usr/bin/env python3
"""
Test script to demonstrate comment filtering functionality
"""

import sys
import os
sys.path.append('./src')

from utils import (
    is_tag_only_comment, 
    is_name_only_comment, 
    is_numbers_only_comment, 
    should_filter_comment,
    filter_comments
)

def test_filtering_functions():
    """Test all filtering functions with sample data"""
    
    print("="*70)
    print(" COMMENT FILTERING DEMONSTRATION")
    print("="*70)
    
    # Test data
    test_comments = [
        # Should be filtered out
        "@kalwone",
        "@user123",
        "@john.doe",
        "John Smith",
        "Mary Johnson",
        "Dr. Brown",
        "Jane",
        "123",
        "456.78",
        "$50.99",
        "100%",
        "42",
        "",
        "  ",
        
        # Should NOT be filtered out
        "This is a great product!",
        "I love this so much",
        "The quality is amazing",
        "Great customer service from John",
        "Called 123 times but finally got help",
        "My name is John and I recommend this",
        "Product arrived in 2 days, excellent!",
        "5 stars! Would buy again",
        "Thanks @support for the help",
        "Contact us at 555-1234 for more info"
    ]
    
    print("\n1. TESTING TAG DETECTION:")
    print("-" * 40)
    for comment in test_comments[:4]:
        result = is_tag_only_comment(comment)
        print(f"'{comment}' -> Tag only: {result}")
    
    print("\n2. TESTING NAME DETECTION:")
    print("-" * 40)
    for comment in test_comments[4:8]:
        result = is_name_only_comment(comment)
        print(f"'{comment}' -> Name only: {result}")
    
    print("\n3. TESTING NUMBERS DETECTION:")
    print("-" * 40)
    for comment in test_comments[8:14]:
        result = is_numbers_only_comment(comment)
        print(f"'{comment}' -> Numbers only: {result}")
    
    print("\n4. OVERALL FILTERING RESULTS:")
    print("-" * 40)
    
    # Convert to comment dictionaries
    comment_dicts = [{'id': i, 'text': comment, 'metadata': {}} 
                    for i, comment in enumerate(test_comments)]
    
    print(f"Original comments: {len(comment_dicts)}")
    
    # Show what would be filtered
    filtered_out = []
    kept = []
    
    for comment_dict in comment_dicts:
        if should_filter_comment(comment_dict['text']):
            filtered_out.append(comment_dict['text'])
        else:
            kept.append(comment_dict['text'])
    
    print(f"\nWOULD BE FILTERED OUT ({len(filtered_out)} comments):")
    for comment in filtered_out:
        print(f"  ❌ '{comment}'")
    
    print(f"\nWOULD BE KEPT ({len(kept)} comments):")
    for comment in kept:
        print(f"  ✅ '{comment}'")
    
    # Test the main filtering function
    print(f"\n5. TESTING MAIN FILTER FUNCTION:")
    print("-" * 40)
    filtered_comments = filter_comments(comment_dicts)
    
    print(f"\nFinal Results:")
    print(f"  • Original: {len(comment_dicts)} comments")
    print(f"  • Filtered out: {len(comment_dicts) - len(filtered_comments)} comments")
    print(f"  • Remaining: {len(filtered_comments)} comments")
    
    return filtered_comments

def test_with_real_data():
    """Test with some realistic social media comments"""
    print("\n" + "="*70)
    print(" REALISTIC DATA TEST")
    print("="*70)
    
    realistic_comments = [
        "I absolutely love this product! Best purchase ever!",
        "@sarah",  # Should be filtered
        "Amazing quality and fast shipping",
        "John",   # Should be filtered
        "Customer service was very helpful",
        "123",    # Should be filtered
        "The color is exactly what I expected",
        "@customer_support thanks for the quick response",
        "Mary Jane",  # Should be filtered
        "Will definitely order again!",
        "99.99",  # Should be filtered
        "Great value for money",
        "@", # Should be filtered (malformed but close to tag)
        "Five stars! Highly recommended",
        "Contact me at john@email.com",  # Should NOT be filtered (has more context)
        "456",    # Should be filtered
        "The delivery was super fast"
    ]
    
    comment_dicts = [{'id': i, 'text': comment, 'metadata': {}} 
                    for i, comment in enumerate(realistic_comments)]
    
    print(f"Testing with {len(realistic_comments)} realistic comments...")
    
    filtered_comments = filter_comments(comment_dicts)
    
    print(f"\nRemaining comments after filtering:")
    for i, comment in enumerate(filtered_comments, 1):
        print(f"  {i}. {comment['text']}")

if __name__ == "__main__":
    test_filtering_functions()
    test_with_real_data()
    
    print("\n" + "="*70)
    print(" FILTERING TEST COMPLETE")
    print("="*70)
    print("\nThe filtering functions are now integrated into the analysis pipeline.")
    print("When you run analyze_trends.py, unwanted comments will be automatically filtered out.")
    print("\nTo disable filtering, set apply_filtering=False in the analysis function call.")