#!/usr/bin/env python3
"""
Simple test for JSON cleaning functionality
"""

import json
import re

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

def test_json_cleaning():
    """Test the JSON cleaning function"""
    
    # Test cases with various markdown formatting issues
    test_cases = [
        # Case 1: JSON wrapped in markdown code blocks
        ('```json\n{"test": "value"}\n```', {"test": "value"}),
        
        # Case 2: JSON with extra whitespace
        ('  \n```json\n{"test": "value"}\n```  \n', {"test": "value"}),
        
        # Case 3: Plain JSON (should remain unchanged)
        ('{"test": "value"}', {"test": "value"}),
        
        # Case 4: JSON with newlines like your issue
        ('```json\n{\n    "sentiment_summary": {\n        "positive_count": 5,\n        "negative_count": 2,\n        "neutral_count": 3,\n        "overall_sentiment": "positive"\n    }\n}\n```',
         {"sentiment_summary": {"positive_count": 5, "negative_count": 2, "neutral_count": 3, "overall_sentiment": "positive"}}),
        
        # Case 5: Your specific issue format
        ('```json\n{\n    "sentiment_summary": {\n        "positive_count": 10.0,\n        "negative_count": 5.0,\n        "neutral_count": 15.0,\n        "overall_sentiment": "neutral"\n    },\n    "detailed_analysis": [\n        {\n            "comment": "I love this!",\n            "sentiment": "positive",\n            "confidence": 0.9,\n            "reasoning": "Expresses clear positive emotion"\n        }\n    ]\n}',
         {"sentiment_summary": {"positive_count": 10.0, "negative_count": 5.0, "neutral_count": 15.0, "overall_sentiment": "neutral"}, "detailed_analysis": [{"comment": "I love this!", "sentiment": "positive", "confidence": 0.9, "reasoning": "Expresses clear positive emotion"}]})
    ]
    
    print("Testing JSON cleaning function...")
    print("=" * 50)
    
    all_passed = True
    
    for i, (test_input, expected) in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Input: {repr(test_input[:100])}{'...' if len(test_input) > 100 else ''}")
        
        try:
            cleaned = clean_json_response(test_input)
            parsed = json.loads(cleaned)
            
            # Verify it's valid JSON
            print("✅ JSON is valid")
            
            # Check if the structure matches expected
            if parsed == expected:
                print("✅ Output matches expected structure")
            else:
                print("⚠️  Output structure differs from expected")
                print(f"   Expected: {expected}")
                print(f"   Got: {parsed}")
                all_passed = False
                
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON - {e}")
            print(f"   Cleaned output: {repr(cleaned)}")
            all_passed = False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! JSON cleaning works correctly.")
        print("\nThe fixes should resolve your issue where the LLM was returning:")
        print("```json\\n{\\n    \"sentiment_summary\": { ... ")
        print("\nInstead of clean JSON like:")
        print("{ \"sentiment_summary\": { ... ")
    else:
        print("⚠️ Some tests failed.")
    print("=" * 50)

if __name__ == "__main__":
    test_json_cleaning()