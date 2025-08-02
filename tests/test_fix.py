#!/usr/bin/env python3

import sys
sys.path.append('src')

try:
    from socialMediaInsightsBot import SocialMediaInsightsBot
    
    print("Creating SocialMediaInsightsBot instance...")
    bot = SocialMediaInsightsBot()
    
    print("Available templates:", list(bot.prompt_templates.keys()))
    print("sentiment_analysis template exists:", 'sentiment_analysis' in bot.prompt_templates)
    
    if 'sentiment_analysis' in bot.prompt_templates:
        print("SUCCESS: sentiment_analysis template is available!")
        print("Template preview:", bot.prompt_templates['sentiment_analysis'][:100] + "...")
    else:
        print("ERROR: sentiment_analysis template not found!")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()