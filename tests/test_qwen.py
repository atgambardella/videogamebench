"""
Test script for Qwen VL models in VideoGameBench.
"""
import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm.llm_client import LLMClient
from PIL import Image
import base64
from io import BytesIO

async def test_qwen():
    """Test the Qwen VL model integration."""
    
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='red')
    buffered = BytesIO()
    test_image.save(buffered, format="PNG")
    image_data = buffered.getvalue()
    
    # Initialize the LLM client with Qwen
    client = LLMClient(
        model="Qwen2.5-VL-7B-Instruct",
        api_key=None,  # No API key needed for local models
        temperature=0.7,
        max_tokens=1024
    )
    
    # Create a system message
    system_message = {
        "role": "system",
        "content": "You are a helpful vision-language assistant. Please describe the image and respond to the user's questions."
    }
    
    # Create messages with an image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                    }
                }
            ]
        }
    ]
    
    print("Testing Qwen VL model...")
    
    # Generate a response
    try:
        response = await client.generate_response(
            system_message=system_message,
            messages=messages
        )
        
        print("\nResponse:")
        print(response)
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Qwen VL model integration")
    args = parser.parse_args()
    
    asyncio.run(test_qwen())
