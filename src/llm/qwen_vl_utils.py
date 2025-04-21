"""
Utility functions for Qwen VL models.
Adapted from https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
"""
import torch
from typing import Dict, List, Any, Optional


def process_vision_info(messages: List[Dict[str, Any]]) -> tuple:
    """
    Process vision information from messages for Qwen VL models.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        Tuple of (image_inputs, video_inputs) ready for the processor
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        content = message.get("content", [])
        
        # Handle both string content and list content formats
        if isinstance(content, str):
            continue
        
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "image":
                    # Direct image path or URL
                    image_inputs.append(item.get("image"))
                elif item.get("type") == "image_url":
                    # Handle image_url format used by API models
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image/"):
                        # Handle base64 encoded images
                        import base64
                        import io
                        from PIL import Image
                        # Extract the base64 part
                        base64_data = image_url.split(",")[1]
                        image_data = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_data))
                        image_inputs.append(image)
                    else:
                        # Regular URL
                        image_inputs.append(image_url)
                elif item.get("type") == "video":
                    # Video path or URL
                    video_inputs.append(item.get("video"))
    
    return image_inputs, video_inputs


def format_qwen_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format messages to be compatible with Qwen's expected format.
    This converts OpenAI/Anthropic format to Qwen format.
    
    Args:
        messages: List of message dictionaries in OpenAI/Anthropic format
        
    Returns:
        List of messages in Qwen format
    """
    qwen_messages = []
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            # System messages should be prepended to the first user message
            continue
        
        # Map roles to Qwen's expected format
        qwen_role = "user" if role == "user" else "assistant"
        
        # Handle multimodal content
        if isinstance(content, list):
            qwen_content = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    
                    if item_type == "text":
                        qwen_content.append({
                            "type": "text",
                            "text": item.get("text", "")
                        })
                    elif item_type == "image_url":
                        # Convert to Qwen's image format
                        image_url = item.get("image_url", {}).get("url", "")
                        qwen_content.append({
                            "type": "image",
                            "image": image_url
                        })
                    elif item_type == "image":
                        qwen_content.append(item)  # Already in correct format
                    elif item_type == "video":
                        qwen_content.append(item)  # Already in correct format
                else:
                    # Plain text content
                    qwen_content.append({
                        "type": "text",
                        "text": str(item)
                    })
        else:
            # Simple text content
            qwen_content = [{"type": "text", "text": content}]
        
        qwen_messages.append({
            "role": qwen_role,
            "content": qwen_content
        })
    
    return qwen_messages
