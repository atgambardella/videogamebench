# Using Local Qwen Models with VideoGameBench

This document explains how to use local Qwen VL (Vision-Language) models with VideoGameBench.

## Prerequisites

Before using local Qwen models, ensure you have the following dependencies installed:

```bash
pip install transformers torch
```

## Using Local Qwen Models

The implementation supports using local Qwen2.5-VL models alongside API-based models. Here's how to use them:

### Example: GameBoyAgent with Qwen

```python
from src.llm.realtime_agent import GameBoyAgent

# Create an agent with a local Qwen model
agent = GameBoyAgent(
    model="Qwen2.5-VL-7B-Instruct",  # Use the Qwen model name
    api_key=None,                    # No API key needed for local models
    game="pokemon_red",
    task_prompt="Play Pokemon Red and defeat the first gym leader."
)

# The rest of your code remains the same
```

### Example: WebBrowsingAgent with Qwen

```python
from src.llm.realtime_agent import WebBrowsingAgent

# Create an agent with a local Qwen model
agent = WebBrowsingAgent(
    model="Qwen2.5-VL-7B-Instruct",  # Use the Qwen model name
    api_key=None,                    # No API key needed for local models
    game="dos",
    task_prompt="Navigate to the specified directory and run the game."
)

# The rest of your code remains the same
```

## Supported Qwen Models

Currently, the implementation supports the following Qwen models:

- `Qwen2.5-VL-7B-Instruct`
- `Qwen2.5-VL-72B-Instruct`

You can also specify the full HuggingFace path, e.g., `Qwen/Qwen2.5-VL-7B-Instruct`.

## Technical Details

When using a Qwen model:

1. The system loads the model and processor on first use to avoid unnecessary memory usage
2. Images are automatically processed for the Qwen VL model
3. The implementation handles the message formatting differences between API-based models and local models

## Notes

- Local Qwen models use GPU memory and require a system with sufficient VRAM
- For maximum performance with large models, ensure your system has sufficient GPU memory
- If you encounter memory issues, you can try reducing the model size or using a model with flash attention
