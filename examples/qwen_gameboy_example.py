"""
Example script demonstrating how to use Qwen with GameBoyAgent.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm.realtime_agent import GameBoyAgent
from src.llm.prompts import TASK_PROMPTS
from PIL import Image
import time

async def main():
    """Run a GameBoyAgent with a local Qwen model."""
    
    # Create a GameBoyAgent with Qwen
    agent = GameBoyAgent(
        model="Qwen2.5-VL-7B-Instruct",  # Use the Qwen model name
        api_key=None,                    # No API key needed for local models
        game="pokemon_red",
        task_prompt=TASK_PROMPTS["pokemon_red"],
        realtime=True,
        enable_ui=True
    )
    
    print("Agent initialized with Qwen model.")
    
    # Create a mock observation for testing
    # In a real scenario, this would come from the emulator
    mock_observation = {
        'screen': Image.new('RGB', (160, 144), color='blue'),
        'buttons': ['A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT']
    }
    
    print("Getting action from agent...")
    
    # Get an action from the agent
    action = await agent.get_action(mock_observation)
    
    print(f"Agent returned action: {action}")
    
    # In a real scenario, you would apply this action to the emulator
    # and then get the next observation

if __name__ == "__main__":
    asyncio.run(main())
