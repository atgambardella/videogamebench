"""
Example script demonstrating how to use Qwen with WebBrowsingAgent.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm.realtime_agent import WebBrowsingAgent
from src.llm.prompts import TASK_PROMPTS

async def main():
    """Run a WebBrowsingAgent with a local Qwen model."""
    
    # Create a WebBrowsingAgent with Qwen
    agent = WebBrowsingAgent(
        model="Qwen2.5-VL-7B-Instruct",  # Use the Qwen model name
        api_key=None,                    # No API key needed for local models
        game="dos",
        task_prompt=TASK_PROMPTS["dos"],
        enable_ui=True,
        lite=True  # Use lite mode for faster interactions
    )
    
    print("Agent initialized with Qwen model.")
    
    # Start the agent with an initial URL
    await agent.start("https://dos.zone/doom/")
    
    # Define a simple task
    task = "Play DOOM. Use the arrow keys to move around and spacebar to shoot."
    
    try:
        # Run an episode
        await agent.run_episode(task, max_steps=10)
    finally:
        # Make sure to stop the agent
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
