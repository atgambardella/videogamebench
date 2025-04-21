import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image
from io import BytesIO
import re
import threading
from src.utils import is_same_image

from src.emulators.dos.browser_controller import BrowserController
from src.llm.llm_client import LLMClient
from src.llm.prompts import SYSTEM_PROMPTS, TASK_PROMPTS, GBA_PROMPT, REFLECTION_PROMPT, GBA_REALTIME_PROMPT
from src.llm.utils import parse_actions_response, convert_to_dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Message:
    """Container for a message in the conversation history."""
    
    def __init__(self, role: str, content: Any, has_image: bool = False, tokens: int = 0):
        self.role = role
        self.content = content
        self.has_image = has_image
        self.tokens = tokens  # Approximate token count
        self.timestamp = datetime.now()
        
    def __str__(self):
        if isinstance(self.content, str):
            preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
            return f"{self.role}: {preview}"
        else:
            return f"{self.role}: [Complex content with image={self.has_image}]"

class VideoGameBenchAgent:
    """
    Base class for all VG agents.
    """
    def __init__(self, 
                 model: str, 
                 api_key: Optional[str] = None, 
                 game: str = "", 
                 task_prompt: str = "",
                 headless: bool = False, 
                 temperature: float = 0.7, 
                 max_tokens: int = 1024, 
                 max_history_tokens: int = 4000, 
                 context_window: int = 10, 
                 log_dir: Optional[Path] = None,
                 enable_ui: bool = False,
                 record: bool = False,
                 num_screenshots_per_action: int = 3,
                 api_base: Optional[str] = None):
        # Set up logging directory
        if log_dir is None:
            model_name = model.replace("/", "-").replace(".", "-")
            self.log_dir = Path("logs") / f"{game.lower()}" / model_name / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = log_dir

        # Set up file logger
        self.file_logger = self._setup_file_logger()
        self.file_logger.info(f"Initializing {self.__class__.__name__} with model: {model}")

        # Initialize LLM client
        self.llm_client = LLMClient(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            log_dir=self.log_dir / "llm",
            api_base=api_base
        )

        # Common attributes
        self.max_history_tokens = max_history_tokens
        self.context_window = context_window
        self.headless = headless # TODO: make this do something
        self.game = game
        self.task_prompt = task_prompt
        self.system_instruction_prompt = SYSTEM_PROMPTS.get(game, "")
        self.system_prompt = {
            "role": "system",
            "content": f"{self.system_instruction_prompt}\n\n{self.task_prompt}"
        }
        self.num_screenshots_per_action = num_screenshots_per_action # TODO: make this work for GB
        
        self.reflection_memory = ""
        self.step_count = 0
        
        # Create consolidated log files
        self.reflection_log_file = self.log_dir / "reflections.txt"
        self.reflection_log_file.touch()

        logger.info(f"{self.__class__.__name__} initialized. Logging to: {self.log_dir}")


        # Initialize UI if enabled
        self.ui = None

        # TODO: Add better screen recording
        if record:
            pass

        if enable_ui:
            from src.ui.agent_monitor import AgentMonitorUI
            self.ui = AgentMonitorUI(f"{model} agent playing {self.game} on VideoGameBench")

    def _setup_file_logger(self) -> logging.Logger:
        """Set up a file logger for this session."""
        file_logger = logging.getLogger(f"{self.__class__.__name__.lower()}_{id(self)}")
        file_logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / "agent_session.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add handler to logger
        file_logger.addHandler(file_handler)
        
        return file_logger

    def save_reflection(self) -> None:
        """Save reflection to consolidated file."""
        if self.reflection_memory:
            with open(self.reflection_log_file, "a") as f:
                f.write(f"\n\n-=-=-=-=-=-reflection_{self.step_count}-=-=-=-=-=\n\n")
                f.write(self.reflection_memory)
            self.file_logger.info(f"Updated reflection memory saved to {self.reflection_log_file}")

        if self.ui:
            print("Updating reflection memory in UI...")
            self.ui.update_reflection_memory(self.reflection_memory)
    
    def update_steps_count(self, count: Optional[int] = None) -> None:
        """Update the steps counter display."""
        if self.ui:
            self.ui.update_steps_count(count)

class GameBoyAgent(VideoGameBenchAgent):
    """
    VideoGameBench Agent that uses the ReACT method to play a Game Boy game.
    Unlike the web browsing agent, this agent does not hold the browser / game object.
    """
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        game: str = "",
        task_prompt: str = "",
        headless: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_history_tokens: int = 4000,
        context_window: int = 10,
        log_dir: Optional[Path] = None,
        realtime: bool = False,
        enable_ui: bool = False,
        record: bool = False,
        num_screenshots_per_action: int = 3,
        api_base: Optional[str] = None
    ):
        """
        Initialize the GBA agent.
        
        Args:
            model: The LLM model to use
            api_key: The API key for the model provider
            game_type: The type of game being played
            headless: Whether to run the browser in headless mode
            temperature: The temperature for LLM generation
            max_tokens: The maximum number of tokens to generate
            max_history_tokens: The maximum number of tokens to keep in history
            log_dir: Optional custom log directory path
        """
        super().__init__(
            model=model,
            api_key=api_key,
            game=game,
            headless=headless,
            temperature=temperature,
            max_tokens=max_tokens,
            max_history_tokens=max_history_tokens,
            context_window=context_window,
            log_dir=log_dir,
            enable_ui=enable_ui,
            record=record,
            task_prompt=task_prompt,
            num_screenshots_per_action=num_screenshots_per_action,
            api_base=api_base
        )
        
        self.realtime = realtime
        self.system_instruction_prompt = SYSTEM_PROMPTS["gba_realtime"] if realtime else SYSTEM_PROMPTS["gba"]
        self.system_prompt["content"] = f"{self.system_instruction_prompt}\n\n{self.task_prompt}"
        self.gba_prompt = GBA_REALTIME_PROMPT if realtime else GBA_PROMPT
        
        self.context_history: List[Message] = []
        self.full_history: List[Dict[str, Any]] = []
        self.image_dir = None
        self.context_window = context_window

        self.action = ""
        self.prev_action = ""

        logger.info(f"{self.__class__.__name__} initialized. Logging to: {self.log_dir}")

    def add_to_history(self, role: str, content: Any, has_image: bool = False, tokens: int = 0) -> None:
        """Add a message to both full history and context history."""
        message = Message(role, content, has_image, tokens)
        
        # Add to context history (with images)
        self.context_history.append(message)
        if len(self.context_history) > self.context_window * 2:  # *2 because each step has user+assistant message
            self.context_history = self.context_history[-self.context_window * 2:]
        
        # Add to full history (without images)
        if isinstance(content, list):
            # Extract text content from multimodal messages
            text_content = ""
            for item in content:
                if item.get("type") == "text":
                    text_content += item.get("text", "") + " "
        else:
            text_content = content
            
        self.full_history.append({
            "role": role,
            "content": text_content.strip(),
            "timestamp": datetime.now(),
            "tokens": tokens
        })
        
        # Log the exchange
        if role == "user":
            log_msg = f"Step {self.step_count} - User observation"
            if has_image:
                log_msg += " with image"
            self.file_logger.info(log_msg)
        else:
            self.file_logger.info(f"Step {self.step_count} - AI action: {text_content}")
        
        self._prune_history()


    def _prune_history(self) -> None:
        """Remove oldest messages if token count exceeds max_history_tokens."""
        # Calculate total tokens
        total_tokens = sum(msg.tokens for msg in self.context_history)
        
        # Keep removing oldest messages until we're under the limit
        while total_tokens > self.max_history_tokens and len(self.context_history) > 10:
            # Always keep the most recent exchange (last 10 messages)
            removed = self.context_history.pop(0)
            total_tokens -= removed.tokens
            logger.debug(f"Pruned message from history: {removed}")
    
    def _save_image(self, image: Image.Image) -> Path:
        """Save the image to the log directory."""
        # Create image directory if it doesn't exist yet
        if self.image_dir is None:
            self.image_dir = self.log_dir / "game_screen"
            self.image_dir.mkdir(exist_ok=True)

        if self.ui:
            monitor_dir = self.image_dir / "monitor"
            monitor_dir.mkdir(exist_ok=True)
            self.ui.take_screenshot(monitor_dir, f"screenshot_{self.step_count}.jpg")
            
        image_path = self.image_dir / f"screenshot_{self.step_count}.png"
        image.save(image_path)
        return image_path
        
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def store_observation(self, observation: Dict[str, Any], 
                          prev_action: Optional[str] = None) -> None:
        """Store the observation in the history."""
        image = observation['screen']
        buttons = observation['buttons']
        
        # Save image to log directory
        image_path = self._save_image(image)
        self.file_logger.info(f"Saved observation image to {image_path}")
        
        # Prepare the image for API
        base64_image = self._encode_image(image)
        
        # Prepare user message with game state
        if prev_action:
            user_content = [
                {
                    "type": "text",
                    "text": f"You previously pressed: {prev_action}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        else:
            user_content = [
                {"type": "text", "text": "Your current screen is:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        self.add_to_history("user", user_content, has_image=True)
        
    async def get_action(self, observation: Dict[str, Any], prev_action: Optional[str] = None) -> Dict[str, bool] | List[Dict[str, bool]]:
        """
        Get the next action based on the current game observation.
        
        Args:
            observation: Dictionary containing 'screen' (PIL Image) and 'buttons' (available buttons)
            
        Returns:
            Dictionary mapping button names to boolean values (pressed/not pressed)
        """
        if self.ui:
            self.ui.update_last_action(prev_action)

        self.step_count += 1
        self.store_observation(observation, prev_action=prev_action)
        
        # Create messages with reflection prompt and current memory
        messages = [{"role": m.role, "content": m.content} for m in self.context_history]
        messages.append({
            "role": "user",
            "content": f"{REFLECTION_PROMPT}\n\n[Your current reflection memory]:\n{self.reflection_memory}"
        })
        messages.append({
            "role": "user",
            "content": f"{self.gba_prompt}"
        })

        # Get the next action from the LLM using context history
        start_time = time.time()
        response = await self.llm_client.generate_response(
            system_message=self.system_prompt,
            messages=messages
        )
        response_time = time.time() - start_time
        
        # Log the response
        self.file_logger.info(f"Response time: {response_time:.2f}s")
        # self.file_logger.info(f"LLM response: {response}")
        # logger.info(f"LLM response for step {self.step_count}")
        # logger.info(response)
        
        # Parse reflection from response
        reflection_match = re.search(r'```reflection\s*(.*?)\s*```', response, re.DOTALL)
        if reflection_match:
            self.reflection_memory = reflection_match.group(1).strip()
            self.save_reflection()
        
        # Add response to history
        # Add response to history, excluding reflection content
        response_without_reflection = re.sub(r'```reflection\s*.*?\s*```', '', response, flags=re.DOTALL).strip()
        self.add_to_history("assistant", "[Your thought]: " + response_without_reflection)

        if self.ui:
            self.ui.update_context_history(self.context_history)
            self.ui.update_steps_count(self.step_count)

        # Parse the response to get button states
        self.prev_action = self.action
        try:
            
            actions = parse_actions_response(response)
            button_states = convert_to_dict(actions)
            self.action = actions

            if self.ui:
                self.ui.update_executing_action(actions)

            return button_states
        except Exception as e:
            self.file_logger.error(f"Error parsing response: {e}")
            return None


class WebBrowsingAgent(VideoGameBenchAgent):
    """
    Agent that uses the ReACT method to browse the web and complete tasks.
    """
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        game: str = "",
        task_prompt: str = "",
        headless: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_history_tokens: int = 4000,
        context_window: int = 10,
        lite: bool = False,
        press_key_delay: int = 100,
        log_dir: Optional[Path] = None,
        enable_ui: bool = False,
        record: bool = False,
        num_screenshots_per_action: int = 3,
        api_base: Optional[str] = None
    ):
        """
        Initialize the web browsing agent.
        
        Args:
            model: The LLM model to use
            api_key: The API key for the model provider
            game: The name of the game to play
            headless: Whether to run the browser in headless mode
            temperature: The temperature for LLM generation
            max_tokens: The maximum number of tokens to generate
            max_history_tokens: The maximum number of tokens to keep in history
            context_window: Number of recent timesteps to include
            lite: Whether to run in lite mode with reduced functionality
            log_dir: Optional custom log directory path
            enable_ui: Whether to enable the UI monitor
            record: Whether to record the gameplay session
        """
        super().__init__(
            model=model,
            api_key=api_key,
            game=game,
            headless=headless,
            temperature=temperature,
            max_tokens=max_tokens,
            max_history_tokens=max_history_tokens,
            context_window=context_window,
            log_dir=log_dir,
            enable_ui=enable_ui,
            record=record,
            task_prompt=task_prompt,
            num_screenshots_per_action=num_screenshots_per_action,
            api_base=api_base
        )
        
        # Initialize browser controller
        self.browser = BrowserController(headless=headless)
        self.max_history_length = 16
        self.conversation_history: List[Dict[str, str]] = []
        self.task: Optional[str] = None

        # Game-specific settings
        self.press_key_delay = press_key_delay
        
        # Additional WebBrowsingAgent-specific log file
        self.conversation_log_file = self.log_dir / "conversation_history.json"
        self.conversation_log_file.touch()

        self.system_instruction_prompt = SYSTEM_PROMPTS["dos"]
        self.system_prompt = {
            "role": "system", 
            "content": f"{self.system_instruction_prompt}\n\n{task_prompt}"
        }
        self.lite = lite
        self.lite_counter = 0

    async def start(self, initial_url: str) -> None:
        """
        Start the agent by initializing the browser.
        """
        self.file_logger.info("Starting browser")
        await self.browser.start()
        

        self.file_logger.info(f"Navigating to URL: {initial_url}")
        
        # Navigate to the initial URL
        start_time = time.time()
        await self.browser.navigate(initial_url)
        load_time = time.time() - start_time
        self.file_logger.info(f"Page loaded in {load_time:.2f}s")

        # Pre-loaded actions based on game
        await self.browser.pre_load(self.game)

    async def stop(self) -> None:
        """
        Stop the agent by closing the browser.
        """
        self.file_logger.info("Stopping browser")
        await self.browser.close()
        
    async def run_episode(self, task: str, max_steps: int = 400, checkpoints: Optional[List[str]] = None) -> None:
        """
        Run an episode with the ReACT and memory agent. 
        TODO: Eventually move this into the evaluator.
        
        Args:
            task: The task to execute
            max_steps: The maximum number of steps to take
        """
        self.task = self.task_prompt + "\n\n" + task 
        self.conversation_history = []
        
        # Get initial screenshot
        screenshot = await self.browser.get_screenshot()
        screenshots = [screenshot]
        
        # Save screenshot
        screenshot_dir = self.log_dir / "game_screen"
        monitor_dir = self.log_dir / "monitor"
        screenshot_dir.mkdir(exist_ok=True)
        monitor_dir.mkdir(exist_ok=True)
        screenshot_path = screenshot_dir / f"screenshot_initial.jpg"
        with open(screenshot_path, "wb") as f:
            f.write(screenshot)
            if self.ui:
                self.ui.take_screenshot(monitor_dir, "screenshot_initial.jpg")
        self.file_logger.info(f"Saved initial screenshot to {screenshot_path}")

        if self.lite:
            await self.browser.press_key("Alt+Pause", delay_ms=0)

        for step in range(max_steps):
            self.step_count = step + 1
            self.file_logger.info(f"Step {self.step_count}/{max_steps}")
            logger.info(f"Step {self.step_count}/{max_steps}")
            
            # Generate the next action using ReACT
            print("SYSTEM PROMPT: ", self.system_prompt)
            print("Task: ", task)
            start_time = time.time()
            # Add reflection prompt to the task
            task_with_reflection = (
                f"{REFLECTION_PROMPT}\n\n[memory]:\n{self.reflection_memory}"
                f"Your mouse is currently at coordinates: {self.browser.current_mouse_position}. Move it with move or drag actions."
                f"{self.task_prompt}\n\n"
            )
            
            react_response = await self.llm_client.generate_react_response(
                task=task_with_reflection,
                system_message=self.system_prompt,
                history=self.conversation_history,
                screenshot=screenshots
            )
            response_time = time.time() - start_time
            
            # Update reflection memory from the JSON response
            memory = react_response.get("memory", "")
            if memory:  # Only update if memory exists and is not empty
                self.reflection_memory = memory
            
            # Save reflection to consolidated file
            if self.reflection_memory:
                self.save_reflection()
            
            # Log the thought process
            thought = react_response.get("thought", "")
            action = react_response.get("action", "")
            action_input = react_response.get("action_input", "")
            
            self.file_logger.info(f"Response time: {response_time:.2f}s")
            self.file_logger.info(f"Thought: {thought}")
            self.file_logger.info(f"Action: {action}, Input: {action_input}")
            logger.info(f"Thought: {thought}")
            logger.info(f"Action: {action}, Input: {action_input}")
            
            # Execute the action
            action_start_time = time.time()
            observation, screenshots = await self._execute_action(action, action_input)
            action_time = time.time() - action_start_time
            
            self.file_logger.info(f"Action execution time: {action_time:.2f}s")
            self.file_logger.info(f"Observation: {observation}")
            logger.info(f"Observation: {observation}")
            logger.info(f"Current memory: {self.reflection_memory}")
            
            # Update the conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": json.dumps({
                    "thought": thought,
                    "action": action,
                    "action_input": action_input
                })
            })
            
            self.conversation_history.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
            
            # Prune conversation history if it exceeds maximum length
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
                # self.file_logger.info(f"Pruned conversation history to {self.max_history_length} messages")

            # Save conversation history to consolidated file
            with open(self.conversation_log_file, "a", encoding="utf-8") as f:
                json.dump(self.conversation_history[-2:], f, indent=2, ensure_ascii=False)  # Only save the latest exchange
            self.file_logger.info(f"Saved conversation history to {self.conversation_log_file}")

            if self.ui:
                self.ui.update_context_history(self.conversation_history)
                self.ui.update_last_action(action + ", " + action_input)
                self.ui.update_steps_count(self.step_count)

            ### NEW ACTION
            # Under real benchmark (not lite), take screenshot here
            if not screenshots or len(screenshots) == 0:
                screenshot = await self.browser.get_screenshot()
                screenshots = [screenshot]
                # Save screenshot
                screenshot_path = screenshot_dir / f"game_screen_step_{self.step_count}.jpg"
                with open(screenshot_path, "wb") as f:
                    f.write(screenshot)
                if self.ui:
                    self.ui.take_screenshot(monitor_dir, f"game_screen_step_{self.step_count}.jpg")
                self.file_logger.info(f"Saved step {self.step_count} screenshot to {screenshot_path}")
            
            # Check if the task is complete
            if checkpoints and is_same_image(observation, checkpoints[-1]):
                self.file_logger.info("Task completed successfully!")
                logger.info("Task completed successfully!")
                break
                
        if step == max_steps - 1:
            self.file_logger.warning("Reached maximum number of steps without completing the task.")
            logger.warning("Reached maximum number of steps without completing the task.")
    
    async def _execute_action(self, action: str, action_input: str) -> str:
        """Execute an action and return the observation."""
        try:
            self.file_logger.info(f"Executing action: {action} with input: {action_input}")
            
            # Update UI with currently executing action
            if self.ui:
                self.ui.update_executing_action(f"{action}: {action_input}")

            # Execute the action
            result = None
            screenshots = []
            if self.lite:
                self.file_logger.info("Lite mode is enabled, pausing game with Alt+Pause key...")
                await self.browser.press_key("Alt+Pause", delay_ms=0)
                await asyncio.sleep(0.01)

            if action == "click":
                x, y = self.browser.current_mouse_position
                click_options = {}
                if action_input:
                    if "right" in action_input.lower():
                        click_options["button"] = "right"
                    
                    modifiers = []
                    if "shift" in action_input.lower():
                        modifiers.append("Shift")
                    if "ctrl" in action_input.lower():
                        modifiers.append("Control")
                    if "alt" in action_input.lower():
                        modifiers.append("Alt")
                    if modifiers:
                        click_options["modifiers"] = modifiers
                else:
                    click_options = None
                
                self.file_logger.info(f"Clicking at coordinates: ({x}, {y}) with options: {click_options}")
                await self.browser.click(x, y, click_options)
                result = f"Mouse clicked at ({x}, {y}) with options: {click_options}"

            elif action == "move":
                x, y = map(float, action_input.split(","))
                x_start, y_start = self.browser.current_mouse_position
                self.file_logger.info(f"Moving mouse from: ({x_start}, {y_start}) to: ({x}, {y})")
                await self.browser.move_mouse(x, y)
                result = f"Mouse moved to ({x}, {y})"

            elif action == "drag":
                x, y = map(float, action_input.split(","))
                x_start, y_start = self.browser.current_mouse_position
                self.file_logger.info(f"Dragging from: ({x_start}, {y_start}) to: ({x}, {y})")
                await self.browser.drag(x, y)
                result = f"Mouse dragged to ({x}, {y})"

            elif action == "scroll_down":
                amount = int(action_input)
                self.file_logger.info(f"Scrolling down: {amount}px")
                await self.browser.scroll_down(amount)
                result = f"Scrolled down {amount} pixels."

            elif action == "scroll_up":
                amount = int(action_input)
                self.file_logger.info(f"Scrolling up: {amount}px")
                await self.browser.scroll_up(amount)
                result = f"Scrolled up {amount} pixels."

            elif action == "write":
                self.file_logger.info(f"Typing text: {action_input}")
                await self.browser.type_text(action_input)
                result = f"Typed: {action_input}"

            elif action == "press_key":
                self.file_logger.info(f"Pressing key: {action_input}")
                if "," in action_input:
                    keys = action_input.split(",")
                    for key in keys:
                        await self.browser.press_key(key.strip(), lite_mode=self.lite, delay_ms=self.press_key_delay)
                        for _ in range(self.num_screenshots_per_action): 
                            screenshot = await self.browser.get_screenshot()
                            screenshots.append(screenshot)
                            await asyncio.sleep(0.05) 
                    result = f"Pressed keys: {action_input}"
                else:
                    await self.browser.press_key(action_input, lite_mode=self.lite, delay_ms=self.press_key_delay)
                    result = f"Pressed key: {action_input}"

            elif action == "hold_key":
                parts = action_input.split(",")
                key = parts[0]
                duration = float(parts[1]) if len(parts) > 1 else 0.5
                self.file_logger.info(f"Holding key: {key} for {duration}s")
                await self.browser.press_key(key, lite_mode=self.lite, delay_ms=duration)
                result = f"Held key {key} for {duration} seconds"

            elif action == "done":
                self.file_logger.info("Agent marked task as complete")
                result = "Task completed."

            elif action == "error":
                self.file_logger.error(f"Agent reported error: {action_input}")
                result = f"Error occurred: {action_input}"

            else:
                self.file_logger.warning(f"Unknown action: {action}")
                result = f"Unknown action: {action}"

            if self.lite:
                start_time = time.time()
                # Take screenshots for approximately 0.3 seconds
                for _ in range(5):  # 3 * 0.1s = 0.3s
                    screenshot = await self.browser.get_screenshot()
                    screenshots.append(screenshot)
                    await asyncio.sleep(0.05) 

                # Pause game
                await self.browser.press_key("Alt+Pause", delay_ms=0)

                duration = time.time() - start_time

                for i, screenshot in enumerate(screenshots):
                    self.lite_counter += 1
                    screenshot_dir = self.log_dir / "lite_screenshots"
                    monitor_dir = screenshot_dir / "monitor"
                    screenshot_dir.mkdir(exist_ok=True)
                    monitor_dir.mkdir(exist_ok=True)
                    screenshot_path = screenshot_dir / f"screenshot_{self.lite_counter}.jpg"
                    with open(screenshot_path, "wb") as f:
                        f.write(screenshot)
                    if self.ui:
                        self.ui.take_screenshot(monitor_dir, f"screenshot_{self.lite_counter}.jpg")
                duration = time.time() - start_time

                self.file_logger.info(f"Paused for {duration:.2f}s and took {len(screenshots)} screenshots")

            # Clear the executing action status when done
            if self.ui:
                self.ui.update_executing_action("")
            
            return result if result else f"Unknown action: {action}", screenshots

        except Exception as e:
            error_msg = f"Error executing action: {str(e)}"
            self.file_logger.error(error_msg)
            logger.error(error_msg)
            
            # Clear the executing action status on error
            if self.ui:
                self.ui.update_executing_action("")
            if self.lite:
                await self.browser.press_key("Alt+Pause", delay_ms=0)
            
            return error_msg, None

    async def capture_screenshots(self):

        screenshot_dir = self.log_dir / "game_screen_cont"
        monitor_dir = self.log_dir / "monitor_cont"
        screenshot_dir.mkdir(exist_ok=True)
        monitor_dir.mkdir(exist_ok=True)
        count = 0
        await asyncio.sleep(10)
        while True:
            try:
                screenshot = await self.browser.get_screenshot()
                # Save screenshot
                screenshot_path = screenshot_dir / f"game_screen_step_{count}.jpg"
                with open(screenshot_path, "wb") as f:
                    f.write(screenshot)
                if self.ui:
                    self.ui.take_screenshot(monitor_dir, f"game_screen_step_{count}.jpg")
                self.file_logger.info(f"Saved step {count} screenshot to {screenshot_path}")
                count += 1
                await asyncio.sleep(0.1)  # Wait 0.1 seconds before next capture
            except Exception as e:
                self.file_logger.error(f"Error capturing screenshot: {e}")
                await asyncio.sleep(0.1)  # Still wait on error before retrying

    async def start(self, initial_url: str) -> None:
        """
        Start the agent by initializing the browser.
        """
        self.file_logger.info("Starting browser")
        await self.browser.start()
        

        self.file_logger.info(f"Navigating to URL: {initial_url}")
        
        # Navigate to the initial URL
        start_time = time.time()
        await self.browser.navigate(initial_url)
        load_time = time.time() - start_time
        self.file_logger.info(f"Page loaded in {load_time:.2f}s")

        # Pre-loaded actions based on game
        await self.browser.pre_load(self.game)

        # Start the screenshot capture loop
        # asyncio.create_task(self.capture_screenshots())
