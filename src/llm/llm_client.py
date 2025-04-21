import base64
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal
import re
import litellm

# For local Qwen models
QWEN_AVAILABLE = True
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch
    from src.llm.qwen_vl_utils import process_vision_info, format_qwen_messages
except ImportError:
    QWEN_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_client")

class LLMClient:
    """
    Client for interacting with language models using litellm.
    """
    def __init__(
        self, 
        model: str, 
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_cost: float = 10.0,  # Maximum cost in USD
        log_dir: Optional[Path] = None,
        api_base: Optional[str] = None # For Ollama
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: The model to use (e.g., "gpt-4o", "claude-3-opus-20240229")
            api_key: The API key for the model provider
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate
            max_cost: The maximum cost in USD for the current session
            log_dir: Optional custom log directory path
            api_base: Optional API base URL for Ollama or other providers
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.total_cost = 0.0
        self.api_base = api_base
        
        # Set up logging directory
        if log_dir is None:
            model_name = model.replace("/", "-").replace(".", "-")
            self.log_dir = Path("logs") / "dos" / model_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.log_dir = log_dir
            
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logger
        self.file_logger = self._setup_file_logger()
        self.step_count = 0
        
        # Set the API key based on the model
        if "Qwen2.5-VL" in model and (api_key is None or api_key == ""):
            # Local Qwen model
            if not QWEN_AVAILABLE:
                raise ImportError("Qwen dependencies not available. Please install transformers and torch.")
            self.provider = "qwen_local"
            self.qwen_model = None
            self.qwen_processor = None
            # We'll load the model on first use to avoid memory usage if not needed
        elif "gpt" in model.lower() or "openai" in model.lower():
            if api_key is not None and api_key != "":
                os.environ["OPENAI_API_KEY"] = api_key
            self.provider = "openai"
        elif "claude" in model.lower() or "anthropic" in model.lower():
            if api_key is not None and api_key != "":
                os.environ["ANTHROPIC_API_KEY"] = api_key
            # Ensure model has the correct format for LiteLLM
            if not model.startswith("anthropic/"):
                self.model = f"anthropic/{model}"
            self.provider = "anthropic"
        elif self.api_base and "ollama" in self.api_base:
            # Configure for Ollama
            self.provider = "ollama"
            if not model.startswith("ollama/"):
                self.model = f"ollama/{model}"
            litellm.api_base = self.api_base
        else:
            # For other models, set a generic API key
            litellm.api_key = api_key
            self.provider = "other"
            
        self.file_logger.info(f"Initialized LLMClient with model: {self.model}, provider: {self.provider}")
        self.file_logger.info(f"Logging to: {self.log_dir}")
        logger.info(f"LLMClient logging to: {self.log_dir}")
        
        # Define cost per 1K tokens for different models
        self.token_costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-vision-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        }

    def _setup_file_logger(self) -> logging.Logger:
        """Set up a file logger for this session."""
        file_logger = logging.getLogger(f"llm_client_{id(self)}")
        file_logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / "llm_session.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add handler to logger
        file_logger.addHandler(file_handler)
        
        return file_logger
        
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for the current request."""
        model_base = self.model.split('/')[1] if '/' in self.model else self.model
        costs = self.token_costs.get(model_base, {"input": 0.0001, "output": 0.0002})  # Default low cost
        
        cost = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000
        return cost

    def _load_qwen_model(self):
        """Load the Qwen model and processor on first use"""
        if self.qwen_model is None or self.qwen_processor is None:
            self.file_logger.info(f"Loading Qwen model: {self.model}")
            try:
                # Extract the actual model name from our identifier
                model_name = self.model
                if '/' not in model_name:
                    model_name = f"Qwen/{model_name}"
                
                # Load the model
                self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype="auto", device_map="auto"
                )
                # Load the processor
                self.qwen_processor = AutoProcessor.from_pretrained(model_name)
                self.file_logger.info(f"Successfully loaded Qwen model and processor")
            except Exception as e:
                self.file_logger.error(f"Failed to load Qwen model: {str(e)}")
                raise
    
    async def generate_response(
        self, 
        system_message: Dict[str, str],
        messages: List[Dict[str, str]], 
        image_data: Optional[bytes | List[bytes]] = None
    ) -> str:
        """
        Generate a response from the language model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            image_data: Optional screenshot data to include in the prompt
            
        Returns:
            The generated response text
        """
        self.step_count += 1
        self.file_logger.info(f"Step {self.step_count} - Generating response")

        
        # If image data is provided, save it and add it to the last user message
        screenshot_path = None
        if image_data and messages and messages[-1]["role"] == "user":
            # Convert the last text message to a list format
            original_text = messages[-1]["content"]
            
            # Initialize the content list with the text message
            content_list = [{"type": "text", "text": original_text}]
            
            # Handle single image or list of images
            if isinstance(image_data, list):
                # Add each image in sequence
                for img in image_data:
                    base64_image = base64.b64encode(img).decode("utf-8")
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            else:
                # Single image case
                base64_image = base64.b64encode(image_data).decode("utf-8")
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            
            # Update the message content with the full list
            messages[-1]["content"] = content_list
        
        # Log request details
        self.file_logger.info(f"Request to model: {self.model}")
        self.file_logger.info(f"Temperature: {self.temperature}, Max tokens: {self.max_tokens}")
        
        if system_message is not None:
            messages = [system_message] + messages

        # Log the messages (excluding image data for brevity)
        messages_log = []
        for msg in messages:
            if isinstance(msg["content"], list):
                # Handle multimodal content
                text_parts = [item["text"] for item in msg["content"] if item.get("type") == "text"]
                content = " ".join(text_parts) + " [IMAGE]"
            else:
                content = msg["content"]
                
            messages_log.append({
                "role": msg["role"],
                "content": content[:200] + "..." if len(content) > 200 else content
            })
            
        self.file_logger.info(f"Messages: {json.dumps(messages_log, indent=2)}")
        
        # Generate response using the appropriate provider
        try:
            start_time = time.time()
            
            if self.provider == "qwen_local":
                # Use local Qwen model
                self._load_qwen_model()
                
                # Format messages for Qwen
                if system_message is not None:
                    # Add system message as the first user message
                    messages = [{
                        "role": "user",
                        "content": system_message["content"]
                    }] + messages
                
                # Process messages for Qwen
                qwen_messages = format_qwen_messages(messages)
                
                # Apply chat template
                text = self.qwen_processor.apply_chat_template(
                    qwen_messages, tokenize=False, add_generation_prompt=True
                )
                
                # Process images and videos
                image_inputs, video_inputs = process_vision_info(messages)
                
                # Prepare inputs
                inputs = self.qwen_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.qwen_model.device)
                
                # Generate output
                generated_ids = self.qwen_model.generate(
                    **inputs, max_new_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                # Decode output
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response_text = self.qwen_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # Estimate token counts for logging
                input_tokens = len(inputs.input_ids[0])
                output_tokens = len(generated_ids_trimmed[0])
                
                # No actual cost for local models, but log token counts
                request_cost = 0.0
                
            else:
                # Use litellm for API-based models
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Calculate and update cost
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                request_cost = self._calculate_cost(input_tokens, output_tokens)
                self.total_cost += request_cost
                
                # Check if we've exceeded the maximum cost
                if self.total_cost > self.max_cost:
                    raise Exception(f"Maximum cost limit (${self.max_cost}) exceeded. Total cost: ${self.total_cost:.2f}")
                
                # Extract response text
                response_text = response.choices[0].message.content
            
            response_time = time.time() - start_time
            
            # Log cost information
            self.file_logger.info(f"Request cost: ${request_cost:.4f}")
            self.file_logger.info(f"Total cost so far: ${self.total_cost:.2f}")
            self.file_logger.info(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
            logger.info(f"Total cost so far: ${self.total_cost:.2f}")
            
            # Log response details
            self.file_logger.info(f"Response time: {response_time:.2f}s")
            self.file_logger.info(f"Response length: {len(response_text)} characters")
            self.file_logger.info(f"Response: {response_text[:500]}...")
            
            # Write full response to a separate file
            response_file = self.log_dir / f"llm_responses.txt"
            with open(response_file, "a") as f:
                f.write(response_text)
                
            return response_text
        except Exception as e:
            self.file_logger.error(f"Error generating response: {str(e)}")
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

    async def generate_react_response(
        self, 
        task: str, 
        system_message: Dict[str, str],
        history: List[Dict[str, str]], 
        screenshot: Optional[bytes | List[bytes]] = None
    ) -> Dict[str, Any]:
        """
        Generate a ReACT (Reasoning, Action, Observation) response.
        
        Args:
            task: The task description
            history: The conversation history
            screenshot: Optional screenshot data
            
        Returns:
            A dictionary containing thought, action, and action_input
        """
        self.file_logger.info(f"Step {self.step_count} - Generating ReACT response")
        self.file_logger.info(f"Task: {task}")
        
        
        # Create the user message with the task
        user_message = {
            "role": "user",
            "content": f"{task}"
        }
        
        # Combine messages
        messages = [system_message] + history + [user_message]
        
        # Generate response
        response_text = await self.generate_response(None, messages, screenshot)
        
        # Parse the JSON response
        try:
            import json
            # Extract JSON from the response (in case there's additional text)
            json_match = re.search(r'.*```json\s*(.*?)\s*```(?!.*```)', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
                
            # Clean up any non-JSON text
            json_str = re.sub(r'^[^{]*', '', json_str)
            json_str = re.sub(r'[^}]*$', '', json_str)
            
            # Log the parsed response
            self.file_logger.info(f"Parsed json_str: {json_str}")

            response_dict = json.loads(json_str)
            
            # Ensure the response has the required fields
            required_fields = ["thought", "action", "action_input"]
            for field in required_fields:
                if field not in response_dict:
                    raise ValueError(f"Response missing required field: {field}")
            
            # Log the parsed response
            self.file_logger.info(f"Parsed ReACT response: {json.dumps(response_dict, indent=2)}")
            
            return response_dict
        except Exception as e:
            # If parsing fails, return a default response
            error_msg = f"Error parsing response: {str(e)}\nOriginal response: {response_text}"
            self.file_logger.error(error_msg)
            
            return {
                "thought": "I couldn't parse the response properly. Let me try again.",
                "action": "error",
                "action_input": error_msg,
                "memory": ""
            }
