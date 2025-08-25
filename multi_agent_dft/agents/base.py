import logging
from typing import List, Dict, Any, Optional
import time
import json
import os
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_dft.utils.logging import get_logger

logger = get_logger(__name__)

class Agent:
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, model=None, config=None):
        """
        Initialize an agent.
        
        Args:
            name: The name of the agent
            model: Optional model name to use
            config: Optional configuration dictionary
        """
        self.name = name
        self.memory: List[Dict[str, str]] = []
        self.memory_size = 10  # Default memory size
        self.config = config or {}
        
        # Try to import OpenAI. If it fails, log a warning but don't crash.
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            self.model = model or "gpt-4"
            self.temperature = 0.4
            self.top_p = 0.9  # default nucleus sampling
            self.top_p = 0.9
            self.max_tokens = 1000
            self.timeout = 30
        except ImportError:
            logger.warning("OpenAI package not found. Chat functionality will not work.")
            self.client = None
        
    def _add_to_memory(self, message: Dict[str, str]) -> None:
        """
        Add a message to the agent's memory.
        
        Args:
            message: The message to add
        """
        self.memory.append(message)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Send a chat request to the model.
        
        Args:
            messages: List of message dictionaries
            temperature: Optional temperature override
            top_p: Optional nucleus sampling override
            
        Returns:
            The model's response
        
        Raises:
            Exception: If the API call fails after retries
        """
        if self.client is None:
            return "Error: OpenAI client not initialized."
        
        try:
            logger.debug(f"Agent {self.name} sending chat request")
            full_messages = self.memory + messages
            
            # Determine sampling parameters
            temp = temperature if temperature is not None else self.temperature
            p_val = top_p if top_p is not None else self.top_p
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=temp,
                top_p=p_val,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            
            content = response.choices[0].message.content.strip()
            
            # Add the conversation to memory
            for message in messages:
                self._add_to_memory(message)
            self._add_to_memory({"role": "assistant", "content": content})
            
            return content
            
        except Exception as e:
            logger.error(f"Error in chat request for agent {self.name}: {str(e)}")
            return f"Error in chat request: {str(e)}"
    
    def query(self, user_content, system_content=None):
        """
        A simple interface to query the agent.
        
        Args:
            user_content: User message content
            system_content: Optional system message content
            
        Returns:
            The model's response
        """
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        
        return self.chat(messages)
    
    def reset_memory(self) -> None:
        """Reset the agent's memory."""
        self.memory = []
        
    def save_memory(self, file_path: str) -> None:
        """
        Save the agent's memory to a file.
        
        Args:
            file_path: Path to save the memory
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.memory, f, indent=2)
            logger.info(f"Agent {self.name} memory saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving agent memory: {str(e)}")
    
    def load_memory(self, file_path: str) -> None:
        """
        Load the agent's memory from a file.
        
        Args:
            file_path: Path to load the memory from
        """
        try:
            with open(file_path, 'r') as f:
                self.memory = json.load(f)
            logger.info(f"Agent {self.name} memory loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading agent memory: {str(e)}")
