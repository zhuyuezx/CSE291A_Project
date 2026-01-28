#!/usr/bin/env python3
"""
Visual Game Agent for Zork
==========================
A proof-of-concept AI agent that plays the text adventure game Zork by:
1. Capturing the terminal screen (pixels only - no process memory access)
2. Using Qwen3-VL-8B vision model to read and understand the text
3. Generating commands via keyboard automation

Author: AI Research PoC
Usage: python zork_agent.py
"""

import time
import sys
import os
from typing import Optional, Tuple, List

import torch
from PIL import Image
import mss
import pyautogui
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ============================================================================
# CONFIGURATION - Adjust these values for your setup
# ============================================================================

# Screen capture region: (x, y, width, height)
# To find your terminal coordinates:
#   1. Open your terminal and run Zork
#   2. Take a screenshot and use an image editor to find pixel coordinates
#   3. x, y = top-left corner of the terminal window
#   4. width, height = dimensions of the capture area
SCREEN_REGION = {
    "left": 416,     # X coordinate of top-left corner
    "top": 294,       # Y coordinate of top-left corner  
    "width": 953,   # Width of capture area
    "height": 597, # Height of capture area
}

# Timing configuration
COUNTDOWN_SECONDS = 3     # Seconds before starting (to switch to game window)
TURN_DELAY_SECONDS = 5    # Seconds between turns (for game to render text)

# Model configuration
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 64

# Knowledge base file with game info and strategies
KNOWLEDGE_FILE = "zork_knowledge.txt"

# Memory configuration - track recent commands to avoid repetition
MAX_COMMAND_HISTORY = 100  # Number of recent commands to remember


# ============================================================================
# VISUAL AGENT CLASS
# ============================================================================

class VisualAgent:
    """
    An AI agent that plays games by looking at screen pixels and typing commands.
    
    Uses Qwen3-VL-8B-Instruct for vision-language understanding to read
    terminal text and generate appropriate game commands.
    """
    
    def __init__(self):
        """
        Initialize the Visual Agent by loading the Qwen3-VL model and processor.
        
        The model is loaded with automatic device mapping, preferring MPS on Mac.
        """
        print("[Agent] Initializing Visual Agent...")
        print(f"[Agent] Loading model: {MODEL_ID}")
        
        # Determine the best available device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("[Agent] Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("[Agent] Using CUDA GPU")
        else:
            self.device = "cpu"
            print("[Agent] Using CPU (this will be slow)")
        
        # Load the model with appropriate settings for the device
        # Note: For MPS, we use float16 to reduce memory usage
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        
        # Move to device if not using device_map="auto"
        if self.device in ["mps", "cpu"]:
            self.model = self.model.to(self.device)
        
        # Load the processor (tokenizer + image processor)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        
        # Initialize screen capture
        self.sct = mss.mss()
        
        # Initialize command history for long-term memory
        self.command_history: List[str] = []
        
        # Load knowledge base
        self.knowledge = self._load_knowledge()
        
        print("[Agent] Initialization complete!")
    
    def _load_knowledge(self) -> str:
        """Load the Zork knowledge base from file."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        knowledge_path = os.path.join(script_dir, KNOWLEDGE_FILE)
        
        if os.path.exists(knowledge_path):
            with open(knowledge_path, 'r') as f:
                knowledge = f.read()
            print(f"[Agent] Loaded knowledge base from {KNOWLEDGE_FILE}")
            return knowledge
        else:
            print(f"[Agent] Warning: Knowledge file {KNOWLEDGE_FILE} not found")
            return ""
    
    def capture_state(self, region: Optional[dict] = None) -> Image.Image:
        """
        Capture a region of the screen and return as a PIL Image.
        
        Args:
            region: Dictionary with 'left', 'top', 'width', 'height' keys.
                   If None, uses the default SCREEN_REGION configuration.
        
        Returns:
            PIL.Image.Image: The captured screen region as an RGB image.
        """
        if region is None:
            region = SCREEN_REGION
        
        # Capture the screen region
        screenshot = self.sct.grab(region)
        
        # Convert to PIL Image (mss returns BGRA, convert to RGB)
        img = Image.frombytes(
            "RGB",
            (screenshot.width, screenshot.height),
            screenshot.rgb,
        )
        
        return img
    
    def think(self, image: Image.Image) -> str:
        """
        Analyze the screen image and decide the next game command.
        
        Uses Qwen3-VL to read the terminal text in the image and generate
        an appropriate Zork command based on the game state, knowledge base,
        and command history to avoid repetition.
        
        Args:
            image: PIL Image of the game screen.
        
        Returns:
            str: The next command to execute (e.g., 'open mailbox', 'go north').
        """
        # Build history string for context
        history_str = ""
        stuck_warning = ""
        if self.command_history:
            recent_cmds = self.command_history[-MAX_COMMAND_HISTORY:]
            history_str = "\n\nYour recent commands (DO NOT REPEAT these if they didn't work):\n"
            for i, cmd in enumerate(recent_cmds, 1):
                history_str += f"  {i}. {cmd}\n"
            
            # REPETITION DETECTION: Check if we're stuck in a loop
            if len(self.command_history) >= 3:
                last_3 = self.command_history[-3:]
                # Check if last 3 commands are similar movement commands
                movement_cmds = ['go', 'north', 'south', 'east', 'west', 'n', 's', 'e', 'w', 'up', 'down', 'u', 'd']
                is_stuck = all(any(m in cmd.lower() for m in movement_cmds) for cmd in last_3)
                
                if is_stuck:
                    stuck_warning = """
*** CRITICAL: YOU ARE STUCK IN A LOOP! ***
You have been repeating movement commands and going nowhere.
YOU MUST TRY SOMETHING COMPLETELY DIFFERENT:
- Try: examine, take, open, read, look, inventory
- Try a DIFFERENT direction: north, south, up, down
- Try: look around, examine room, check inventory
DO NOT USE THE SAME DIRECTION AGAIN!
"""
        
        # System prompt with knowledge and anti-repetition guidance
        system_prompt = f"""You are playing Zork. Your goal is to explore and score points.

*** COMBAT PRIORITY (MOST IMPORTANT!) ***
If you see ANY of these, you MUST type "kill troll with sword" IMMEDIATELY:
- "sword is glowing" = DANGER! Enemy nearby!
- "troll" = ATTACK NOW with: kill troll with sword
- "brandishing" = Enemy is attacking, fight back!
DO NOT try to move when troll is present - you will DIE!

HOW TO READ THE TERMINAL:
1. The room description tells you EXACTLY which directions you can go
   - "passage leads to the west" = you can go WEST
   - "exits to the south and to the east" = you can go SOUTH or EAST only
   - "There is a wall there" = that direction is BLOCKED, try different direction
2. Objects mentioned in the text can be interacted with
   - "A lamp is here" = you can TAKE LAMP
   - "There is a mailbox here" = you can OPEN MAILBOX

VALID COMMANDS:
- Movement: go north, go south, go east, go west, go up, go down, n, s, e, w, u, d
- Objects: take [thing], open [thing], read [thing], drop [thing]
- Look: look, inventory
- Combat: kill troll with sword (keep repeating until troll dies!)

INVALID (do NOT use):
- examine room, examine floor, examine ceiling, examine wall
- go northeast, go northwest

{self.knowledge}

RULES:
- Output ONLY the command
- If troll or glowing sword mentioned = "kill troll with sword"
- READ room text for available exits
- Only interact with objects MENTIONED in text
{stuck_warning}{history_str}"""
        
        # Build the message format for Qwen3-VL
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": "Based on what you see, what is the best NEW command to try next?",
                    },
                ],
            },
        ]
        
        # Prepare inputs using the processor
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Process vision info from the messages
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Create model inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the correct device
        inputs = inputs.to(self.device)
        
        # Generate the response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # Deterministic for consistency
            )
        
        # Decode the generated tokens (skip the input tokens)
        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        command = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up the command (remove extra whitespace, newlines)
        command = command.strip().split('\n')[0].strip()
        
        # ANTI-REPETITION: If this command was used in last 5 commands, force something different
        if len(self.command_history) >= 2:
            recent_5 = [c.lower().strip() for c in self.command_history[-5:]]
            if command.lower().strip() in recent_5:
                print(f"[Agent] WARNING: '{command}' was used recently, forcing alternative...")
                command = self._get_alternative_command(command)
        
        # Add to command history
        self.command_history.append(command)
        
        return command
    
    def _get_alternative_command(self, repeated_cmd: str) -> str:
        """
        Generate an alternative command when the model keeps repeating.
        """
        repeated_lower = repeated_cmd.lower()
        
        # If it's a movement command, try opposite direction or look
        opposites = {
            'go east': 'go north',
            'go west': 'go south', 
            'go north': 'go west',
            'go south': 'go east',
            'east': 'north',
            'west': 'south',
            'north': 'west',
            'south': 'east',
            'e': 'n',
            'w': 's',
            'n': 'w',
            's': 'e',
        }
        
        for cmd, alt in opposites.items():
            if cmd in repeated_lower:
                print(f"[Agent] Trying opposite direction: {alt}")
                return alt
        
        # Default fallback: try useful exploration commands
        fallbacks = ['look', 'inventory', 'examine room', 'go up', 'go down']
        import random
        fallback = random.choice(fallbacks)
        print(f"[Agent] Using fallback: {fallback}")
        return fallback
    
    def act(self, command: str) -> None:
        """
        Execute the command by typing it and pressing Enter.
        
        Uses pyautogui for keyboard automation. The game window must be
        in focus for this to work.
        
        Args:
            command: The game command to type (e.g., 'go north').
        """
        if not command:
            print("[Agent] Warning: Empty command, skipping action")
            return
        
        # Type the command with a small interval between keys for reliability
        pyautogui.typewrite(command, interval=0.05)
        
        # Press Enter to submit the command
        pyautogui.press('enter')


# ============================================================================
# MAIN LOOP
# ============================================================================

def countdown(seconds: int) -> None:
    """Display a countdown timer before starting the agent."""
    print(f"\n[Agent] Starting in {seconds} seconds...")
    print("[Agent] Switch to your Zork terminal window NOW!")
    print("[Agent] Move mouse to top-left corner (0,0) to abort at any time.\n")
    
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    print("  GO!\n")


def main():
    """
    Main entry point for the Visual Game Agent.
    
    Runs the Capture -> Think -> Act loop until interrupted.
    """
    # Enable PyAutoGUI fail-safe: moving mouse to corner (0,0) aborts
    pyautogui.FAILSAFE = True
    
    print("=" * 60)
    print("  VISUAL GAME AGENT FOR ZORK")
    print("  Proof of Concept")
    print("=" * 60)
    print()
    print("Safety: Move mouse to top-left corner (0,0) to abort.")
    print(f"Screen region: {SCREEN_REGION}")
    print()
    
    # Initialize the agent (loads the model)
    try:
        agent = VisualAgent()
    except Exception as e:
        print(f"[Error] Failed to initialize agent: {e}")
        sys.exit(1)
    
    # Countdown to allow switching to the game window
    countdown(COUNTDOWN_SECONDS)
    
    turn = 0
    
    try:
        while True:
            turn += 1
            print(f"\n{'='*40}")
            print(f"[Turn {turn}]")
            print('='*40)
            
            # Step 1: Capture the screen
            print("[Agent] Capturing screen...")
            image = agent.capture_state()
            
            # Save screenshot for debugging (overwrites each turn)
            image.save("current_screenshot.png")
            
            # Step 2: Think about the next action
            print("[Agent] Analyzing image and thinking...")
            command = agent.think(image)
            print(f"[Agent] Decided command: '{command}'")
            
            # Step 3: Execute the action
            print(f"[Agent] Typing command...")
            agent.act(command)
            print("[Agent] Command sent!")
            
            # Wait for the game to process and render
            print(f"[Agent] Waiting {TURN_DELAY_SECONDS}s for game to respond...")
            time.sleep(TURN_DELAY_SECONDS)
    
    except pyautogui.FailSafeException:
        print("\n\n[Agent] FAIL-SAFE TRIGGERED!")
        print("[Agent] Mouse moved to corner - aborting.")
    
    except KeyboardInterrupt:
        print("\n\n[Agent] Interrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n\n[Agent] Error: {e}")
        raise
    
    finally:
        print("\n[Agent] Agent stopped. Goodbye!")


if __name__ == "__main__":
    main()
