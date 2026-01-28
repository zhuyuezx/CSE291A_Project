#!/usr/bin/env python3
"""
Zork Vision Debug Agent - Tests if VL model can read game text correctly.

This script captures the screen and asks the model to describe what it sees,
without taking any actions. Useful for debugging vision/text detection.
"""

import time
import torch
import mss
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Screen capture region - MUST MATCH zork_agent.py settings!
SCREEN_REGION = {
    "left": 415,     # X coordinate of top-left corner
    "top": 295,       # Y coordinate of top-left corner  
    "width": 952,   # Width of capture area
    "height": 597, # Height of capture area
}

# Model configuration
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 512  # More tokens for detailed description

# Image resize to prevent memory issues
MAX_IMAGE_SIZE = 800


class VisionDebugAgent:
    """Debug agent that just reads and describes the screen."""
    
    def __init__(self):
        print("[Debug] Initializing Vision Debug Agent...")
        
        # Detect device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[Debug] Using MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("[Debug] Using CUDA")
        else:
            self.device = torch.device("cpu")
            print("[Debug] Using CPU")
        
        # Load model
        print(f"[Debug] Loading model: {MODEL_ID}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if self.device.type == "mps" else torch.bfloat16,
            device_map="auto" if self.device.type != "mps" else None,
            trust_remote_code=True,
        )
        
        if self.device.type == "mps":
            self.model = self.model.to(self.device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        
        # Initialize screen capture
        self.sct = mss.mss()
        
        print("[Debug] Initialization complete!")
    
    def capture_screen(self) -> Image.Image:
        """Capture and resize screen."""
        screenshot = self.sct.grab(SCREEN_REGION)
        image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        # Resize to prevent memory issues
        width, height = image.size
        if max(width, height) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.LANCZOS)
        
        return image
    
    def analyze_screen(self, image: Image.Image) -> str:
        """Ask the model to describe what it sees."""
        
        prompt = """Look at this Zork game terminal screenshot and answer:

1. ROOM NAME: What room or location is shown? (if visible)

2. ROOM DESCRIPTION: What is the full text description of the current room?

3. OBJECTS: List any objects mentioned (mailbox, lamp, sword, etc.)

4. AVAILABLE EXITS: What directions can the player go? (north, south, east, west, up, down)

5. LAST COMMAND: What was the last command typed? (shown after the > prompt)

6. GAME RESPONSE: What was the game's response to the last command?

7. ANY DANGER?: Is there mention of troll, glowing sword, or other dangers?

Please read the terminal text carefully and report exactly what you see."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        
        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        return response


def main():
    """Main debug loop."""
    agent = VisionDebugAgent()
    
    print("\n" + "="*60)
    print("VISION DEBUG MODE")
    print("="*60)
    print("This will capture the screen and show what the model sees.")
    print("Press Ctrl+C to stop.\n")
    
    # Countdown
    print("[Debug] Starting in 3 seconds...")
    print("[Debug] Switch to your Zork terminal window NOW!")
    for i in [3, 2, 1]:
        print(f"  {i}...")
        time.sleep(1)
    print("  GO!\n")
    
    turn = 0
    try:
        while True:
            turn += 1
            print("="*60)
            print(f"[Turn {turn}] Capturing screen...")
            print("="*60)
            
            # Capture
            image = agent.capture_screen()
            
            # Save screenshot for reference
            image.save(f"debug_screenshot_{turn}.png")
            print(f"[Debug] Screenshot saved to debug_screenshot_{turn}.png")
            
            # Analyze
            print("\n[Debug] Analyzing image...\n")
            analysis = agent.analyze_screen(image)
            
            print("-"*40)
            print("MODEL'S ANALYSIS:")
            print("-"*40)
            print(analysis)
            print("-"*40)
            
            # Wait for user input or timeout
            print("\n[Debug] Press Enter to capture next frame, or Ctrl+C to quit...")
            try:
                input()
            except EOFError:
                time.sleep(5)  # Auto-continue after 5 seconds if no input
                
    except KeyboardInterrupt:
        print("\n\n[Debug] Stopped by user.")
    
    print("[Debug] Debug session ended.")


if __name__ == "__main__":
    main()
