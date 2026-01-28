#!/usr/bin/env python3
"""
Screen Region Finder - Helps you find the correct coordinates for Zork window.

This script will:
1. Show your current mouse position in real-time
2. Let you click two corners to define a region
3. Save a test screenshot of that region
"""

import mss
from PIL import Image, ImageDraw
import time

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False


def get_mouse_position():
    """Get current mouse position."""
    if HAS_PYAUTOGUI:
        return pyautogui.position()
    else:
        print("pyautogui not available for mouse tracking")
        return (0, 0)


def capture_full_screen():
    """Capture the entire screen."""
    with mss.mss() as sct:
        # Get primary monitor
        monitor = sct.monitors[1]  # Monitor 1 is the primary display
        screenshot = sct.grab(monitor)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"), monitor


def capture_region(left, top, width, height):
    """Capture a specific region."""
    with mss.mss() as sct:
        region = {"left": left, "top": top, "width": width, "height": height}
        screenshot = sct.grab(region)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")


def main():
    print("="*60)
    print("SCREEN REGION FINDER")
    print("="*60)
    
    # Show monitor info
    with mss.mss() as sct:
        print("\nAvailable monitors:")
        for i, m in enumerate(sct.monitors):
            print(f"  Monitor {i}: {m}")
    
    print("\n" + "-"*60)
    print("MODE 1: Mouse Position Tracker")
    print("-"*60)
    print("Move your mouse to the corners of your Zork window.")
    print("Press Ctrl+C when done to see the coordinates.\n")
    
    positions = []
    print("Recording mouse positions... (Ctrl+C to stop)")
    print("Move to TOP-LEFT corner of Zork and wait 2 seconds...")
    
    try:
        time.sleep(5)
        pos1 = get_mouse_position()
        print(f"  TOP-LEFT: ({pos1[0]}, {pos1[1]})")
        positions.append(pos1)
        
        print("Move to BOTTOM-RIGHT corner of Zork and wait 2 seconds...")
        time.sleep(5)
        pos2 = get_mouse_position()
        print(f"  BOTTOM-RIGHT: ({pos2[0]}, {pos2[1]})")
        positions.append(pos2)
        
    except KeyboardInterrupt:
        print("\nStopped.")
        return
    
    if len(positions) >= 2:
        left = min(positions[0][0], positions[1][0])
        top = min(positions[0][1], positions[1][1])
        right = max(positions[0][0], positions[1][0])
        bottom = max(positions[0][1], positions[1][1])
        width = right - left
        height = bottom - top
        
        print("\n" + "="*60)
        print("CALCULATED REGION:")
        print("="*60)
        print(f"""
SCREEN_REGION = {{
    "left": {left},     # X coordinate of top-left corner
    "top": {top},       # Y coordinate of top-left corner  
    "width": {width},   # Width of capture area
    "height": {height}, # Height of capture area
}}
""")
        
        print("Taking test screenshot of this region...")
        try:
            test_img = capture_region(left, top, width, height)
            test_img.save("region_test.png")
            print(f"Saved to region_test.png - CHECK THIS FILE!")
            print(f"Image size: {test_img.size}")
        except Exception as e:
            print(f"Error capturing region: {e}")
    
    print("\n" + "-"*60)
    print("MODE 2: Test Current Settings")
    print("-"*60)
    
    # Test current settings from both files
    current_settings = [
        ("zork_agent.py settings", {"left": 832, "top": 590, "width": 1911, "height": 1187}),
        ("zork_vision_debug.py settings", {"left": 590, "top": 832, "width": 1911, "height": 1187}),
    ]
    
    for name, region in current_settings:
        print(f"\nTesting {name}:")
        print(f"  Region: left={region['left']}, top={region['top']}, width={region['width']}, height={region['height']}")
        try:
            img = capture_region(**region)
            filename = f"test_{name.replace(' ', '_').replace('.py', '')}.png"
            img.save(filename)
            print(f"  Saved to: {filename}")
            print(f"  Size: {img.size}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
