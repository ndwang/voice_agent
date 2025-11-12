"""
Example usage of OcrMonitorApp from other scripts.

This demonstrates different ways to integrate the OCR monitoring system
into your own applications.
"""

import tkinter as tk
from ocr import OcrMonitorApp


# Example 1: Basic usage with custom text handler
def example_custom_handler():
    """Example with a custom text handler that processes extracted text."""
    
    def my_text_handler(text):
        """Custom handler that processes OCR results."""
        print(f"[CUSTOM HANDLER] Got text: {text}")
        # You can do anything here:
        # - Save to file
        # - Send to API
        # - Process with NLP
        # - Update a database
        # etc.
    
    root = tk.Tk()
    app = OcrMonitorApp(root)
    
    # Replace the default handler with your custom one
    app.text_handler_hook = my_text_handler
    
    root.mainloop()


# Example 2: Programmatic control (set region and start automatically)
def example_programmatic_control():
    """Example showing how to control the app programmatically."""
    
    def save_to_file_handler(text):
        """Handler that saves text to a file."""
        with open("ocr_output.txt", "a", encoding="utf-8") as f:
            f.write(f"{time.ctime()}: {text}\n")
    
    root = tk.Tk()
    app = OcrMonitorApp(root)
    
    # Set custom handler
    app.text_handler_hook = save_to_file_handler
    
    # Set region programmatically (x, y, width, height)
    # You can get these coordinates from another tool or hardcode them
    app.region = (100, 100, 800, 600)
    
    # Set interval programmatically (in milliseconds)
    app.interval_ms = 2000  # 2 seconds
    
    # Update status to reflect changes
    app.update_status()
    
    # Start monitoring automatically
    app.start_monitoring()
    
    root.mainloop()


# Example 3: Integration with other systems (e.g., webhook, database)
def example_integration():
    """Example showing integration with external systems."""
    
    class TextProcessor:
        """Example class that processes OCR text."""
        
        def __init__(self):
            self.text_history = []
        
        def process_text(self, text):
            """Process extracted text and send to external system."""
            # Store in history
            self.text_history.append(text)
            
            # Example: Send to webhook
            # import requests
            # requests.post("https://api.example.com/ocr", json={"text": text})
            
            # Example: Update database
            # db.insert("ocr_results", {"text": text, "timestamp": time.time()})
            
            print(f"Processed text #{len(self.text_history)}: {text[:50]}...")
    
    processor = TextProcessor()
    
    root = tk.Tk()
    app = OcrMonitorApp(root)
    
    # Use the processor's method as the handler
    app.text_handler_hook = processor.process_text
    
    root.mainloop()


# Example 4: Minimal usage (just use default behavior)
def example_minimal():
    """Simplest possible usage."""
    root = tk.Tk()
    app = OcrMonitorApp(root)
    root.mainloop()


# Example 5: Access OCR functionality directly
def example_direct_ocr_access():
    """Example showing how to access OCR reader directly for one-off OCR."""
    
    root = tk.Tk()
    app = OcrMonitorApp(root)
    
    # You can access the OCR reader directly
    # Note: This requires a region to be set first
    def do_one_off_ocr():
        if app.region:
            import pyautogui
            import numpy as np
            
            x, y, width, height = app.region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            img_array = np.array(screenshot)
            
            # Use the OCR reader directly
            results = app.ocr_reader.readtext(img_array)
            text = '\n'.join([result[1] for result in results])
            print(f"One-off OCR result: {text}")
        else:
            print("Please select a region first!")
    
    # Add a button to trigger one-off OCR
    import tkinter.ttk as ttk
    button = ttk.Button(root, text="Do One-Off OCR", command=do_one_off_ocr)
    button.pack()
    
    root.mainloop()


if __name__ == "__main__":
    import time
    
    # Uncomment the example you want to run:
    # example_minimal()
    # example_custom_handler()
    # example_programmatic_control()
    # example_integration()
    example_direct_ocr_access()

