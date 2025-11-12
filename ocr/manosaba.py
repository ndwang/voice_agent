"""
Script demonstrating OcrMonitorApp with a custom handler.

This script creates an OcrMonitorApp instance and sets up a custom handler
to process OCR results from monitored regions.
"""

import tkinter as tk
import numpy as np
from rapidfuzz import fuzz, process
from ocr import OcrMonitorApp


def custom_text_handler(outputs):
    """
    Custom handler function that processes OCR results.
    
    Args:
        outputs: List of dicts or None [region1_result, region2_result]
                - None if region didn't change
                - Dict with 'rec_texts' and 'rec_polys' keys if changed
    """
    # Sort region1_result by leftmost x coordinate if it exists
    region1_result = outputs[0] if len(outputs) > 0 else None
    if region1_result is not None:
        texts = region1_result['rec_texts']
        polygons = region1_result['rec_polys']
        
        if not texts:
            print("__")
        if texts and polygons and len(texts) == len(polygons):
            # Create list of (leftmost_x, text, polygon) tuples
            text_poly_pairs = []
            for text, poly in zip(texts, polygons):
                # Extract x coordinates from all 4 corners of the polygon
                # poly is a numpy array of 4 points, each point is [x, y]
                leftmost_x = np.min(poly[:, 0])
                text_poly_pairs.append((leftmost_x, text, poly))
            
            # Sort by leftmost x coordinate
            text_poly_pairs.sort(key=lambda x: x[0])
            
            # Merge texts into a single string
            name = ' '.join(pair[1] for pair in text_poly_pairs)
            
            # Predefined names to match against
            predefined_names = [
                "紫藤亚里沙",
                "夏目安安",
                "泽渡可可",
                "樱羽艾玛",
                "远野汉娜",
                "二阶堂希罗",
                "看守",
                "莲见蕾雅",
                "宝生玛格",
                "冰上梅露露",
                "佐伯米莉亚",
                "黑部奈叶香",
                "城崎诺亚",
                "橘雪莉",
                "典狱长",
                "月代雪",
            ]
            
            # Use fuzzy matching to find the best match
            if name.strip():  # Only match if name is not empty
                best_match, score, index = process.extractOne(
                    name, 
                    predefined_names, 
                    scorer=fuzz.ratio
                )
            print(f"\033[92m{best_match}\033[0m")

    if outputs[1] is not None:
        text = outputs[1]['rec_texts']
        if not text:
            print("......")
        if text:
            text_output = '\n'.join(text)
            print(text_output)


def main():
    """Main function to create and run the OCR monitor app."""
    # Create the root window
    root = tk.Tk()
    root.title("OCR Monitor with Custom Handler")
    root.geometry("350x300")
    root.resizable(False, False)
    
    # Create the OCR monitor app
    app = OcrMonitorApp(root)
    
    # Set the custom handler
    app.text_handler_hook = custom_text_handler
    
    # Start the tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()

