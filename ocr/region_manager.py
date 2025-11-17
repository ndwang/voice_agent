"""
OCR Region Manager

Standalone utility for selecting OCR monitoring region.
Uses interactive overlay for region selection.
"""
import json
import tkinter as tk
from tkinter import messagebox
import time
from ocr import RegionSelector


def select_region(config_file: str = "ocr_region.json"):
    """
    Interactive region selection utility.
    
    Args:
        config_file: Path to save region configuration
    """
    root = tk.Tk()
    root.title("OCR Region Selector")
    root.geometry("300x150")
    root.resizable(False, False)
    
    region = None
    
    def on_region_selected(selected_region):
        nonlocal region
        region = selected_region
        if region:
            # Save to config file
            config = {
                "region": region,
                "x": region[0],
                "y": region[1],
                "width": region[2],
                "height": region[3]
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Region saved to {config_file}: {region}")
            messagebox.showinfo("Success", f"Region selected and saved:\n{region}")
        root.quit()
    
    def start_selection():
        selector = RegionSelector(root, on_region_selected)
        selector.start_selection()
    
    # Instructions
    label = tk.Label(
        root,
        text="Click the button below to select a screen region\nfor OCR monitoring.",
        justify=tk.CENTER,
        padx=20,
        pady=20
    )
    label.pack()
    
    # Select button
    select_btn = tk.Button(
        root,
        text="Select Region",
        command=start_selection,
        padx=20,
        pady=10
    )
    select_btn.pack()
    
    root.mainloop()
    return region


def load_region(config_file: str = "ocr_region.json"):
    """Load region from config file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            if "region" in config:
                return tuple(config["region"])
            elif all(k in config for k in ["x", "y", "width", "height"]):
                return (config["x"], config["y"], config["width"], config["height"])
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading region config: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        # Load and print region
        region = load_region()
        if region:
            print(f"Loaded region: {region}")
        else:
            print("No region configuration found")
    else:
        # Interactive selection
        region = select_region()
        if region:
            print(f"Selected region: {region}")

