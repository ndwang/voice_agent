# OCR Monitor

A Python application for monitoring screen regions and extracting text using OCR (Optical Character Recognition). Supports Chinese and English text extraction.

## Features

- **Region Selection**: Interactive GUI to select screen regions for monitoring
- **Periodic Monitoring**: Automatically captures and processes screenshots at configurable intervals
- **Smart Optimization**: Skips OCR when screenshots are identical (hash-based comparison)
- **Customizable Hooks**: Easy integration with custom text processing logic
- **Chinese & English Support**: Uses EasyOCR for multi-language text extraction

## Installation

```bash
# From the project root
uv sync
```

## Basic Usage

### Standalone Application

Run the OCR monitor as a standalone application:

```python
from ocr import OcrMonitorApp
import tkinter as tk

root = tk.Tk()
app = OcrMonitorApp(root)
root.mainloop()
```

Or simply run:
```bash
python ocr/ocr.py
```

### Custom Text Handler

Replace the default text handler with your own processing logic:

```python
from ocr import OcrMonitorApp
import tkinter as tk

def my_custom_handler(text):
    """Process extracted text."""
    print(f"Extracted: {text}")
    # Add your custom logic here:
    # - Save to file
    # - Send to API
    # - Process with NLP
    # - Update database
    # etc.

root = tk.Tk()
app = OcrMonitorApp(root)

# Set custom handler
app.text_handler_hook = my_custom_handler

root.mainloop()
```

### Programmatic Control

Control the app programmatically without using the GUI:

```python
from ocr import OcrMonitorApp
import tkinter as tk

root = tk.Tk()
app = OcrMonitorApp(root)

# Set region programmatically (x, y, width, height)
app.region = (100, 100, 800, 600)

# Set interval in milliseconds
app.interval_ms = 2000  # 2 seconds

# Update status display
app.update_status()

# Start monitoring
app.start_monitoring()

root.mainloop()
```

## API Reference

### OcrMonitorApp

Main application class for OCR monitoring.

#### Initialization

```python
app = OcrMonitorApp(master)
```

- `master`: Tkinter root window

#### Properties

- `region`: Tuple `(x, y, width, height)` or `None` - Screen region to monitor
- `interval_ms`: Integer - Monitoring interval in milliseconds (default: 3000)
- `is_running`: Boolean - Whether monitoring is currently active
- `text_handler_hook`: Callable - Function to call when text is extracted
- `ocr_reader`: EasyOCR Reader instance - Direct access to OCR engine

#### Methods

- `select_region()`: Opens region selection overlay
- `start_monitoring()`: Starts the monitoring loop
- `stop_monitoring()`: Stops the monitoring loop
- `toggle_monitoring()`: Toggles monitoring on/off
- `set_interval()`: Updates the monitoring interval from GUI
- `update_status()`: Updates the status label
- `cleanup()`: Cleans up resources and closes the application

#### Text Handler Hook

The text handler hook is a callable that receives extracted text:

```python
def text_handler(text: str) -> None:
    """Process extracted OCR text."""
    # Your processing logic here
    pass

app.text_handler_hook = text_handler
```

Default handler prints to terminal with timestamp.

## Examples

See `example_usage.py` for comprehensive examples including:
- Custom text handlers
- Programmatic control
- Integration with external systems
- Direct OCR access

## Architecture

- **RegionSelector**: Handles interactive region selection with transparent overlay
- **OcrMonitorApp**: Main application class managing GUI, monitoring loop, and OCR processing

## Dependencies

- `pyautogui`: Screenshot capture
- `easyocr`: OCR engine (supports Chinese and English)
- `pillow`: Image processing
- `numpy`: Array operations
- `tkinter`: GUI (usually included with Python)

## Notes

- The OCR reader is initialized once at startup (can take 10-30 seconds)
- Screenshots are hashed to skip OCR when content is unchanged
- The monitoring loop uses tkinter's `after()` method for non-blocking operation
- Region selection requires at least 10x10 pixels

