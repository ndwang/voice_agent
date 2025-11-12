import tkinter as tk
from tkinter import messagebox, ttk
import time
import hashlib
import pyautogui
from paddleocr import PaddleOCR
from PIL import Image
import io
import numpy as np


class RegionSelector:
    """Handles region selection with a transparent overlay."""
    
    MIN_REGION_SIZE = 10  # Minimum width/height in pixels
    
    def __init__(self, parent_window, on_region_selected):
        """
        Initialize the region selector.
        
        Args:
            parent_window: The main tkinter window to hide during selection
            on_region_selected: Callback function that receives (x, y, width, height)
        """
        self.parent_window = parent_window
        self.on_region_selected = on_region_selected
        self.overlay = None
        self.canvas = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None
    
    def start_selection(self):
        """Start the region selection process."""
        # Hide the main window
        self.parent_window.withdraw()
        
        # Wait a moment for the window to hide
        self.parent_window.update()
        time.sleep(0.3)
        
        # Create a fullscreen transparent overlay
        self.overlay = tk.Toplevel()
        self.overlay.attributes('-fullscreen', True)
        self.overlay.attributes('-alpha', 0.3)
        self.overlay.attributes('-topmost', True)
        self.overlay.configure(bg='black')
        
        # Create canvas for drawing selection rectangle
        self.canvas = tk.Canvas(self.overlay, highlightthickness=0, bg='black')
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Bind events
        self.overlay.bind('<Button-1>', self._on_button_press)
        self.overlay.bind('<B1-Motion>', self._on_mouse_drag)
        self.overlay.bind('<ButtonRelease-1>', self._on_button_release)
        self.overlay.bind('<Escape>', self._cancel_selection)
        
        # Instructions label
        instructions = tk.Label(
            self.overlay,
            text="Click and drag to select region. Press ESC to cancel.",
            bg='black',
            fg='white',
            font=('Arial', 14)
        )
        instructions.place(relx=0.5, rely=0.05, anchor=tk.CENTER)
    
    def _on_button_press(self, event):
        """Handle mouse button press to start selection."""
        self.start_x = event.x
        self.start_y = event.y
        
        # Create selection rectangle
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=2
        )
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag to update selection rectangle."""
        if self.rect_id is not None:
            self.canvas.coords(
                self.rect_id, 
                self.start_x, self.start_y, event.x, event.y
            )
    
    def _on_button_release(self, event):
        """Handle mouse button release to finalize selection."""
        end_x = event.x
        end_y = event.y
        
        # Calculate region (x, y, width, height)
        x = min(self.start_x, end_x)
        y = min(self.start_y, end_y)
        width = abs(end_x - self.start_x)
        height = abs(end_y - self.start_y)
        
        if width > self.MIN_REGION_SIZE and height > self.MIN_REGION_SIZE:
            # Valid selection
            self._finish_selection((x, y, width, height))
        else:
            # Invalid selection - too small
            messagebox.showwarning(
                "Invalid Selection",
                f"Please select a region with at least {self.MIN_REGION_SIZE}x{self.MIN_REGION_SIZE} pixels."
            )
            self._finish_selection(None)
    
    def _cancel_selection(self, event):
        """Cancel the selection process."""
        self._finish_selection(None)
    
    def _finish_selection(self, region):
        """Clean up and notify parent of selection result."""
        if self.overlay:
            self.overlay.destroy()
        self.parent_window.deiconify()
        
        if region is not None:
            self.on_region_selected(region)


# 2. Main Application Class
class OcrMonitorApp:
    def __init__(self, master):
        """
        Initializes the application, its state, and the GUI.
        'master' is the tkinter root window.
        """
        # --- State Variables ---
        self.master = master
        self.region = None  # Will store (x, y, width, height)
        self.region2 = None  # Second region (x, y, width, height)
        self.interval_ms = 1000  # Interval in milliseconds
        self.is_running = False
        self.previous_image_hash = None
        self.previous_image_hash2 = None
        self.after_id = None  # To store the ID of the scheduled job

        # --- OCR Engine ---
        # Initialize OCR reader once to avoid reloading the model
        # lang='ch' supports both Chinese and English
        self.ocr_reader = PaddleOCR(lang='ch', use_doc_orientation_classify=False, use_doc_unwarping=False, ocr_version="PP-OCRv4")

        # --- Text Hook ---
        self.text_handler_hook = self.default_text_handler

        # --- GUI Setup ---
        self.create_widgets()
        
        # --- Setup cleanup on window close ---
        self.master.protocol("WM_DELETE_WINDOW", self.cleanup)

    def create_widgets(self):
        """
        Creates and lays out all the GUI elements in the control panel.
        """
        # Main frame
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Region selection buttons
        region_frame = ttk.Frame(main_frame)
        region_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.select_region_btn = ttk.Button(
            region_frame, 
            text="Select Region 1", 
            command=self.select_region
        )
        self.select_region_btn.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        
        self.select_region2_btn = ttk.Button(
            region_frame, 
            text="Select Region 2", 
            command=self.select_region2
        )
        self.select_region2_btn.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        region_frame.columnconfigure(0, weight=1)
        region_frame.columnconfigure(1, weight=1)
        
        # Interval frame
        interval_frame = ttk.Frame(main_frame)
        interval_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(interval_frame, text="Interval (seconds):").grid(row=0, column=0, padx=5)
        self.interval_entry = ttk.Entry(interval_frame, width=10)
        self.interval_entry.insert(0, str(self.interval_ms / 1000))
        self.interval_entry.grid(row=0, column=1, padx=5)
        
        self.set_interval_btn = ttk.Button(
            interval_frame, 
            text="Set Interval", 
            command=self.set_interval
        )
        self.set_interval_btn.grid(row=0, column=2, padx=5)
        
        # Start/Stop button
        self.toggle_btn = ttk.Button(
            main_frame, 
            text="Start Monitoring", 
            command=self.toggle_monitoring
        )
        self.toggle_btn.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Status label
        self.status_label = ttk.Label(
            main_frame, 
            text="Status: Stopped | Region 1: Not selected | Region 2: Not selected",
            wraplength=300
        )
        self.status_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Configure grid weights
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

    def select_region(self):
        """
        Hides the main window, shows a transparent overlay for selection,
        and then stores the selected region in self.region.
        """
        def on_region_selected(region):
            """Callback when region selection is complete."""
            if region is not None:
                self.region = region
                self.previous_image_hash = None  # Reset hash when region changes
                self.update_status()
        
        selector = RegionSelector(self.master, on_region_selected)
        selector.start_selection()
    
    def select_region2(self):
        """
        Hides the main window, shows a transparent overlay for selection,
        and then stores the selected region in self.region2.
        """
        def on_region_selected(region):
            """Callback when region selection is complete."""
            if region is not None:
                self.region2 = region
                self.previous_image_hash2 = None  # Reset hash when region changes
                self.update_status()
        
        selector = RegionSelector(self.master, on_region_selected)
        selector.start_selection()

    def set_interval(self):
        """
        Reads the interval from the GUI's Entry widget, validates it,
        and updates self.interval_ms.
        """
        try:
            interval_seconds = float(self.interval_entry.get())
            if interval_seconds <= 0:
                raise ValueError("Interval must be positive")
            self.interval_ms = int(interval_seconds * 1000)
            self.update_status()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter a valid positive number.\n{str(e)}")

    def toggle_monitoring(self):
        """
        Called by the "Start/Stop" button.
        If not running, it calls start_monitoring().
        If running, it calls stop_monitoring().
        """
        if self.is_running:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        """
        Performs pre-flight checks (is at least one region selected?) and starts the
        monitoring loop using tkinter's `after()` method.
        """
        # 1. Check if at least one region is set. If not, show an error.
        if self.region is None and self.region2 is None:
            messagebox.showerror("Error", "Please select at least one region first!")
            return
        
        # 2. Set self.is_running = True.
        self.is_running = True
        
        # 3. Update GUI elements (button text, status label).
        self.toggle_btn.config(text="Stop Monitoring")
        self.select_region_btn.config(state='disabled')
        self.select_region2_btn.config(state='disabled')
        self.set_interval_btn.config(state='disabled')
        self.update_status()
        
        # 4. Call self.perform_check() to start the cycle immediately.
        self.perform_check()

    def stop_monitoring(self):
        """
        Stops the monitoring loop.
        """
        # 1. Set self.is_running = False.
        self.is_running = False
        
        # 2. If self.after_id is not None, cancel the scheduled job:
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
            self.after_id = None
        
        # 3. Update GUI elements.
        self.toggle_btn.config(text="Start Monitoring")
        self.select_region_btn.config(state='normal')
        self.select_region2_btn.config(state='normal')
        self.set_interval_btn.config(state='normal')
        self.update_status()

    def _process_region(self, region, previous_hash):
        """
        Process a single region: take screenshot, check hash, perform OCR if changed.
        
        Returns:
            tuple: (result_data, new_hash)
            - result_data: dict with 'rec_texts' and 'rec_polys' keys, or None if unchanged
            - new_hash: str - New hash value
        """
        if region is None:
            return None, previous_hash
        
        try:
            # Take screenshot
            x, y, width, height = region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            
            # Calculate hash
            img_array = np.array(screenshot)
            current_hash = hashlib.md5(img_array.tobytes()).hexdigest()
            
            # If unchanged, return None
            if current_hash == previous_hash:
                return None, previous_hash
            
            # If changed, perform OCR
            results = self.ocr_reader.predict(img_array)
            
            # Extract rec_texts and rec_polys from OCR results
            if results and results[0]:
                result_data = {
                    'rec_texts': results[0]['rec_texts'],
                    'rec_polys': results[0]['rec_polys']
                }
            else:
                result_data = {
                    'rec_texts': [],
                    'rec_polys': []
                }
            
            return result_data, current_hash
        
        except Exception as e:
            print(f"Error processing region: {e}")
            return None, previous_hash
    
    def perform_check(self):
        """
        This is the core OCR logic, executed on each interval.
        Monitors both regions and passes a list of outputs to the handler.
        """
        if not self.is_running:
            return
        
        # Process both regions
        result1, new_hash1 = self._process_region(self.region, self.previous_image_hash)
        result2, new_hash2 = self._process_region(self.region2, self.previous_image_hash2)
        
        # Update hashes
        if self.region is not None:
            self.previous_image_hash = new_hash1
        if self.region2 is not None:
            self.previous_image_hash2 = new_hash2
        
        # Create list of outputs (None if region didn't change, otherwise dict with rec_texts and rec_polys)
        outputs = [result1, result2]
        
        # Call handler with list of outputs (even if both are None)
        self.text_handler_hook(outputs)
        
        # Schedule next check
        if self.is_running:
            self.after_id = self.master.after(self.interval_ms, self.perform_check)

    def default_text_handler(self, outputs):
        """
        The default hook: prints list of outputs to the terminal.
        
        Args:
            outputs: List of dicts or None [region1_result, region2_result]
                    None if region didn't change
                    Dict with 'rec_texts' and 'rec_polys' keys if changed
        """
        for i, result in enumerate(outputs, 1):
            if result is not None:  # Only print if region changed
                texts = result['rec_texts']
                if texts:
                    text_output = '\n'.join(texts)
                    print(f"Region {i}:\n{text_output}\n")
    
    def update_status(self):
        """Updates the status label with current state."""
        status = "Running" if self.is_running else "Stopped"
        
        if self.region:
            x, y, w, h = self.region
            region1_str = f"({x}, {y}, {w}x{h})"
        else:
            region1_str = "Not selected"
        
        if self.region2:
            x, y, w, h = self.region2
            region2_str = f"({x}, {y}, {w}x{h})"
        else:
            region2_str = "Not selected"
        
        self.status_label.config(
            text=f"Status: {status} | R1: {region1_str} | R2: {region2_str} | Interval: {self.interval_ms/1000:.1f}s"
        )
    
    def cleanup(self):
        """
        Clean up resources before closing the application.
        Stops monitoring, cancels scheduled tasks, and closes the window.
        """
        # Stop monitoring if running
        if self.is_running:
            self.stop_monitoring()
        
        # Clean up OCR reader if needed (PaddleOCR doesn't have explicit cleanup,
        # but we can set it to None for garbage collection)
        if hasattr(self, 'ocr_reader'):
            self.ocr_reader = None
        
        # Destroy the window
        self.master.destroy()


# 3. Main Execution Block
if __name__ == "__main__":
    root = tk.Tk()
    root.title("OCR Monitor Control")
    root.geometry("350x200")
    root.resizable(False, False)

    app = OcrMonitorApp(root)

    # This starts the tkinter event loop. The window will appear,
    # and the program will now wait for user interaction.
    root.mainloop()