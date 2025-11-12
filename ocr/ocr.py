import tkinter as tk
from tkinter import messagebox, ttk
import time
import hashlib
import pyautogui
import easyocr
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
        self.interval_ms = 3000  # Interval in milliseconds
        self.is_running = False
        self.previous_image_hash = None
        self.after_id = None  # To store the ID of the scheduled job

        # --- OCR Engine ---
        # Initialize OCR reader once to avoid reloading the model
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'])

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
        
        # Select Region button
        self.select_region_btn = ttk.Button(
            main_frame, 
            text="Select Region", 
            command=self.select_region
        )
        self.select_region_btn.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
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
            text="Status: Stopped | Region: Not selected",
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
            messagebox.showinfo("Success", f"Interval set to {interval_seconds} seconds")
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
        Performs pre-flight checks (is a region selected?) and starts the
        monitoring loop using tkinter's `after()` method.
        """
        # 1. Check if self.region is set. If not, show an error.
        if self.region is None:
            messagebox.showerror("Error", "Please select a region first!")
            return
        
        # 2. Set self.is_running = True.
        self.is_running = True
        
        # 3. Update GUI elements (button text, status label).
        self.toggle_btn.config(text="Stop Monitoring")
        self.select_region_btn.config(state='disabled')
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
        self.set_interval_btn.config(state='normal')
        self.update_status()

    def perform_check(self):
        """
        This is the core OCR logic, executed on each interval.
        It replaces the old blocking `while` loop.
        """
        if not self.is_running:
            return
        
        try:
            # 1. Take screenshot of self.region.
            x, y, width, height = self.region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            
            # 2. Calculate the hash of the new screenshot efficiently.
            # Convert PIL Image to numpy array for both hashing and OCR
            img_array = np.array(screenshot)
            # Hash the numpy array directly
            current_hash = hashlib.md5(img_array.tobytes()).hexdigest()
            
            # 3. Compare with self.previous_image_hash. If same, do nothing.
            if current_hash == self.previous_image_hash:
                # Schedule next check
                if self.is_running:
                    self.after_id = self.master.after(self.interval_ms, self.perform_check)
                return
            
            # 4. If different:
            #    a. Update self.previous_image_hash.
            self.previous_image_hash = current_hash
            
            #    b. Perform OCR on the image.
            results = self.ocr_reader.readtext(img_array)
            
            # Extract text from OCR results
            extracted_text = '\n'.join([result[1] for result in results])
            
            #    c. If text is extracted, call self.text_handler_hook(text).
            if extracted_text.strip():
                self.text_handler_hook(extracted_text)
        
        except Exception as e:
            print(f"Error during OCR check: {e}")
        
        # 5. *** THE CRITICAL STEP ***
        #    If self.is_running is still True, schedule this same function
        #    to be called again after the specified interval.
        if self.is_running:
            self.after_id = self.master.after(self.interval_ms, self.perform_check)

    def default_text_handler(self, text):
        """The default hook: prints text to the terminal."""
        print(f"Timestamp: {time.ctime()}")
        print(f"Extracted Text:\n{text}")
    
    def update_status(self):
        """Updates the status label with current state."""
        status = "Running" if self.is_running else "Stopped"
        if self.region:
            x, y, w, h = self.region
            region_str = f"({x}, {y}, {w}x{h})"
        else:
            region_str = "Not selected"
        self.status_label.config(
            text=f"Status: {status} | Region: {region_str} | Interval: {self.interval_ms/1000:.1f}s"
        )
    
    def cleanup(self):
        """
        Clean up resources before closing the application.
        Stops monitoring, cancels scheduled tasks, and closes the window.
        """
        # Stop monitoring if running
        if self.is_running:
            self.stop_monitoring()
        
        # Clean up OCR reader if needed (EasyOCR doesn't have explicit cleanup,
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