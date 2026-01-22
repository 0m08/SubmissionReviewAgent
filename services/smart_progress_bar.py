import time
import datetime
import streamlit as st


class SmartProgressBar:
    def __init__(self, total_tasks, description="Processing", save_interval=5):
        self.total_tasks = total_tasks
        self.completed_count = 0
        self.start_time = time.time()
        self.description = description
        self.save_interval = save_interval
        self.progress_bar = st.progress(0, text=f"{description}: 0%")
        if total_tasks == 0:
            self.progress_bar.empty()
    
    def update(self, increment=1):
        if self.total_tasks == 0:
            # Avoid division by zero when there are no tasks
            return

        self.completed_count += increment
        
        # Clamp completed_count to prevent exceeding total_tasks
        if self.completed_count > self.total_tasks:
            self.completed_count = self.total_tasks
        
        current_time = time.time()
        elapsed_seconds = current_time - self.start_time
        fraction_complete = self.completed_count / self.total_tasks
        
        # Clamp fraction to valid range
        fraction_complete = max(0.0, min(1.0, fraction_complete))
        
        # Estimate remaining time
        if self.completed_count > 0:
            seconds_per_task = elapsed_seconds / self.completed_count
            remaining_tasks = self.total_tasks - self.completed_count
            estimated_remaining_seconds = max(0, seconds_per_task * remaining_tasks)
            
            elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
            remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_seconds)))
            
            self.progress_bar.progress(
                fraction_complete,
                text=f"{self.description}: {int(fraction_complete * 100)}% | Elapsed: {elapsed_time_str} | Remaining: {remaining_time_str}"
            )

            # Close the bar once all tasks are done
            if self.completed_count >= self.total_tasks:
                self.progress_bar.empty()

        else:
            self.progress_bar.progress(fraction_complete, text=f"{self.description}: {int(fraction_complete * 100)}% | Just started...")
    
    def update_progress(self, fraction_complete: float, custom_text: str = None):
        """
        Update progress bar with a specific fraction (0.0 to 1.0).
        
        Args:
            fraction_complete: Progress fraction (0.0 to 1.0)
            custom_text: Optional custom text to display
        """
        if self.total_tasks == 0:
            return
        
        # Clamp fraction to valid range
        fraction_complete = max(0.0, min(1.0, fraction_complete))
        
        current_time = time.time()
        elapsed_seconds = current_time - self.start_time
        
        # Calculate equivalent completed count for ETA calculation
        equivalent_completed = fraction_complete * self.total_tasks
        
        # Estimate remaining time based on current progress
        if equivalent_completed > 0:
            seconds_per_task = elapsed_seconds / equivalent_completed
            remaining_tasks = self.total_tasks - equivalent_completed
            estimated_remaining_seconds = seconds_per_task * remaining_tasks
            
            elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
            remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_seconds)))
            
            if custom_text:
                display_text = custom_text
            else:
                display_text = f"{self.description}: {int(fraction_complete * 100)}% | Elapsed: {elapsed_time_str} | Remaining: {remaining_time_str}"
            
            self.progress_bar.progress(fraction_complete, text=display_text)
        else:
            if custom_text:
                display_text = custom_text
            else:
                display_text = f"{self.description}: {int(fraction_complete * 100)}% | Just started..."
            self.progress_bar.progress(fraction_complete, text=display_text)
    
    def should_save(self):
        if self.save_interval <= 0:
            return False
        return self.completed_count % self.save_interval == 0