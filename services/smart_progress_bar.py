import time
import datetime
import streamlit as st
from collections import deque
from statistics import mean


class SmartProgressBar:
    def __init__(self, total_tasks, description="Processing", save_interval=5, window_size=10):
        self.total_tasks = total_tasks
        self.completed_count = 0
        self.start_time = time.time()
        self.description = description
        self.save_interval = save_interval
        self.last_update_time = self.start_time
        self.task_times = deque(maxlen=window_size)  # Store recent task completion times
        self.progress_bar = st.progress(0, text=f"{description}: 0%")
        self.last_progress = 0
        if total_tasks == 0:
            self.progress_bar.empty()
    
    def _get_smoothed_estimate(self):
        """Calculate smoothed time estimate based on recent task completion times"""
        if not self.task_times:
            return None
        recent_avg = mean(self.task_times)
        remaining_tasks = self.total_tasks - self.completed_count
        return recent_avg * remaining_tasks
    
    def update(self, increment=1):
        current_time = time.time()
        time_since_last = current_time - self.last_update_time
        
        # Store the time taken for this batch of tasks
        if increment > 0:
            time_per_task = time_since_last / increment
            for _ in range(min(increment, 5)):  # Limit number of entries for large increments
                self.task_times.append(time_per_task)
        
        self.completed_count += increment
        elapsed_seconds = current_time - self.start_time
        fraction_complete = min(self.completed_count / self.total_tasks, 1.0)
        
        # Ensure we show some progress even for very quick tasks
        if fraction_complete > self.last_progress:
            self.last_progress = fraction_complete
            
            # Calculate time estimates
            elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
            
            if self.completed_count > 0:
                estimated_remaining = self._get_smoothed_estimate()
                if estimated_remaining is not None:
                    remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining)))
                    status_text = f"{self.description}: {int(fraction_complete * 100)}% | Elapsed: {elapsed_time_str} | Remaining: {remaining_time_str}"
                else:
                    status_text = f"{self.description}: {int(fraction_complete * 100)}% | Elapsed: {elapsed_time_str} | Calculating remaining time..."
            else:
                status_text = f"{self.description}: {int(fraction_complete * 100)}% | Just started..."
            
            self.progress_bar.progress(fraction_complete, text=status_text)
            
            # Close the bar once all tasks are done
            if self.completed_count >= self.total_tasks:
                final_text = f"{self.description}: Completed in {elapsed_time_str}"
                self.progress_bar.progress(1.0, text=final_text)
        
        self.last_update_time = current_time
    
    def should_save(self):
        return self.completed_count % self.save_interval == 0
