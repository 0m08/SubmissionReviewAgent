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
    
    def update(self, increment=1):
        self.completed_count += increment
        current_time = time.time()
        elapsed_seconds = current_time - self.start_time
        fraction_complete = self.completed_count / self.total_tasks
        
        # Estimate remaining time
        if self.completed_count > 0:
            seconds_per_task = elapsed_seconds / self.completed_count
            remaining_tasks = self.total_tasks - self.completed_count
            estimated_remaining_seconds = seconds_per_task * remaining_tasks
            
            elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
            remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_seconds)))
            
            self.progress_bar.progress(
                fraction_complete, 
                text=f"{self.description}: {int(fraction_complete * 100)}% | Elapsed: {elapsed_time_str} | Remaining: {remaining_time_str}"
            )
        else:
            self.progress_bar.progress(fraction_complete, text=f"{self.description}: {int(fraction_complete * 100)}% | Just started...")
    
    def should_save(self):
        return self.completed_count % self.save_interval == 0