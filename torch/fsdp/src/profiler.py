"""
Memory Profiling Tool for CUDA Memory Visualization

This module provides a pluggable memory profiling tool that uses
torch.cuda.memory._record_memory_history and torch.cuda.memory._dump_snapshot
for CUDA memory visualization.

Usage:
    # 1. Using decorator (recommended for functions):
    profiler = MemoryProfiler(enabled=True, rank=0)
    profiler.start_recording()
    
    @profiler.profile_decorator("my_function")
    def my_function():
        # your code here
        pass
    
    # 2. Using context manager (recommended for code blocks):
    with profiler.profile("my_code_block"):
        # your code here
        pass
    
    # 3. Manual snapshot:
    profiler.dump_snapshot("custom_name.pickle")
"""

import torch
from functools import wraps
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime


@contextmanager
def null_context():
    """A null context manager that does nothing."""
    yield


class MemoryProfiler:
    """
    A pluggable memory profiling tool for CUDA memory visualization.
    Supports context managers and decorators for easy integration.
    """
    
    def __init__(self, 
                 output_dir: str = "memory_snapshots",
                 enabled: bool = True,
                 rank: int = 0,
                 enabled_ranks: Optional[list] = None,
                 dump_on_enter: bool = False,
                 dump_on_exit: bool = True):
        """
        Initialize memory profiler.
        
        Args:
            output_dir: Directory to save memory snapshots
            enabled: Whether profiling is enabled
            rank: Process rank
            enabled_ranks: List of ranks to dump snapshots. Default is [0] (only rank 0).
            dump_on_enter: Whether to dump snapshot when entering context/decorator
            dump_on_exit: Whether to dump snapshot when exiting context/decorator
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.rank = rank
        self.enabled_ranks = enabled_ranks if enabled_ranks is not None else [0]
        self.dump_on_enter = dump_on_enter
        self.dump_on_exit = dump_on_exit
        self._recording = False
        self._snapshot_counter = 0
        self._current_step = None
        
        if self.enabled and self.rank in self.enabled_ranks:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(self, step: Optional[int] = None, suffix: Optional[str] = None) -> str:
        """
        Generate filename in format: memory_dump_step{step}_rank{rank_id}_{%Y%m%d-%H%M}.pickle
        
        Args:
            step: Training step number. If None, uses current step or counter.
            suffix: Optional suffix to append before timestamp.
        
        Returns:
            Generated filename string.
        """
        if step is None:
            step = self._current_step if self._current_step is not None else self._snapshot_counter
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        
        parts = [f"memory_dump_step{step}_rank{self.rank}"]
        if suffix:
            parts.append(suffix)
        parts.append(timestamp)
        
        return "_".join(parts) + ".pickle"
    
    def start_recording(self):
        """Start recording CUDA memory history."""
        if self.enabled and self.rank in self.enabled_ranks and not self._recording:
            torch.cuda.memory._record_memory_history(
                max_entries=200000,
                context="all",
                stacks="all",
            )
            self._recording = True
            if self.rank in self.enabled_ranks:
                print(f'[MemoryProfiler] Rank {self.rank}: Started recording memory history')
    
    def stop_recording(self):
        """Stop recording CUDA memory history."""
        if self.enabled and self.rank in self.enabled_ranks and self._recording:
            torch.cuda.memory._record_memory_history(enabled=None)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self._recording = False
            if self.rank in self.enabled_ranks:
                print(f'[MemoryProfiler] Rank {self.rank}: Stopped recording memory history')
    
    def dump_snapshot(self, name: Optional[str] = None, step: Optional[int] = None) -> Optional[str]:
        """
        Dump a memory snapshot.
        
        Args:
            name: Optional name for the snapshot file. If None, auto-generated using step and timestamp.
            step: Training step number. Used for filename generation if name is None.
        
        Returns:
            Path to the snapshot file if dumped, None otherwise.
        """
        if not self.enabled or self.rank not in self.enabled_ranks:
            return None
        
        if name is None:
            self._snapshot_counter += 1
            name = self._generate_filename(step=step)
        
        snapshot_path = self.output_dir / name
        
        try:
            torch.cuda.synchronize()
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            if self.rank in self.enabled_ranks:
                print(f'[MemoryProfiler] Rank {self.rank}: Snapshot saved: {snapshot_path}')
            return str(snapshot_path)
        except Exception as e:
            if self.rank in self.enabled_ranks:
                print(f'[MemoryProfiler] Rank {self.rank}: Failed to dump snapshot: {e}')
            return None
    
    @contextmanager
    def profile(self, snapshot_name: Optional[str] = None, step: Optional[int] = None):
        """
        Context manager for profiling a code block.
        
        Usage:
            with profiler.profile("my_function", step=100):
                # code to profile
        
        Args:
            snapshot_name: Optional name suffix for the snapshot.
            step: Training step number for filename generation.
        """
        if self.dump_on_enter:
            if snapshot_name:
                name = self._generate_filename(step=step, suffix=f"{snapshot_name}_enter")
            else:
                name = self._generate_filename(step=step, suffix=None)
            self.dump_snapshot(name, step=step)
        
        try:
            yield self
        finally:
            if self.dump_on_exit:
                # Use current step if step was not provided
                exit_step = step if step is not None else self._current_step
                if snapshot_name:
                    name = self._generate_filename(step=exit_step, suffix=f"{snapshot_name}_exit")
                else:
                    name = self._generate_filename(step=exit_step, suffix=None)
                self.dump_snapshot(name, step=exit_step)
    
    def profile_decorator(self, snapshot_name: Optional[str] = None):
        """
        Decorator for profiling a function.
        
        Usage:
            @profiler.profile_decorator("train_step")
            def train_step(...):
                ...
        
        Note: The decorator will use the current step from profiler._current_step
        if available, otherwise uses auto-generated step counter.
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                step = self._current_step
                if self.dump_on_enter:
                    name = snapshot_name or func.__name__
                    filename = self._generate_filename(step=step, suffix=f"{name}_enter")
                    self.dump_snapshot(filename, step=step)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    if self.dump_on_exit:
                        name = snapshot_name or func.__name__
                        filename = self._generate_filename(step=step, suffix=f"{name}_exit")
                        self.dump_snapshot(filename, step=step)
            
            return wrapper
        return decorator
    
    def set_step(self, step: int):
        """Set the current training step for filename generation."""
        self._current_step = step
    
    def __enter__(self):
        """Context manager entry."""
        self.start_recording()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.dump_on_exit:
            name = self._generate_filename(step=self._current_step, suffix="final")
            self.dump_snapshot(name, step=self._current_step)
        self.stop_recording()
        return False
