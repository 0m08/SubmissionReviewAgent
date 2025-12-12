"""
Async event loop management for Streamlit applications.

Provides a persistent event loop that solves the issue of multiple asyncio.run()
calls creating incompatible event loops for long-lived async resources.

This is specifically designed for Streamlit apps that need to maintain async
connections (like MCP servers) across multiple user interactions.
"""

import asyncio
import logging
import threading
from typing import Any, Coroutine

logger = logging.getLogger(__name__)


class StreamlitAsyncRunner:
    """
    Persistent event loop for Streamlit sessions.

    Solves the issue of multiple asyncio.run() calls creating
    incompatible event loops for long-lived async resources like
    MCP server connections.

    Usage:
        # In session state initialization:
        runner = StreamlitAsyncRunner()
        runner.start()

        # To run async functions:
        result = runner.run_coroutine(my_async_function())

        # On cleanup:
        runner.stop()

    Why This Is Needed:
        Streamlit apps are synchronous by default, but async operations require
        an event loop. Calling asyncio.run() multiple times creates separate
        event loops, which makes async resources (connections, streams, tasks)
        created in one loop invalid in another loop.

        This class maintains a single persistent event loop in a background
        thread, allowing all async operations to share the same loop.
    """

    def __init__(self):
        """Initialize the async runner (doesn't start the loop yet)."""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running: bool = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Check if the event loop is currently running."""
        return self._running

    def start(self) -> None:
        """
        Start the persistent event loop in a background thread.

        Safe to call multiple times - will do nothing if already started.
        """
        with self._lock:
            if self._running:
                logger.debug("StreamlitAsyncRunner already running")
                return

            logger.info("Starting StreamlitAsyncRunner...")

            # Create a new event loop
            self._loop = asyncio.new_event_loop()

            # Start the loop in a daemon thread
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="StreamlitAsyncLoop"
            )
            self._thread.start()

            # Wait a bit to ensure loop is running
            import time
            time.sleep(0.1)

            self._running = True
            logger.info("StreamlitAsyncRunner started successfully")

    def _run_loop(self) -> None:
        """
        Run the event loop forever in the background thread.

        This method runs in the background thread and should not be called directly.
        """
        try:
            asyncio.set_event_loop(self._loop)
            logger.debug("Event loop thread started, running forever...")
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Error in event loop thread: {e}")
        finally:
            logger.debug("Event loop thread terminated")

    def run_coroutine(self, coro: Coroutine) -> Any:
        """
        Execute a coroutine in the persistent event loop.

        This method blocks until the coroutine completes and returns its result.

        Args:
            coro: The coroutine to execute

        Returns:
            The result of the coroutine

        Raises:
            RuntimeError: If the event loop is not running
            Exception: Any exception raised by the coroutine

        Example:
            async def fetch_data():
                return {"data": "example"}

            runner = StreamlitAsyncRunner()
            runner.start()
            result = runner.run_coroutine(fetch_data())
            print(result)  # {"data": "example"}
        """
        if not self._running:
            raise RuntimeError(
                "Event loop is not running. Call start() first."
            )

        if self._loop is None:
            raise RuntimeError("Event loop is not initialized")

        # Submit the coroutine to the event loop and wait for result
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        try:
            # Block until the coroutine completes
            result = future.result()
            return result
        except Exception as e:
            logger.error(f"Error executing coroutine: {e}")
            raise

    def stop(self) -> None:
        """
        Stop the event loop and clean up resources.

        Safe to call multiple times. Blocks until the thread terminates
        (with a timeout).
        """
        with self._lock:
            if not self._running:
                logger.debug("StreamlitAsyncRunner already stopped")
                return

            logger.info("Stopping StreamlitAsyncRunner...")

            if self._loop and self._running:
                # Stop the event loop
                self._loop.call_soon_threadsafe(self._loop.stop)

                # Wait for the thread to finish
                if self._thread:
                    self._thread.join(timeout=5.0)

                    if self._thread.is_alive():
                        logger.warning(
                            "Event loop thread did not stop within timeout"
                        )

                # Close the loop
                try:
                    self._loop.close()
                except Exception as e:
                    logger.warning(f"Error closing event loop: {e}")

            self._loop = None
            self._thread = None
            self._running = False

            logger.info("StreamlitAsyncRunner stopped")

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "running" if self._running else "stopped"
        return f"StreamlitAsyncRunner(status={status})"

    def __del__(self):
        """Cleanup when object is garbage collected."""
        try:
            self.stop()
        except:
            pass
