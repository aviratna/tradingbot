"""Task registry and graceful shutdown manager."""

import asyncio
import signal
import sys
from typing import Callable, Coroutine, Dict, Any
from .logger import get_logger

logger = get_logger(__name__)


class StreamManager:
    """Manages asyncio tasks for all streams with graceful shutdown."""

    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    def register(self, name: str, coro: Coroutine) -> None:
        """Register a coroutine to run as a named task."""
        task = asyncio.ensure_future(self._run_with_restart(name, coro))
        self._tasks[name] = task
        logger.info("stream_registered", name=name)

    async def _run_with_restart(self, name: str, coro: Coroutine) -> None:
        """Run a coroutine and restart it on unexpected failure."""
        while self._running:
            try:
                await coro
                logger.info("stream_completed", name=name)
                break  # Clean exit
            except asyncio.CancelledError:
                logger.info("stream_cancelled", name=name)
                break
            except Exception as e:
                logger.error("stream_crashed", name=name, error=str(e), exc_info=True)
                if self._running:
                    logger.info("stream_restarting", name=name, delay=5)
                    await asyncio.sleep(5)

    async def start_all(self) -> None:
        """Mark as running (tasks are already created via register)."""
        self._running = True
        logger.info("stream_manager_started", task_count=len(self._tasks))

    async def stop_all(self) -> None:
        """Cancel all running tasks gracefully."""
        self._running = False
        logger.info("stream_manager_stopping", task_count=len(self._tasks))
        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                logger.info("stream_cancelled", name=name)
        # Wait for all to finish
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        logger.info("stream_manager_stopped")

    async def wait_all(self) -> None:
        """Wait for all tasks to complete."""
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)


def setup_signal_handlers(manager: StreamManager, loop: asyncio.AbstractEventLoop) -> None:
    """Set up SIGINT/SIGTERM handlers for graceful shutdown."""
    def _handle_signal():
        logger.info("shutdown_signal_received")
        loop.create_task(manager.stop_all())

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal)
    else:
        # Windows doesn't support add_signal_handler well
        signal.signal(signal.SIGINT, lambda s, f: loop.create_task(manager.stop_all()))
