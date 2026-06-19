import signal
import threading
import time

from server.deploy import deploy

from loguru import logger
from ray import serve
import ray


shutdown_event = threading.Event()


def _request_shutdown(signum, _frame) -> None:
    logger.info(f"Received signal {signum}; requesting graceful shutdown...")
    shutdown_event.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    deploy()

    try:
        while not shutdown_event.wait(1):
            time.sleep(0.1)
    finally:
        logger.info("Shutting down…")
        try:
            serve.shutdown()
        except Exception:
            logger.exception("serve.shutdown() failed")

        # Give Serve a brief window to flush replica teardown.
        time.sleep(2)

        try:
            ray.shutdown()
        except Exception:
            logger.exception("ray.shutdown() failed")