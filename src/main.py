"""Main entry point for Emotional Scanner."""
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
def main():
    """Main application entry point."""
    from src.utils.helpers import setup_logging
    logger = setup_logging()
    try:
        logger.info("Starting Emotional Scanner...")
        from src.services.video_processor import start
        start()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()