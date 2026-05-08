#!/usr/bin/env python3
"""Simple entry point script to avoid circular imports."""

import sys
import os
from pathlib import Path

# Add src to path and change working directory
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
os.chdir(project_root)

def main():
    """Main entry point."""
    try:
        # Import and start the application
        import services.video_processor
        services.video_processor.start()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
