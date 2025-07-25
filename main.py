import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))
os.environ['PYTHONPATH'] = str(src_dir)

if __name__ == "__main__":
    try:
        from backtester.ui.app import main
        main()
    
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        print(f"Src directory: {src_dir}")
        print("Please ensure you're running this from the Strategy-Backtester directory")
        sys.exit(1)