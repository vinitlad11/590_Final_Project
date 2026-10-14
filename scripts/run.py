"""Run the project without installing it; works in Windows PowerShell and Bash."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adaptive_qa.cli import main

if __name__ == "__main__":
    main()
