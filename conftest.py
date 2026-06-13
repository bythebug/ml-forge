import sys
from pathlib import Path

# Ensure the project root is on sys.path so all packages resolve correctly
# when running pytest from any working directory.
sys.path.insert(0, str(Path(__file__).parent))
