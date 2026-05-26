import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

class VersioningManager:
    """
    Manages a manifest of data files across different pipeline stages.
    Ensures deterministic file discovery and historical tracking.
    """
    
    DEFAULT_MANIFEST_PATH = Path(__file__).parent.parent.parent / "manifest.json"

    def __init__(self, manifest_path: Optional[str] = None):
        self.manifest_path = Path(manifest_path) if manifest_path else self.DEFAULT_MANIFEST_PATH
        self.data = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "project": "VaccineNLP-Thesis",
            "last_updated": datetime.now().isoformat(),
            "stages": {
                "collection": [],
                "preprocessing": [],
                "annotation": [],
                "model": [],
                "evaluation": []
            }
        }

    def _save_manifest(self):
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def register_entry(self, stage: str, filepath: str, version: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Registers a new file entry in the manifest.
        """
        if stage not in self.data["stages"]:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {list(self.data['stages'].keys())}")
        
        # Absolute path normalization for consistency
        abs_path = os.path.abspath(filepath)
        
        entry = {
            "version": version,
            "path": abs_path,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.data["stages"][stage].append(entry)
        self._save_manifest()
        return entry

    def get_latest_entry(self, stage: str) -> Optional[Dict]:
        """
        Returns the most recent entry for a given stage.
        """
        entries = self.data["stages"].get(stage, [])
        if not entries:
            return None
        # Sort by timestamp (ISO format sorts correctly)
        return sorted(entries, key=lambda x: x["timestamp"])[-1]

    def get_version(self, stage: str, version: str) -> Optional[Dict]:
        """
        Returns a specific version for a stage.
        """
        entries = self.data["stages"].get(stage, [])
        for entry in entries:
            if entry["version"] == version:
                return entry
        return None

    def get_path(self, key: str) -> Optional[str]:
        """
        Retrieves a static path from the 'paths' section of the manifest.
        Falls back to stages if not found (legacy behavior).
        """
        # 1. Check in 'paths' section
        path = self.data.get("paths", {}).get(key)
        if path:
            # Normalize to absolute path if it's relative to project root
            if not os.path.isabs(path):
                return str(self.manifest_path.parent / path)
            return path
        
        # 2. Legacy fallback: check stage entries (optional, but good for safety)
        return None

if __name__ == "__main__":
    # Quick test/initialization
    vm = VersioningManager()
    print(f"Initialized manifest at: {vm.manifest_path}")
    print(f"Current stages: {list(vm.data['stages'].keys())}")
