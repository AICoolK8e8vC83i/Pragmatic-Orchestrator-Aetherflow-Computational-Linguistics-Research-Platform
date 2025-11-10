"""
Message History Storage: Store all messages with timestamps for scrolling
Stores in JSON format with timestamps for each message
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class MessageHistory:
    """Store all messages locally in JSON with timestamps."""
    
    def __init__(self, history_file: str = "message_history.json"):
        self.history_file = history_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure history file exists."""
        if not os.path.exists(self.history_file):
            with open(self.history_file, "w") as f:
                json.dump([], f)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to history."""
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,  # "user", "assistant", "system", "agent", etc.
            "content": content,
            "metadata": metadata or {}
        }
        
        # Load existing history
        try:
            with open(self.history_file, "r") as f:
                history = json.load(f)
        except:
            history = []
        
        # Append new message
        history.append(message)
        
        # Save back to file
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict]:
        """Get all messages, optionally limited."""
        try:
            with open(self.history_file, "r") as f:
                history = json.load(f)
            if limit:
                return history[-limit:]
            return history
        except:
            return []
    
    def clear_history(self):
        """Clear all history."""
        with open(self.history_file, "w") as f:
            json.dump([], f)

