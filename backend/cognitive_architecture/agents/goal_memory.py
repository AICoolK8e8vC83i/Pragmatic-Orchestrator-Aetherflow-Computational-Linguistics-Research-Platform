"""
Goal Memory: Simple goal context management
Theory: Episodic memory + goal-directed behavior
Storage: Simple text file (user_profile.md) + SQLite for history
"""

import os
import sqlite3
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class GoalMemory:
    """
    Manages user goals and context.
    Simple implementation: text file + SQLite history
    """
    
    def __init__(self, profile_path: str = "context/user_profile.md", db_path: str = "goal_history.db"):
        self.profile_path = profile_path
        self.db_path = db_path
        self._ensure_profile_exists()
        self._init_db()
    
    def _ensure_profile_exists(self):
        """Ensure user profile file exists."""
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        if not os.path.exists(self.profile_path):
            with open(self.profile_path, "w") as f:
                f.write("# User Profile\n\n**Current Goal**: \n**Focus Area**: \n**Language Preference**: \n")
    
    def _init_db(self):
        """Initialize goal history database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS goal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_current_goal(self) -> str:
        """Read current goal from profile file."""
        try:
            with open(self.profile_path, "r") as f:
                content = f.read()
                # Extract goal line
                for line in content.split("\n"):
                    if "**Current Goal**" in line:
                        goal = line.split(":", 1)[1].strip() if ":" in line else ""
                        return goal.strip()
        except:
            pass
        return ""
    
    def set_goal(self, goal: str):
        """Set current goal in profile file and store in history."""
        try:
            with open(self.profile_path, "r") as f:
                lines = f.readlines()
            
            # Update goal line
            updated = False
            for i, line in enumerate(lines):
                if "**Current Goal**" in line:
                    lines[i] = f"**Current Goal**: {goal}\n"
                    updated = True
                    break
            
            if not updated:
                # Add if not found
                lines.append(f"**Current Goal**: {goal}\n")
            
            with open(self.profile_path, "w") as f:
                f.writelines(lines)
            
            # Store in history
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO goal_history (goal) VALUES (?)", (goal,))
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error setting goal: {e}")
    
    def get_goal_context(self) -> str:
        """Get formatted goal context for prompts."""
        goal = self.get_current_goal()
        if goal:
            return f"User goal: {goal}"
        return ""

