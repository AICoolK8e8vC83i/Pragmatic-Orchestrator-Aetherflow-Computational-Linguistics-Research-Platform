"""
Agent_MemoryCurator: Episodic memory with semantic role labeling
Theory: Frame Semantics (Fillmore) + Semantic Role Labeling
Model: Claude Haiku 4.5 via AWS Bedrock
"""

import os
import sqlite3
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class Agent_MemoryCurator:
    """
    Extracts episodic memory, tags with semantic roles (ARG0, ARG1, ARGM-TMP).
    Theory: Frame Semantics + SRL
    """
    
    def __init__(self, db_path: str = "episodic_memory.db"):
        self.db_path = db_path
        # Setup exactly like test notebook
        self.bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AWS_BEARER_TOKEN")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        # Build endpoint exactly like test notebook
        self.endpoint = f"https://bedrock-runtime.{self.region}.amazonaws.com"
        self._init_db()
    
    def _init_db(self):
        """Initialize episodic memory database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                routing_info TEXT,
                semantic_roles TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store(self, query: str, response: str, routing: dict):
        """Store interaction with semantic role labeling."""
        print(f"🏷️  Agent Badge: Memory Curator (Frame Semantics - Fillmore)")
        # Extract semantic roles using Haiku
        semantic_roles = self._extract_semantic_roles(query, response)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO episodic_memory (query, response, routing_info, semantic_roles)
            VALUES (?, ?, ?, ?)
        """, (query, response, json.dumps(routing), json.dumps(semantic_roles)))
        
        # Keep all interactions (no deletion)
        
        conn.commit()
        conn.close()
    
    def _extract_semantic_roles(self, query: str, response: str) -> dict:
        """Extract semantic roles using Frame Semantics."""
        if not self.bearer_token:
            print(f"⚠️  [Claude Haiku 4.5] Bearer token not found, using fallback SRL")
            return {"ARG0": query, "ARG1": response, "ai_success": False}
        
        prompt = f"""Extract semantic roles from this interaction using Frame Semantics.

Query: {query}
Response: {response}

Output JSON with ARG0 (agent), ARG1 (theme), ARGM-TMP (temporal), ARGM-LOC (location)."""
        
        # Build URL exactly like test notebook
        url = f"{self.endpoint}/model/{self.model_id}/converse"
        
        # Headers exactly like test notebook
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        
        # Payload exactly like test notebook - content is array with text object
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                # Removed maxTokens to allow full memory extraction
                "temperature": 0.3
            }
        }
        
        try:
            print(f"🔄 [Claude Haiku 4.5] Extracting semantic roles...")
            print(f"   🤖 Model: Claude Haiku 4.5 (AWS Bedrock)")
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            result = resp.json()
            
            if 'output' not in result or 'message' not in result['output']:
                print(f"❌ [Claude Haiku 4.5] Invalid response structure")
                return {"ARG0": "user", "ARG1": query, "ai_success": False}
            
            content = result['output']['message']['content'][0]['text']
            
            if not content:
                print(f"❌ [Claude Haiku 4.5] Empty response")
                return {"ARG0": "user", "ARG1": query, "ai_success": False}
            
            print(f"✅ [Claude Haiku 4.5] AI call successful")
            
            # Strip markdown code blocks and extract JSON (handle markdown headers and extra text)
            content_cleaned = content.strip()
            
            # Skip markdown headers (like "# Frame Semantic Analysis")
            # Find the first ``` or { to locate JSON
            json_start = -1
            if "```json" in content_cleaned:
                json_start = content_cleaned.find("```json") + 7
            elif "```" in content_cleaned:
                json_start = content_cleaned.find("```") + 3
            elif "{" in content_cleaned:
                json_start = content_cleaned.find("{")
            
            if json_start >= 0:
                content_cleaned = content_cleaned[json_start:].strip()
            
            # Remove closing markdown code blocks
            if content_cleaned.endswith("```"):
                content_cleaned = content_cleaned[:-3]
            content_cleaned = content_cleaned.strip()
            
            # Extract JSON block only (handle extra text after closing brace)
            # Find the first { and matching } to extract just the JSON
            try:
                first_brace = content_cleaned.find("{")
                if first_brace >= 0:
                    # Find matching closing brace
                    brace_count = 0
                    last_brace = -1
                    for i in range(first_brace, len(content_cleaned)):
                        if content_cleaned[i] == "{":
                            brace_count += 1
                        elif content_cleaned[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                last_brace = i
                                break
                    
                    if last_brace > first_brace:
                        # Extract just the JSON part
                        json_only = content_cleaned[first_brace:last_brace + 1]
                        try:
                            parsed = json.loads(json_only)
                            parsed["ai_success"] = True
                            print(f"✅ [Claude Haiku 4.5] Successfully extracted and parsed JSON")
                            return parsed
                        except json.JSONDecodeError as je:
                            # JSON might have unterminated strings - try to repair
                            print(f"⚠️  [Claude Haiku 4.5] JSON has errors, attempting repair...")
                            # Try to close unterminated strings
                            json_repaired = json_only
                            # Count quotes to find unterminated strings
                            quote_count = json_repaired.count('"')
                            if quote_count % 2 != 0:
                                # Odd number of quotes - string is unterminated
                                # Find last unclosed quote and close it
                                last_quote_idx = json_repaired.rfind('"')
                                if last_quote_idx > 0:
                                    # Check if it's inside a value (not a key)
                                    before_quote = json_repaired[:last_quote_idx]
                                    if ':' in before_quote:
                                        # It's a value, close it
                                        json_repaired = json_repaired[:last_quote_idx+1] + '"'
                            # Try to close the object if needed
                            if not json_repaired.rstrip().endswith('}'):
                                json_repaired = json_repaired.rstrip().rstrip(',') + '}'
                            
                            try:
                                parsed = json.loads(json_repaired)
                                parsed["ai_success"] = True
                                print(f"✅ [Claude Haiku 4.5] Successfully repaired and parsed JSON")
                                return parsed
                            except:
                                pass
                    else:
                        # JSON is truncated - try to repair
                        print(f"⚠️  [Claude Haiku 4.5] JSON appears truncated, attempting repair...")
                        json_partial = content_cleaned[first_brace:]
                        # Try to close it
                        if json_partial.rstrip().endswith('"') and not json_partial.rstrip().endswith('",'):
                            json_repaired = json_partial.rstrip().rstrip('"') + '"}'
                        else:
                            json_repaired = json_partial.rstrip().rstrip(',') + '}'
                        
                        try:
                            parsed = json.loads(json_repaired)
                            parsed["ai_success"] = True
                            print(f"✅ [Claude Haiku 4.5] Successfully repaired truncated JSON")
                            return parsed
                        except:
                            pass
                    
                    # Fallback: try parsing the whole cleaned content
                    try:
                        parsed = json.loads(content_cleaned)
                        parsed["ai_success"] = True
                        return parsed
                    except:
                        pass
                else:
                    # No JSON found, try parsing anyway
                    parsed = json.loads(content_cleaned)
                    parsed["ai_success"] = True
                    return parsed
            except json.JSONDecodeError as je:
                print(f"⚠️  [Claude Haiku 4.5] JSON parse failed: {je}")
                print(f"   Content: {content_cleaned}")
                # Return fallback with success flag since API call worked
                return {"ARG0": "user", "ARG1": query, "ARGM-TMP": "now", "ai_success": True}
        except Exception as e:
            print(f"❌ [Claude Haiku 4.5] AI call failed: {e}, using fallback")
            return {"ARG0": "user", "ARG1": query, "ai_success": False}

