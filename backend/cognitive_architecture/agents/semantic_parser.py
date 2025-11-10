"""
Agent_SemanticParser: Natural language to formal logic
Theory: Montague Semantics (Montague, 1973)
Model: Claude Sonnet 4.5 via AWS Bedrock
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class Agent_SemanticParser:
    """
    Converts natural language to typed lambda calculus expressions.
    Theory: Montague Semantics (PTQ)
    """
    
    def __init__(self):
        # Setup exactly like test notebook
        self.bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AWS_BEARER_TOKEN")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        # Build endpoint exactly like test notebook
        self.endpoint = f"https://bedrock-runtime.{self.region}.amazonaws.com"
        self.system_prompt = """Convert natural language to typed lambda calculus expressions.

Theory:
- Montague (1973) - PTQ (Proper Treatment of Quantification)
- Heim & Kratzer (1998) - Semantics in Generative Grammar

Example: "Every student passed" → λP.∀x[student(x) → passed(x)]

Output: JSON with {"lambda_expression": "...", "type": "...", "explanation": "..."}"""
    
    def process(self, query: str, context: list = None):
        """Parse natural language to lambda calculus."""
        if not self.bearer_token:
            print(f"❌ [Claude Sonnet 4.5] AWS Bearer Token not found")
            return {"error": "AWS Bearer Token not found", "ai_success": False}
        
        print(f"🏷️  Agent Badge: Semantic Parser (Montague Semantics 1973)")
        # Build prompt exactly like test notebook
        prompt = f"{self.system_prompt}\n\nQuery: {query}"
        
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
                # Removed maxTokens to allow full semantic parsing
                "temperature": 0.2
            }
        }
        
        print(f"🔄 [Claude Sonnet 4.5] Parsing to lambda calculus...")
        print(f"   🤖 Model: Claude Sonnet 4.5 (AWS Bedrock)")
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Verify response structure
            if 'output' not in result or 'message' not in result['output']:
                print(f"❌ [Claude Sonnet 4.5] Invalid response structure")
                return {"error": "Invalid response structure", "lambda_expression": "", "ai_success": False}
            
            content = result['output']['message']['content'][0]['text']
            
            if not content:
                print(f"❌ [Claude Sonnet 4.5] Empty response content")
                return {"error": "Empty response", "lambda_expression": "", "ai_success": False}
            
            print(f"✅ [Claude Sonnet 4.5] AI call successful, response length: {len(content)} chars")
            
            # Strip markdown code blocks and extract JSON (handle extra text after JSON)
            content_cleaned = content.strip()
            
            # Remove markdown code blocks
            if content_cleaned.startswith("```json"):
                content_cleaned = content_cleaned[7:]
            elif content_cleaned.startswith("```"):
                content_cleaned = content_cleaned[3:]
            if content_cleaned.endswith("```"):
                content_cleaned = content_cleaned[:-3]
            content_cleaned = content_cleaned.strip()
            
            # Extract JSON block only (handle extra text after closing brace)
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
                        parsed = json.loads(json_only)
                        parsed["ai_success"] = True
                        print(f"✅ [Claude Sonnet 4.5] Successfully extracted and parsed JSON")
                        return parsed
                    else:
                        # Fallback: try parsing the whole cleaned content
                        parsed = json.loads(content_cleaned)
                        parsed["ai_success"] = True
                        return parsed
                else:
                    # No JSON found, try parsing anyway
                    parsed = json.loads(content_cleaned)
                    parsed["ai_success"] = True
                    return parsed
            except json.JSONDecodeError as je:
                print(f"⚠️  [Claude Sonnet 4.5] JSON parse failed: {je}, using plain text")
                return {
                    "lambda_expression": content_cleaned if content_cleaned else content,
                    "type": "e → t",
                    "explanation": "Parsed using Montague Semantics",
                    "ai_success": True
                }
        except requests.exceptions.HTTPError as e:
            print(f"❌ [Claude Sonnet 4.5] HTTP error: {e.response.status_code} - {e.response.text}")
            return {"error": f"HTTP {e.response.status_code}", "lambda_expression": "", "ai_success": False}
        except Exception as e:
            print(f"❌ [Claude Sonnet 4.5] AI call failed: {e}")
            return {"error": str(e), "lambda_expression": "", "ai_success": False}

