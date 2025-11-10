"""
Agent_CodeSwitcher: Bilingual Chinese-English code-switching
Theory: Myers-Scotton (1993) Matrix Language Frame + Sperber & Wilson (1986) Relevance Theory
Model: Kimi K2 Thinking
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Test the API key first
api_key = os.getenv("MOONSHOT_API_KEY")
if not api_key:
    raise ValueError("MOONSHOT_API_KEY not found in environment variables")
print(f"✅ API key found for Kimi K2")

class Agent_CodeSwitcher:
    """
    Handles mixed Chinese-English queries, preserves culturally-loaded terms.
    Theory: Code-switching linguistics + Relevance Theory
    """
    
    def __init__(self):
        # Initialize exactly like test notebook
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("MOONSHOT_API_KEY not found in environment variables")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1"
        )
        self.model = "kimi-k2-thinking"
        self.system_prompt = """You are a bilingual AI assistant specialized in Chinese-English code-switching discourse.

Task: Preserve culturally-loaded Chinese terms (潜规则, 关系, 面子) in English translation.

Theory:
- Myers-Scotton (1993) - Matrix Language Frame Model
- Sperber & Wilson (1986) - Relevance Theory

Input: Mixed-language query
Output: JSON with {"english_response": "...", "cultural_notes": "..."}

Preserve cultural context and explain untranslatable concepts."""
    
    def process(self, query: str, context: list = None):
        """Process bilingual query with cultural preservation."""
        print(f"🏷️  Agent Badge: Code Switcher (Myers-Scotton 1993 + Relevance Theory)")
        # Build query with system prompt inline (like test notebook pattern)
        full_query = f"{self.system_prompt}\n\nQuery: {query}"
        
        # Use simple messages array like test notebook - NO system role, just user
        messages = [{"role": "user", "content": full_query}]
        
        print(f"🔄 [Kimi K2 Thinking] Processing bilingual query...")
        print(f"   🤖 Model: Kimi K2 Thinking (Moonshot AI)")
        print(f"   Query length: {len(full_query)} chars")
        
        try:
            # Call exactly like test notebook - NO timeout parameter
            # Remove max_tokens to allow full response (Kimi K2 can handle it)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
                # max_tokens removed - let Kimi K2 generate full response
            )
            
            # Debug: Print full response structure
            print(f"📥 [Kimi K2] Response type: {type(response)}")
            print(f"📥 [Kimi K2] Response attributes: {dir(response)[:10]}")
            
            # Verify response structure
            if not response:
                print(f"❌ [Kimi K2] Response is None")
                return {"error": "Response is None", "english_response": "", "cultural_notes": "", "ai_success": False}
            
            if not hasattr(response, 'choices') or not response.choices:
                print(f"❌ [Kimi K2] No choices in response")
                print(f"   Response: {response}")
                return {"error": "No choices in response", "english_response": "", "cultural_notes": "", "ai_success": False}
            
            # Access content EXACTLY like test notebook - simple direct access
            message = response.choices[0].message
            content = message.content
            
            # Check finish_reason for debugging
            finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None
            if finish_reason:
                print(f"📊 [Kimi K2] Finish reason: {finish_reason}")
                if finish_reason == "length":
                    print(f"   ⚠️  Response was truncated (max_tokens reached), but content exists")
            
            # If content is None or empty, that's an error
            # BUT: finish_reason: length means truncated, NOT empty - use the truncated content!
            if content is None:
                print(f"❌ [Kimi K2] Content is None")
                print(f"   Finish reason: {finish_reason}")
                return {"error": "Content is None", "english_response": "", "cultural_notes": "", "ai_success": False}
            
            # Empty string is also an error (but length means truncated, which is OK)
            if content == "":
                print(f"❌ [Kimi K2] Content is empty string")
                print(f"   Finish reason: {finish_reason}")
                return {"error": "Empty response", "english_response": "", "cultural_notes": "", "ai_success": False}
            
            print(f"✅ [Kimi K2] AI call successful, response length: {len(content)} chars")
            print(f"   Content: {content}")
            
            # Strip markdown code blocks and extract JSON (handle truncated JSON)
            content_cleaned = content.strip()
            
            # Remove markdown code blocks
            if content_cleaned.startswith("```json"):
                content_cleaned = content_cleaned[7:]
            elif content_cleaned.startswith("```"):
                content_cleaned = content_cleaned[3:]
            if content_cleaned.endswith("```"):
                content_cleaned = content_cleaned[:-3]
            content_cleaned = content_cleaned.strip()
            
            # Try to extract and parse JSON (handle truncated JSON)
            try:
                # Find JSON block (first { to matching })
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
                        print(f"✅ [Kimi K2] Successfully extracted and parsed JSON")
                        return parsed
                    else:
                        # JSON is truncated - try to repair it
                        print(f"⚠️  [Kimi K2] JSON appears truncated, attempting repair...")
                        # Try to close unclosed strings/objects
                        json_partial = content_cleaned[first_brace:]
                        # Remove trailing incomplete string if present
                        if json_partial.rstrip().endswith('"') and not json_partial.rstrip().endswith('",'):
                            # Try to close it
                            json_repaired = json_partial.rstrip().rstrip('"') + '"}'
                        else:
                            json_repaired = json_partial + '}'
                        
                        try:
                            parsed = json.loads(json_repaired)
                            parsed["ai_success"] = True
                            print(f"✅ [Kimi K2] Successfully repaired and parsed truncated JSON")
                            return parsed
                        except:
                            # Fallback: extract what we can
                            pass
                
                # Fallback: try parsing the whole cleaned content
                parsed = json.loads(content_cleaned)
                parsed["ai_success"] = True
                return parsed
            except json.JSONDecodeError as je:
                print(f"⚠️  [Kimi K2] JSON parse failed: {je}, using plain text")
                print(f"   Cleaned content: {content_cleaned}")
                # Extract english_response and cultural_notes from plain text if possible
                english_match = None
                cultural_match = None
                if '"english_response"' in content_cleaned:
                    # Try to extract even from truncated JSON
                    import re
                    eng_match = re.search(r'"english_response"\s*:\s*"([^"]*)', content_cleaned)
                    if eng_match:
                        english_match = eng_match.group(1)
                if '"cultural_notes"' in content_cleaned:
                    cult_match = re.search(r'"cultural_notes"\s*:\s*"([^"]*)', content_cleaned)
                    if cult_match:
                        cultural_match = cult_match.group(1)
                
                return {
                    "english_response": english_match if english_match else (content_cleaned if content_cleaned else content),
                    "cultural_notes": cultural_match if cultural_match else "Response generated using Relevance Theory",
                    "ai_success": True,
                    "raw_response": content
                }
        except Exception as e:
            print(f"❌ [Kimi K2] AI call failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "english_response": "", "cultural_notes": "", "ai_success": False}

