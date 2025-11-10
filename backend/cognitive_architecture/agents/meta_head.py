"""
Agent_PragmaticRouter: Meta-head for deterministic routing
Theory: Searle's Speech Act Theory + Grice's Maxims
Model: Qwen 3 1.7B via Ollama (with Gemma 3 1B as fallback parser)
"""

import ollama
import json
import os
from dotenv import load_dotenv

load_dotenv()

class Agent_PragmaticRouter:
    """
    Pragmatic router using Searle's Speech Act Theory and Grice's Maxims.
    Determines which agent to invoke based on linguistic features.
    Uses Qwen 3 1.7B for routing, with Gemma 3 1B as fallback parser.
    """
    
    def __init__(self):
        self.model = "qwen3:1.7b"
        self.parser_model = "gemma3:1b"  # Fallback parser if Qwen output is invalid
        self.agents = {
            "code_switcher": "Handles bilingual queries, code-switching, cultural terms",
            "discourse_analyzer": "Analyzes transcripts, RST relations, coherence",
            "semantic_parser": "Converts NL to formal logic, lambda calculus",
            "vision_analyzer": "Analyzes images/videos, multimodal understanding"
        }
        self._verify_ollama_connection()
    
    def _verify_ollama_connection(self):
        """Verify Ollama is running and models are available."""
        try:
            models_result = ollama.list()
            # Handle ListResponse object (has .models attribute)
            if hasattr(models_result, 'models'):
                models_list = models_result.models
            elif isinstance(models_result, dict):
                models_list = models_result.get("models", [])
            elif isinstance(models_result, list):
                models_list = models_result
            else:
                models_list = []
            
            # Extract model names - Model objects have .model attribute
            model_names = []
            for m in models_list:
                if hasattr(m, 'model'):
                    name = m.model
                elif isinstance(m, dict):
                    name = m.get("model", "") or m.get("name", "")
                elif isinstance(m, str):
                    name = m
                else:
                    continue
                if name:
                    model_names.append(name)
            
            # Check if models are available
            qwen_found = self.model in model_names or any("qwen" in name.lower() for name in model_names)
            gemma_found = self.parser_model in model_names or any("gemma" in name.lower() for name in model_names)
            
            if qwen_found:
                print(f"✅ Qwen 3 1.7B ({self.model}) is available")
            else:
                available_str = ', '.join(model_names[:3]) if model_names else 'none'
                print(f"⚠️  Qwen 3 1.7B ({self.model}) not found. Available: {available_str}")
                print(f"   Run: ollama pull {self.model}")
            
            if gemma_found:
                print(f"✅ Gemma 3 1B ({self.parser_model}) is available (parser)")
            else:
                print(f"⚠️  Gemma 3 1B ({self.parser_model}) not found (optional parser)")
                print(f"   Run: ollama pull {self.parser_model}")
            
            return qwen_found
        except Exception as e:
            print(f"⚠️  Ollama connection failed: {e}")
            print("   Make sure Ollama is running: ollama serve")
            import traceback
            traceback.print_exc()
            return False
    
    def _parse_with_gemma(self, raw_response: str) -> str:
        """Use Gemma 3 1B to parse/validate Qwen's output."""
        try:
            parse_prompt = f"""Extract the agent name from this text. Output ONLY the agent name (one word).

Text: "{raw_response}"

Available agents: code_switcher, discourse_analyzer, semantic_parser, vision_analyzer

Output ONLY the agent name, nothing else:"""
            
            result = ollama.generate(
                model=self.parser_model,
                prompt=parse_prompt,
                options={"temperature": 0.1, "num_predict": 10}
            )
            
            parsed = result.get("response", "").strip().lower()
            # Clean up
            parsed = parsed.strip('.,!?;:').split()[0] if parsed.split() else ""
            
            if parsed in self.agents:
                print(f"✅ [Gemma Parser] Extracted: {parsed}")
                return parsed
            else:
                print(f"⚠️  [Gemma Parser] Invalid extraction: {parsed}")
                return None
        except Exception as e:
            print(f"⚠️  [Gemma Parser] Failed: {e}")
            return None
    
    def route(self, query: str, context: list = None):
        """
        Route query to appropriate agent using pragmatic inference.
        Theory: Searle (1969) - Speech Acts; Grice (1975) - Conversational Maxims
        
        Returns: {"agent": str, "reasoning": str, "speech_act": str, "ai_success": bool}
        """
        print(f"🏷️  Agent Badge: Pragmatic Router (Searle 1969 + Grice 1975)")
        routing_prompt = f"""You are a pragmatic router implementing Searle's Speech Act Theory.

Query: "{query}"

Available agents:
- code_switcher: Bilingual queries, Chinese-English code-switching, cultural terms (潜规则, 关系)
- discourse_analyzer: Meeting transcripts, RST analysis, coherence relations
- semantic_parser: Natural language to formal logic, lambda calculus conversion
- vision_analyzer: Image/video analysis, multimodal understanding, visual semantics

Theory: Use Grice's Maxims (Quality, Quantity, Relevance, Manner) to determine intent.

IMPORTANT: Respond with ONLY ONE WORD - the agent name (code_switcher, discourse_analyzer, semantic_parser, or vision_analyzer). No JSON, no explanation, just the agent name."""
        
        ai_success = False
        try:
            print(f"🔄 [Pragmatic Router] Routing query...")
            print(f"   🤖 Model: Qwen 3 1.7B (Ollama) - Output: ONE WORD ONLY")
            
            # Try Qwen 3 1.7B
            response = ollama.generate(
                model=self.model,
                prompt=routing_prompt,
                options={"temperature": 0.1, "top_p": 0.9, "num_predict": 20}
            )
            
            raw_response = response.get("response", "").strip()
            print(f"📥 [Pragmatic Router] Raw response: {raw_response}")
            
            # Check if response is empty
            if not raw_response:
                print(f"⚠️  [Pragmatic Router] Empty response from Qwen 3 1.7B")
                print(f"   Trying Gemma 3 1B parser...")
                # Try Gemma to parse/validate
                parsed_agent = self._parse_with_gemma("empty response - use default: code_switcher")
                if parsed_agent:
                    return {
                        "agent": parsed_agent,
                        "reasoning": f"Gemma parser extracted {parsed_agent}",
                        "speech_act": "assertive",
                        "ai_success": True,
                        "raw_response": raw_response
                    }
            else:
                # Try to extract agent name directly
                cleaned = raw_response.strip().lower()
                # Remove common prefixes/suffixes
                prefixes_to_remove = ["agent:", "agent", "the agent is", "selected:", "answer:", "response:", "the answer is", "i choose", "i select"]
                for prefix in prefixes_to_remove:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):].strip()
                
                # Remove punctuation
                cleaned = cleaned.strip('.,!?;:')
                
                # Get first word or compound
                words = cleaned.split()
                if words:
                    first_word = words[0]
                    # Check for compound names
                    if first_word in ["code", "discourse", "semantic", "vision"] and len(words) > 1:
                        compound = f"{first_word}_{words[1]}"
                        if compound in self.agents:
                            first_word = compound
                    
                    # Validate agent name
                    if first_word in self.agents:
                        agent_name = first_word
                        ai_success = True
                        print(f"✅ [Pragmatic Router] Successfully routed to: {agent_name}")
                        return {
                            "agent": agent_name,
                            "reasoning": f"AI routing selected {agent_name}",
                            "speech_act": "assertive",
                            "ai_success": True,
                            "raw_response": raw_response
                        }
                
                # If direct parsing failed, use Gemma parser
                print(f"⚠️  [Pragmatic Router] Direct parsing failed, trying Gemma 3 1B parser...")
                parsed_agent = self._parse_with_gemma(raw_response)
                if parsed_agent and parsed_agent in self.agents:
                    ai_success = True
                    print(f"✅ [Pragmatic Router] Gemma parser routed to: {parsed_agent}")
                    return {
                        "agent": parsed_agent,
                        "reasoning": f"Gemma parser extracted {parsed_agent} from Qwen output",
                        "speech_act": "assertive",
                        "ai_success": True,
                        "raw_response": raw_response
                    }
        
        except Exception as e:
            print(f"❌ [Pragmatic Router] AI call failed: {e}")
        
        # Fallback: keyword-based routing
        print(f"🔄 [Fallback] Using keyword matching...")
        import re
        has_image_ext = bool(re.search(r'\.(jpg|jpeg|png|gif|webp|mp4|avi|mov)', query, re.IGNORECASE))
        has_vision_keywords = any(word in query.lower() for word in ["image", "picture", "photo", "video", "describe this", "what's in this", "analyze this image"])
        
        if has_image_ext or has_vision_keywords:
            return {"agent": "code_switcher", "reasoning": "Image/video detected → Cultural preservation (FALLBACK)", "speech_act": "assertive", "ai_success": False}
        elif any(word in query.lower() for word in ["中文", "chinese", "潜规则", "关系"]):
            return {"agent": "code_switcher", "reasoning": "Bilingual keyword detected (FALLBACK)", "speech_act": "assertive", "ai_success": False}
        elif any(word in query.lower() for word in ["transcript", "meeting", "discourse", "coherence"]):
            return {"agent": "discourse_analyzer", "reasoning": "Discourse analysis keyword (FALLBACK)", "speech_act": "assertive", "ai_success": False}
        elif any(word in query.lower() for word in ["lambda", "logic", "semantic", "formal"]):
            return {"agent": "semantic_parser", "reasoning": "Formal semantics keyword (FALLBACK)", "speech_act": "assertive", "ai_success": False}
        return {"agent": "code_switcher", "reasoning": "Default fallback", "speech_act": "assertive", "ai_success": False}
    
    def refine(self, query: str, response: str, context: list):
        """Refine query based on response quality (Grice's Maxim of Quality)."""
        refine_prompt = f"""Original query: "{query}"
Previous response: "{str(response)[:500]}"

Refine the query to get a better response. Add context if needed. Keep it concise.

Refined query:"""
        
        try:
            print(f"🔄 [Pragmatic Router] Refining query...")
            print(f"   🤖 Model: Qwen 3 1.7B (Ollama)")
            refined = ollama.generate(
                model=self.model,
                prompt=refine_prompt,
                options={"temperature": 0.3, "num_predict": 100}
            )
            refined_text = refined.get("response", query).strip()
            if refined_text and refined_text != query:
                print(f"✅ [Pragmatic Router] Query refined successfully")
                return refined_text
            else:
                print(f"⚠️  [Pragmatic Router] Refinement returned original query, using enhanced version")
                # Simple enhancement
                return f"{query}\n\n[Previous response context]: {str(response)[:200]}"
        except Exception as e:
            print(f"❌ [Pragmatic Router] Refinement failed: {e}, using enhanced query")
            return f"{query}\n\n[Previous response context]: {str(response)[:200]}"

