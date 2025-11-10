"""
Agent_RLHF_Critic: Evaluates responses using linguistic features
Theory: Preference learning + linguistic annotation standards
Storage: SQLite with Penn Treebank schema
Uses Qwen 3 1.7B (RLAIF) for linguistic analysis
"""

import sqlite3
import json
import os
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class Agent_RLHF_Critic:
    """
    Evaluates responses using linguistic features, stores in SQLite.
    Theory: RLAIF (Reinforcement Learning from AI Feedback) + linguistic annotation standards
    Uses Qwen 3 1.7B for RLAIF analysis
    """
    
    def __init__(self, db_path: str = "rlhf_annotations.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with Penn Treebank-inspired schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rlhf_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                overall_rating FLOAT,
                pragmatic_coherence FLOAT,
                cultural_appropriateness FLOAT,
                syntactic_clarity FLOAT,
                speech_act_alignment FLOAT,
                syntactic_complexity INT,
                speech_act_type TEXT,
                goal_context TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def evaluate(self, response: Dict, query: str, overall_rating: Optional[float] = None, goal_context: str = "") -> Dict:
        """
        Evaluate response using linguistic features.
        Uses top 2 rated examples as feedback for learning.
        
        Args:
            response: Agent response dictionary
            query: Original query
            overall_rating: User-provided rating (1-10) from Streamlit dashboard
            goal_context: Current goal/memory context
        
        Returns: {"sufficient": bool, "scores": {...}}
        """
        print(f"🏷️  Agent Badge: RLAIF Critic (Preference Learning)")
        response_text = response.get("english_response") or response.get("analysis") or str(response)
        
        # Get top 2 rated examples for feedback
        top_examples = self._get_top_rated_examples(limit=2)
        feedback_context = ""
        if top_examples:
            feedback_context = "\n\nTop rated examples for reference:\n"
            for i, ex in enumerate(top_examples, 1):
                feedback_context += f"\nExample {i} (Rating: {ex['overall_rating']:.1f}/10):\n"
                feedback_context += f"Query: {ex['query']}\n"
                feedback_context += f"Response: {ex['response']}\n"
                feedback_context += f"Scores: Pragmatic={ex['pragmatic_coherence']:.2f}, Cultural={ex['cultural_appropriateness']:.2f}, Clarity={ex['syntactic_clarity']:.2f}\n"
        
        # Use Qwen 3 1.7B (RLAIF) for linguistic analysis if available
        qwen_available = self._qwen_available()
        if qwen_available:
            print(f"🔄 [RLAIF] Analyzing linguistic features...")
            print(f"   🤖 Model: Qwen 3 1.7B (Ollama) - RLAIF")
            if top_examples:
                print(f"   📚 Using top {len(top_examples)} rated examples as feedback")
            analysis = self._analyze_with_qwen(query, response_text, feedback_context)
            if analysis:
                print(f"✅ [RLAIF] Analysis successful")
            else:
                print(f"⚠️  [RLAIF] Analysis failed, using heuristics")
        else:
            print(f"⚠️  [RLAIF] Not available, using heuristic scoring")
            analysis = {}
        
        # Extract scores from Qwen analysis or use heuristics
        pragmatic_coherence = analysis.get("pragmatic_coherence") or self._score_pragmatic_coherence(response_text)
        cultural_appropriateness = analysis.get("cultural_appropriateness") or self._score_cultural_appropriateness(query, response_text)
        syntactic_clarity = analysis.get("syntactic_clarity") or self._score_syntactic_clarity(response_text)
        speech_act_alignment = analysis.get("speech_act_alignment") or self._score_speech_act_alignment(query, response_text)
        
        syntactic_complexity = self._score_syntactic_complexity(response_text)
        speech_act_type = self._classify_speech_act(query)
        
        # Use user-provided rating or compute from linguistic scores
        if overall_rating is not None:
            # Normalize 1-10 scale to 0-1
            rating = overall_rating / 10.0
        else:
            # Compute from linguistic features
            rating = (
                pragmatic_coherence * 0.25 +
                cultural_appropriateness * 0.20 +
                syntactic_clarity * 0.20 +
                speech_act_alignment * 0.20 +
                (1 - min(syntactic_complexity / 20, 1)) * 0.15
            )
        
        # Store in database
        self._store_annotation(
            query, response_text, rating, pragmatic_coherence,
            cultural_appropriateness, syntactic_clarity, speech_act_alignment,
            syntactic_complexity, speech_act_type, goal_context
        )
        
        return {
            "sufficient": rating > 0.6,
            "scores": {
                "overall_rating": rating * 10,  # Convert back to 1-10 scale
                "pragmatic_coherence": pragmatic_coherence,
                "cultural_appropriateness": cultural_appropriateness,
                "syntactic_clarity": syntactic_clarity,
                "speech_act_alignment": speech_act_alignment,
                "syntactic_complexity": syntactic_complexity,
                "speech_act_type": speech_act_type
            }
        }
    
    def _qwen_available(self) -> bool:
        """Check if Qwen 3 1.7B (RLAIF) is available via Ollama."""
        try:
            import ollama
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
            
            # Check for qwen3:1.7b (exact or partial match)
            qwen_found = any("qwen3" in name.lower() and "1.7b" in name.lower() for name in model_names) or "qwen3:1.7b" in model_names
            
            if qwen_found:
                return True
            else:
                available_str = ', '.join(model_names[:3]) if model_names else 'none'
                print(f"⚠️  [RLAIF] Model not found. Available: {available_str}")
                print(f"   Run: ollama pull qwen3:1.7b")
                return False
        except Exception as e:
            print(f"⚠️  [RLAIF] Ollama connection failed: {e}")
            return False
    
    def _get_top_rated_examples(self, limit: int = 2) -> list:
        """Get top rated examples from database for feedback."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query, response, overall_rating, pragmatic_coherence,
                       cultural_appropriateness, syntactic_clarity, speech_act_alignment
                FROM rlhf_annotations
                WHERE overall_rating IS NOT NULL
                ORDER BY overall_rating DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            conn.close()
            
            examples = []
            for row in rows:
                examples.append({
                    "query": row[0],
                    "response": row[1],
                    "overall_rating": row[2],
                    "pragmatic_coherence": row[3],
                    "cultural_appropriateness": row[4],
                    "syntactic_clarity": row[5],
                    "speech_act_alignment": row[6]
                })
            return examples
        except Exception as e:
            print(f"⚠️  [RLAIF] Error fetching top examples: {e}")
            return []
    
    def _analyze_with_qwen(self, query: str, response: str, feedback_context: str = "") -> Dict:
        """Use Qwen 3 1.7B (RLAIF) for linguistic analysis with feedback from top examples."""
        try:
            import ollama
            
            prompt = f"""Analyze this response for linguistic quality. Rate each on 0-1 scale.

Query: {query}
Response: {response}
{feedback_context}

Evaluate:
1. pragmatic_coherence: Does it follow Grice's Maxims (relevance, quantity, quality, manner)?
2. cultural_appropriateness: For bilingual queries, are cultural terms preserved?
3. syntactic_clarity: Is the syntax clear and parseable?
4. speech_act_alignment: Does response match query's illocutionary force (Searle's taxonomy)?

Output JSON:
{{
    "pragmatic_coherence": 0.0-1.0,
    "cultural_appropriateness": 0.0-1.0,
    "syntactic_clarity": 0.0-1.0,
    "speech_act_alignment": 0.0-1.0
}}"""
            
            result = ollama.generate(
                model="qwen3:1.7b",
                prompt=prompt,
                options={"temperature": 0.3}
            )
            
            text = result.get("response", "")
            print(f"📥 [RLAIF] Raw response: {text}")
            
            # Try to extract JSON
            try:
                # Find JSON in response
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                    print(f"✅ [RLAIF] Successfully parsed JSON")
                    return parsed
                else:
                    print(f"⚠️  [RLAIF] No JSON found in response")
            except json.JSONDecodeError as je:
                print(f"⚠️  [RLAIF] JSON parse error: {je}")
        except Exception as e:
            print(f"❌ [RLAIF] AI call failed: {e}")
        
        return {}
    
    def _score_pragmatic_coherence(self, text: str) -> float:
        """Score based on Grice's Maxims (simplified)."""
        # Check for relevance markers
        relevance_markers = ["because", "therefore", "however", "thus", "hence"]
        has_markers = any(marker in text.lower() for marker in relevance_markers)
        
        # Check length (Quantity Maxim)
        length_score = min(len(text.split()) / 100, 1.0)
        
        return 0.7 if has_markers else 0.5 + length_score * 0.2
    
    def _score_cultural_appropriateness(self, query: str, response: str) -> float:
        """Score cultural term preservation for bilingual queries."""
        # Check if query contains Chinese characters
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
        
        if not has_chinese:
            return 0.8  # Default for non-bilingual queries
        
        # Check if response mentions cultural preservation
        cultural_indicators = ["cultural", "context", "untranslatable", "潜规则", "关系", "面子"]
        has_cultural_notes = any(indicator.lower() in response.lower() for indicator in cultural_indicators)
        
        return 0.9 if has_cultural_notes else 0.6
    
    def _score_syntactic_clarity(self, text: str) -> float:
        """Score syntactic clarity (parseability)."""
        # Simple heuristics: balanced parentheses, sentence structure
        paren_balance = abs(text.count("(") - text.count(")"))
        bracket_balance = abs(text.count("[") - text.count("]"))
        
        # Check for sentence endings
        sentence_endings = text.count(".") + text.count("!") + text.count("?")
        word_count = len(text.split())
        
        clarity = 1.0
        if paren_balance > 2 or bracket_balance > 2:
            clarity -= 0.2
        if word_count > 0 and sentence_endings / word_count < 0.05:
            clarity -= 0.2
        
        return max(0.5, clarity)
    
    def _score_speech_act_alignment(self, query: str, response: str) -> float:
        """Score if response matches query's illocutionary force."""
        query_act = self._classify_speech_act(query)
        
        # Check if response type matches query intent
        if query_act == "directive" and any(word in response.lower() for word in ["will", "can", "should", "do"]):
            return 0.9
        elif query_act == "assertive" and any(word in response.lower() for word in ["is", "are", "explain", "describe"]):
            return 0.9
        else:
            return 0.7
    
    def _score_syntactic_complexity(self, text: str) -> int:
        """Estimate parse tree depth (simplified)."""
        # Count nested structures
        nested = text.count("(") + text.count("[") + text.count("{")
        clauses = text.count(",") + text.count(";")
        return nested + clauses
    
    def _classify_speech_act(self, query: str) -> str:
        """Classify using Searle's taxonomy (simplified)."""
        query_lower = query.lower()
        if any(word in query_lower for word in ["analyze", "explain", "describe"]):
            return "assertive"
        elif any(word in query_lower for word in ["do", "make", "create", "write"]):
            return "directive"
        elif any(word in query_lower for word in ["promise", "commit", "will"]):
            return "commissive"
        else:
            return "assertive"
    
    def _store_annotation(self, query: str, response: str, overall_rating: float,
                         pragmatic_coherence: float, cultural_appropriateness: float,
                         syntactic_clarity: float, speech_act_alignment: float,
                         syntactic_complexity: int, speech_act_type: str, goal_context: str):
        """Store annotation in SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO rlhf_annotations 
            (query, response, overall_rating, pragmatic_coherence, cultural_appropriateness,
             syntactic_clarity, speech_act_alignment, syntactic_complexity, speech_act_type, goal_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (query, response, overall_rating, pragmatic_coherence, cultural_appropriateness,
              syntactic_clarity, speech_act_alignment, syntactic_complexity, speech_act_type, goal_context))
        
        conn.commit()
        conn.close()

