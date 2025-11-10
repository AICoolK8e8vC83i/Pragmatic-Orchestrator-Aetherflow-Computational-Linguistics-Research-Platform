"""
Agent_ExplainabilityBridge: Translates agent decisions into human-readable abductions
Theory: Abductive reasoning + Discourse deixis
Model: Claude Haiku 4.5 via AWS Bedrock
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class Agent_ExplainabilityBridge:
    """
    Translates agent decisions into human-readable explanations.
    Theory: Abductive reasoning + Discourse deixis
    """
    
    def __init__(self):
        # Setup exactly like test notebook
        self.bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AWS_BEARER_TOKEN")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        # Build endpoint exactly like test notebook
        self.endpoint = f"https://bedrock-runtime.{self.region}.amazonaws.com"
    
    def explain(self, query: str, routing: dict, response: dict) -> str:
        """
        Generate abductive explanation for routing decision.
        Example: "Because the query contained '潜规则' (culturally-loaded), 
                 the router selected Kimi based on Relevance Theory."
        """
        print(f"🏷️  Agent Badge: Explainability Bridge (Abductive Reasoning)")
        if not self.bearer_token:
            return f"Selected {routing['agent']} because: {routing['reasoning']}"
        
        prompt = f"""Explain this routing decision using abductive reasoning.

Query: {query}
Selected Agent: {routing['agent']}
Reasoning: {routing['reasoning']}
Speech Act: {routing.get('speech_act', 'assertive')}

Generate a human-readable explanation citing the linguistic theory used.
Format: "Because [observation], the router selected [agent] based on [theory]."""
        
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
                # Removed maxTokens to allow full explanation
                "temperature": 0.5
            }
        }
        
        try:
            print(f"🔄 [Claude Haiku 4.5] Generating explanation...")
            print(f"   🤖 Model: Claude Haiku 4.5 (AWS Bedrock)")
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            result = resp.json()
            
            if 'output' not in result or 'message' not in result['output']:
                print(f"❌ [Claude Haiku 4.5] Invalid response structure")
                return f"Selected {routing['agent']} based on {routing['reasoning']}"
            
            explanation = result['output']['message']['content'][0]['text']
            
            if explanation:
                print(f"✅ [Claude Haiku 4.5] AI call successful")
                return explanation
            else:
                print(f"⚠️  [Claude Haiku 4.5] Empty explanation")
                return f"Selected {routing['agent']} based on {routing['reasoning']}"
        except Exception as e:
            print(f"❌ [Claude Haiku 4.5] AI call failed: {e}, using fallback")
            return f"Selected {routing['agent']} based on {routing['reasoning']}"

