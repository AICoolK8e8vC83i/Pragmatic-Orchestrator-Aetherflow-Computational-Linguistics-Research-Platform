"""
Meta Router Agentic Loop with DeepSeek R1 1.5B
DeepSeek R1 acts as meta-router, proactively deciding which models to call
until a SOTA answer is achieved.
"""
import os
import json
import asyncio
from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import httpx
import operator

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

load_dotenv()


class MetaRouterState(TypedDict):
    """State for meta-router agentic loop."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    original_task: str
    current_subtask: str
    model_calls: list  # Track which models were called
    answers: dict  # Store answers from each model
    confidence_scores: dict  # Track confidence per answer
    router_decision: str  # DeepSeek's decision on next model
    sota_achieved: bool  # Whether we've achieved SOTA answer
    iteration_count: int
    max_iterations: int


class DeepSeekMetaRouter:
    """DeepSeek R1 1.5B as meta-router for model selection."""
    
    def __init__(self):
        self.model_name = "deepseek-r1:1.5b"
        if not OLLAMA_AVAILABLE:
            print("⚠️  ollama library not installed. Install with: pip install ollama")
    
    async def route(self, state: MetaRouterState) -> dict:
        """
        DeepSeek R1 decides which model to call next.
        
        Returns:
        {
            "model": "claude_haiku" | "gpt_4o_mini" | "perplexity_sonar" | "continue" | "sota_achieved",
            "reasoning": "Why this model",
            "confidence": 0.85
        }
        """
        task = state["original_task"]
        model_calls = state.get("model_calls", [])
        answers = state.get("answers", {})
        iteration = state.get("iteration_count", 0)
        
        # Build context for DeepSeek
        context = f"""Task: {task}
        
Iteration: {iteration}
Models already called: {', '.join(model_calls) if model_calls else 'None'}
"""
        
        if answers:
            context += "\nPrevious answers:\n"
            for model, answer in list(answers.items())[-3:]:  # Last 3 answers
                context += f"- {model}: {answer[:100]}...\n"
        
        routing_prompt = f"""You are a meta-router for an AI agent system. Your job is to decide which model to call next to achieve the best answer.

{context}

        Available models:
1. claude_4_5_haiku - Fast, cost-effective for structured reasoning and code (claude-4.5-haiku)
2. claude_4_5_sonnet - Best for coding, agentic tasks, general-purpose (claude-4.5-sonnet)
3. claude_4_1_opus - Flagship frontier model for advanced reasoning (claude-4.1-opus)
4. gpt_5_mini - Fast, cost-effective reasoning (gpt-5-mini)
5. gpt_4o_mini - Fast, good for creative tasks (gpt-4o-mini)
6. sonar - Perplexity standard research model (sonar)
7. sonar_reasoning_pro - Perplexity advanced reasoning with research (sonar-reasoning-pro)
8. sonar_deep_research - Perplexity exhaustive deep research (sonar-deep-research)
9. continue - Continue with current model if more iterations needed
10. sota_achieved - Stop, we have a state-of-the-art answer

Respond in JSON format:
{{
    "model": "model_name",
    "reasoning": "Why this model is best for this iteration",
    "confidence": 0.0-1.0,
    "sota_achieved": true/false
}}"""
        
        if not OLLAMA_AVAILABLE:
            # Fallback if ollama not installed
            return {
                "model": "gpt_4o_mini",
                "reasoning": "Ollama library not available, using fallback",
                "confidence": 0.3,
                "sota_achieved": False
            }
        
        try:
            # Use proper ollama library (sync, wrapped in executor for async)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.generate(
                    model=self.model_name,
                    prompt=routing_prompt,
                    options={
                        "temperature": 0.3,  # Lower temp for routing decisions
                        "top_p": 0.9
                    }
                )
            )
            
            response_text = response.get("response", "")
            
            # Try to extract JSON from response
            try:
                # Look for JSON in response
                if "{" in response_text and "}" in response_text:
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    decision = json.loads(response_text[json_start:json_end])
                    return decision
                else:
                    # Fallback: simple parsing
                    return {
                        "model": "gpt_4o_mini",
                        "reasoning": "Fallback routing - no JSON in response",
                        "confidence": 0.5,
                        "sota_achieved": False
                    }
            except json.JSONDecodeError:
                # Fallback decision
                if "sota" in response_text.lower() or "complete" in response_text.lower():
                    return {
                        "model": "sota_achieved",
                        "reasoning": response_text[:100],
                        "confidence": 0.8,
                        "sota_achieved": True
                    }
                return {
                    "model": "gpt_4o_mini",
                    "reasoning": "Fallback routing - JSON decode error",
                    "confidence": 0.5,
                    "sota_achieved": False
                }
        except Exception as e:
            print(f"⚠️ DeepSeek routing error: {e}")
            return {
                "model": "gpt_4o_mini",
                "reasoning": f"Error: {str(e)}",
                "confidence": 0.3,
                "sota_achieved": False
            }


class ModelExecutor:
    """Executes calls to different models."""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def call_claude_4_5_haiku(self, prompt: str) -> str:
        """Call Claude 4.5 Haiku (SOTA for speed)."""
        if not self.anthropic_key:
            return "ANTHROPIC_API_KEY not configured"
        
        try:
            llm = ChatAnthropic(
                model="claude-4.5-haiku",
                api_key=self.anthropic_key,
                temperature=0.7
            )
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Claude 4.5 Haiku error: {str(e)}"
    
    async def call_claude_4_5_sonnet(self, prompt: str) -> str:
        """Call Claude 4.5 Sonnet (SOTA for general/coding)."""
        if not self.anthropic_key:
            return "ANTHROPIC_API_KEY not configured"
        
        try:
            llm = ChatAnthropic(
                model="claude-4.5-sonnet",
                api_key=self.anthropic_key,
                temperature=0.7
            )
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Claude 4.5 Sonnet error: {str(e)}"
    
    async def call_claude_4_1_opus(self, prompt: str) -> str:
        """Call Claude 4.1 Opus (SOTA for top-tier reasoning)."""
        if not self.anthropic_key:
            return "ANTHROPIC_API_KEY not configured"
        
        try:
            llm = ChatAnthropic(
                model="claude-4.1-opus",
                api_key=self.anthropic_key,
                temperature=0.7
            )
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Claude 4.1 Opus error: {str(e)}"
    
    async def call_gpt_5_mini(self, prompt: str) -> str:
        """Call GPT-5-mini (latest reasoning model)."""
        if not self.openai_key:
            return "OPENAI_API_KEY not configured"
        
        try:
            llm = ChatOpenAI(
                model="gpt-5-mini",
                api_key=self.openai_key,
                temperature=0.7
            )
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"GPT-5-mini error: {str(e)}"
    
    async def call_gpt_4o_mini(self, prompt: str) -> str:
        """Call GPT-4o-mini (fast, cost-effective)."""
        if not self.openai_key:
            return "OPENAI_API_KEY not configured"
        
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=self.openai_key,
                temperature=0.7
            )
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"GPT-4o-mini error: {str(e)}"
    
    async def call_sonar(self, query: str) -> str:
        """Call Perplexity Sonar (standard research)."""
        if not self.perplexity_key:
            return "PERPLEXITY_API_KEY not configured"
        
        try:
            response = await self.client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.perplexity_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sonar",
                    "messages": [
                        {"role": "system", "content": "You are a research assistant. Provide accurate, cited information."},
                        {"role": "user", "content": query}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Sonar error: {response.status_code}"
        except Exception as e:
            return f"Sonar error: {str(e)}"
    
    async def call_sonar_reasoning_pro(self, query: str) -> str:
        """Call Perplexity Sonar Reasoning Pro (advanced reasoning + research)."""
        if not self.perplexity_key:
            return "PERPLEXITY_API_KEY not configured"
        
        try:
            response = await self.client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.perplexity_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sonar-reasoning-pro",
                    "messages": [
                        {"role": "system", "content": "You are an advanced research assistant with reasoning capabilities. Provide in-depth analysis with citations."},
                        {"role": "user", "content": query}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1500
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Sonar Reasoning Pro error: {response.status_code}"
        except Exception as e:
            return f"Sonar Reasoning Pro error: {str(e)}"
    
    async def call_sonar_deep_research(self, query: str) -> str:
        """Call Perplexity Sonar Deep Research (exhaustive research)."""
        if not self.perplexity_key:
            return "PERPLEXITY_API_KEY not configured"
        
        try:
            response = await self.client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.perplexity_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sonar-deep-research",
                    "messages": [
                        {"role": "system", "content": "You are an exhaustive research assistant. Provide comprehensive analysis with citations."},
                        {"role": "user", "content": query}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "reasoning_effort": "low"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Sonar Deep Research error: {response.status_code}"
        except Exception as e:
            return f"Sonar Deep Research error: {str(e)}"
    
    async def execute(self, model_name: str, prompt: str) -> str:
        """Execute model call based on model name."""
        if model_name == "claude_4_5_haiku":
            return await self.call_claude_4_5_haiku(prompt)
        elif model_name == "claude_4_5_sonnet":
            return await self.call_claude_4_5_sonnet(prompt)
        elif model_name == "claude_4_1_opus":
            return await self.call_claude_4_1_opus(prompt)
        elif model_name == "gpt_5_mini":
            return await self.call_gpt_5_mini(prompt)
        elif model_name == "gpt_4o_mini":
            return await self.call_gpt_4o_mini(prompt)
        elif model_name == "sonar":
            return await self.call_sonar(prompt)
        elif model_name == "sonar_reasoning_pro":
            return await self.call_sonar_reasoning_pro(prompt)
        elif model_name == "sonar_deep_research":
            return await self.call_sonar_deep_research(prompt)
        else:
            return f"Unknown model: {model_name}"


class ProactiveAgenticLoop:
    """Proactive agentic loop with DeepSeek R1 meta-routing."""
    
    def __init__(self):
        self.router = DeepSeekMetaRouter()
        self.executor = ModelExecutor()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph for proactive model routing."""
        workflow = StateGraph(MetaRouterState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("evaluate", self._evaluate_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add edges
        workflow.add_edge("router", "execute")
        workflow.add_edge("execute", "evaluate")
        
        # Conditional edge: continue or end
        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {
                "continue": "router",  # Loop back to router
                "sota_achieved": END,
                "max_iterations": END
            }
        )
        
        return workflow.compile()
    
    async def _router_node(self, state: MetaRouterState) -> MetaRouterState:
        """Router node: DeepSeek decides which model to call."""
        decision = await self.router.route(state)
        
        state["router_decision"] = decision["model"]
        state["sota_achieved"] = decision.get("sota_achieved", False)
        
        print(f"\n🧭 DeepSeek Router Decision: {decision['model']}")
        print(f"   Reasoning: {decision['reasoning']}")
        print(f"   Confidence: {decision.get('confidence', 0.0):.2f}")
        
        return state
    
    async def _execute_node(self, state: MetaRouterState) -> MetaRouterState:
        """Execute node: Call the selected model."""
        model_name = state["router_decision"]
        task = state["original_task"]
        iteration = state.get("iteration_count", 0)
        
        # Build prompt based on iteration
        if iteration == 0:
            prompt = task
        else:
            # Include previous answers for context
            previous_answers = state.get("answers", {})
            context = "\n\nPrevious model responses:\n"
            for model, answer in previous_answers.items():
                context += f"{model}: {answer[:200]}...\n"
            
            prompt = f"{task}\n\n{context}\n\nRefine or extend the answer based on previous responses."
        
        print(f"\n⚡ Executing: {model_name}")
        
        # Execute model call
        answer = await self.executor.execute(model_name, prompt)
        
        # Store results
        state["model_calls"].append(model_name)
        state["answers"][model_name] = answer
        state["confidence_scores"][model_name] = 0.7  # Default confidence
        
        print(f"✅ {model_name} response: {answer[:150]}...")
        
        return state
    
    async def _evaluate_node(self, state: MetaRouterState) -> MetaRouterState:
        """Evaluate node: Check if we have SOTA answer."""
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        # Check if SOTA achieved
        if state.get("sota_achieved", False):
            print("\n🎯 SOTA answer achieved!")
            return state
        
        # Check max iterations
        if state["iteration_count"] >= state.get("max_iterations", 5):
            print(f"\n⚠️ Reached max iterations ({state['max_iterations']})")
            return state
        
        return state
    
    def _should_continue(self, state: MetaRouterState) -> Literal["continue", "sota_achieved", "max_iterations"]:
        """Determine if loop should continue."""
        if state.get("sota_achieved", False):
            return "sota_achieved"
        
        if state["iteration_count"] >= state.get("max_iterations", 5):
            return "max_iterations"
        
        return "continue"
    
    async def run(self, task: str, max_iterations: int = 5) -> dict:
        """Run proactive agentic loop."""
        initial_state: MetaRouterState = {
            "messages": [HumanMessage(content=task)],
            "original_task": task,
            "current_subtask": task,
            "model_calls": [],
            "answers": {},
            "confidence_scores": {},
            "router_decision": "",
            "sota_achieved": False,
            "iteration_count": 0,
            "max_iterations": max_iterations
        }
        
        print("=" * 70)
        print("🚀 PROACTIVE AGENTIC LOOP - DeepSeek R1 Meta-Router")
        print("=" * 70)
        print(f"\n📝 Task: {task}")
        print(f"   Max iterations: {max_iterations}")
        print("-" * 70)
        
        result = await self.graph.ainvoke(initial_state)
        
        print("-" * 70)
        print(f"\n✅ Loop completed after {result['iteration_count']} iterations")
        print(f"   Models called: {', '.join(result['model_calls'])}")
        
        if result.get("sota_achieved"):
            print("   Status: SOTA answer achieved! 🎯")
        else:
            print("   Status: Max iterations reached")
        
        print("\n📊 Final Answers:")
        for model, answer in result["answers"].items():
            print(f"\n   {model}:")
            print(f"   {answer[:200]}...")
        
        return result


async def main():
    """Main function."""
    loop = ProactiveAgenticLoop()
    
    # Example task
    task = "Research the latest developments in multi-agent AI systems and compare LangGraph, CrewAI, and AutoGen"
    
    result = await loop.run(task, max_iterations=5)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

