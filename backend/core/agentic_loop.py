"""
LangGraph Agentic Loop with Perplexity Research Agent
Autonomous agent system that runs tasks in a loop
"""
import os
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import httpx
import operator

load_dotenv()


class AgentState(TypedDict):
    """State shared between agents in the loop."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_task: str
    research_results: list
    agent_thoughts: list
    loop_count: int
    max_iterations: int


class ResearchAgent:
    """Research Agent using Perplexity Sonar API."""
    
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def research(self, query: str) -> str:
        """Perform research using Perplexity."""
        if not self.api_key:
            return "Perplexity API key not configured"
        
        try:
            response = await self.client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sonar",
                    "messages": [
                        {"role": "system", "content": "You are a research assistant. Provide accurate, cited information."},
                        {"role": "user", "content": query}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Research API error: {response.status_code}"
        except Exception as e:
            return f"Research error: {str(e)}"


class ReasoningAgent:
    """Reasoning agent using Claude or GPT."""
    
    def __init__(self, use_claude: bool = True):
        self.use_claude = use_claude
        if use_claude:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.llm = ChatAnthropic(
                    model="claude-3-5-haiku-20241022",
                    api_key=api_key,
                    temperature=0.7
                )
            else:
                self.llm = None
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(
                    model="gpt-4o",
                    api_key=api_key,
                    temperature=0.7
                )
            else:
                self.llm = None
    
    async def think(self, messages: Sequence[BaseMessage]) -> str:
        """Generate reasoning/thinking."""
        if not self.llm:
            return "No LLM configured"
        
        response = await self.llm.ainvoke(messages)
        return response.content


class AetherflowAgenticLoop:
    """Main agentic loop orchestrator using LangGraph."""
    
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.reasoning_agent = ReasoningAgent(use_claude=True)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state graph for agentic loop."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.add_node("check_loop", self._check_loop_node)
        
        # Set entry point
        workflow.set_entry_point("reasoning")
        
        # Add conditional edge from reasoning to research
        workflow.add_conditional_edges(
            "reasoning",
            lambda state: "research" if "TASK_COMPLETE" not in str(state["messages"][-1].content).upper() else "synthesize"
        )
        
        # Add edges
        workflow.add_edge("research", "synthesize")
        workflow.add_edge("synthesize", "check_loop")
        
        # Conditional edge: check if we should continue or end
        workflow.add_conditional_edges(
            "check_loop",
            self._should_continue,
            {
                "continue": "reasoning",  # Loop back to reasoning
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def _reasoning_node(self, state: AgentState) -> AgentState:
        """Reasoning node: analyze task and decide what to research."""
        task = state["current_task"]
        loop_count = state.get("loop_count", 0)
        
        if loop_count == 0:
            # Initial reasoning
            prompt = f"""Analyze this task and determine what research is needed:
            
Task: {task}

What specific information should be researched? Provide a research query."""
        else:
            # Continue reasoning based on research results
            last_results = state.get("research_results", [])[-1] if state.get("research_results") else ""
            prompt = f"""Based on this research:
            
{last_results}

Task: {task}

What should be researched next, or is the task complete? If complete, say 'TASK_COMPLETE'."""
        
        messages = state["messages"]
        messages.append(HumanMessage(content=prompt))
        
        thinking = await self.reasoning_agent.think(messages)
        state["agent_thoughts"].append(f"Reasoning: {thinking}")
        
        # Check if task is complete
        if "TASK_COMPLETE" in thinking.upper():
            state["messages"].append(AIMessage(content="Task appears complete. Finalizing..."))
        else:
            state["messages"].append(AIMessage(content=thinking))
        
        return state
    
    async def _research_node(self, state: AgentState) -> AgentState:
        """Research node: perform research using Perplexity."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Extract research query from last message
        if "TASK_COMPLETE" not in last_message.upper():
            # Perform research
            research_result = await self.research_agent.research(last_message)
            state["research_results"].append(research_result)
            state["agent_thoughts"].append(f"Research: {research_result[:100]}...")
            state["messages"].append(AIMessage(content=f"Research findings: {research_result[:200]}..."))
        else:
            state["messages"].append(AIMessage(content="No further research needed."))
        
        return state
    
    async def _synthesize_node(self, state: AgentState) -> AgentState:
        """Synthesize node: combine reasoning and research."""
        task = state["current_task"]
        research_results = state.get("research_results", [])
        
        if research_results:
            synthesis_prompt = f"""Task: {task}

Research findings:
{chr(10).join(research_results[-2:])}  # Last 2 research results

Synthesize this information and provide a comprehensive answer."""
        else:
            synthesis_prompt = f"""Task: {task}

Provide your analysis."""
        
        messages = state["messages"]
        messages.append(HumanMessage(content=synthesis_prompt))
        
        synthesis = await self.reasoning_agent.think(messages)
        state["agent_thoughts"].append(f"Synthesis: {synthesis[:100]}...")
        state["messages"].append(AIMessage(content=synthesis))
        
        return state
    
    async def _check_loop_node(self, state: AgentState) -> AgentState:
        """Check if we should continue the loop or end."""
        state["loop_count"] = state.get("loop_count", 0) + 1
        max_iterations = state.get("max_iterations", 3)
        
        # Check if we should continue
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        if state["loop_count"] >= max_iterations:
            state["agent_thoughts"].append(f"Reached max iterations ({max_iterations}). Ending loop.")
            return state
        
        if "TASK_COMPLETE" in last_message.upper() or "complete" in last_message.lower():
            state["agent_thoughts"].append("Task marked as complete. Ending loop.")
            return state
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if loop should continue."""
        loop_count = state.get("loop_count", 0)
        max_iterations = state.get("max_iterations", 3)
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        if loop_count >= max_iterations:
            return "end"
        
        if "TASK_COMPLETE" in last_message.upper() or "complete" in last_message.lower():
            return "end"
        
        return "continue"
    
    async def run(self, task: str, max_iterations: int = 3) -> dict:
        """Run the agentic loop for a given task."""
        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "current_task": task,
            "research_results": [],
            "agent_thoughts": [],
            "loop_count": 0,
            "max_iterations": max_iterations
        }
        
        print(f"\n🚀 Starting agentic loop for: {task}")
        print(f"   Max iterations: {max_iterations}")
        print("-" * 60)
        
        result = await self.graph.ainvoke(initial_state)
        
        print("-" * 60)
        print(f"✅ Loop completed after {result['loop_count']} iterations")
        print("\n📝 Final Answer:")
        print(result["messages"][-1].content)
        
        return result


async def main():
    """Main function to run agentic loop."""
    loop = AetherflowAgenticLoop()
    
    # Example task
    task = "Research the latest developments in AI agent frameworks and summarize the key findings"
    
    result = await loop.run(task, max_iterations=3)
    
    print("\n🔍 Agent Thoughts:")
    for i, thought in enumerate(result["agent_thoughts"], 1):
        print(f"  {i}. {thought[:100]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

