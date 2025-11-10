"""
Cognitive Architecture Orchestrator
Implements pragmatic inference engine with proactive re-invocation loop.
Theory: Searle's Speech Act Theory + Grice's Maxims
"""

import os
import sys
from dotenv import load_dotenv
from agents.meta_head import Agent_PragmaticRouter
from agents.code_switcher import Agent_CodeSwitcher
from agents.discourse_analyzer import Agent_DiscourseAnalyzer
from agents.semantic_parser import Agent_SemanticParser
from agents.vision_analyzer import Agent_VisionAnalyzer
from agents.rlhf_critic import Agent_RLHF_Critic
from agents.memory_curator import Agent_MemoryCurator
from agents.explainability_bridge import Agent_ExplainabilityBridge
from agents.goal_memory import GoalMemory

load_dotenv()

def transcribe_audio(audio_file: str) -> str:
    """
    Transcribe audio file to text using Whisper STT.
    Theory: Scoping to text modalities to isolate pragmatic inference mechanisms.
    """
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(audio_file)
        return result.get("text", "").strip()
    except ImportError:
        print("⚠️  mlx-whisper not installed. Install with: pip install mlx-whisper")
        return ""
    except Exception as e:
        print(f"❌ Error transcribing audio: {e}")
        return ""

# Initialize agents
router = Agent_PragmaticRouter()
code_switcher = Agent_CodeSwitcher()
discourse_analyzer = Agent_DiscourseAnalyzer()
semantic_parser = Agent_SemanticParser()
vision_analyzer = Agent_VisionAnalyzer()
rlhf_critic = Agent_RLHF_Critic()
memory_curator = Agent_MemoryCurator()
explainability_bridge = Agent_ExplainabilityBridge()
goal_memory = GoalMemory()

def process_with_proactivity(query: str, max_iter: int = 2, overall_rating: float = None, image_path: str = None, image_base64: str = None, video_path: str = None, manual_agent: str = None):
    """
    Proactive intelligence loop: Qwen 3 1.7B re-invocation with evaluation.
    Theory: Abductive reasoning + iterative refinement
    
    Args:
        query: User query
        max_iter: Maximum iterations (default 2)
        overall_rating: User-provided rating (1-10) from Streamlit dashboard
        image_path: Path to image file for vision analysis
        image_base64: Base64 encoded image (alternative to image_path)
        video_path: Path to video file for frame-by-frame analysis
    
    Returns: (response, routing, explanation, evaluation, ai_status)
        ai_status: Dict tracking AI call success rates
    """
    iteration = 0
    context = []
    ai_status = {
        "routing_ai_success": [],
        "agent_ai_success": [],
        "refinement_ai_success": [],
        "total_iterations": 0
    }
    
    # Get goal context
    goal_context = goal_memory.get_goal_context()
    if goal_context:
        query = f"{goal_context}\n\nQuery: {query}"
    
    print(f"\n{'='*70}")
    print(f"🔄 PROACTIVE INTELLIGENCE LOOP (x{max_iter} iterations)")
    print(f"{'='*70}")
    
    # Store original query for first iteration
    original_query = query
    previous_response = None
    
    while iteration < max_iter:
        iteration += 1
        ai_status["total_iterations"] = iteration
        print(f"\n📊 Iteration {iteration}/{max_iter}")
        print(f"{'-'*70}")
        
        # Build query with previous response context (proactive chaining)
        if iteration > 1 and previous_response:
            # Feed previous response into next iteration
            response_text = str(previous_response.get("analysis") or previous_response.get("english_response") or previous_response)
            query = f"""Previous response from iteration {iteration-1}:
{response_text}

Original query: {original_query}

Based on the previous response, please refine or expand your analysis."""
            print(f"🔄 [Proactive] Feeding previous response into iteration {iteration}")
            print(f"   Previous response: {response_text}")
        
        # Route query using pragmatic inference (or use manual selection)
        if manual_agent:
            # Manual agent selection - skip routing
            print(f"\n🔀 Step 1: Manual Agent Selection")
            print(f"   → Selected: {manual_agent} (Manual)")
            print(f"   → Reasoning: User manually selected agent")
            routing = {
                "agent": manual_agent,
                "reasoning": "Manual selection",
                "speech_act": "assertive",
                "ai_success": True
            }
            ai_status["routing_ai_success"].append(True)
        else:
            # Automatic routing via pragmatic router
            print(f"\n🔀 Step 1: Pragmatic Routing")
            print(f"🏷️  Agent Badge: Pragmatic Router (Searle 1969 + Grice 1975)")
            routing = router.route(query, context)
            ai_status["routing_ai_success"].append(routing.get("ai_success", False))
            
            if routing.get("ai_success"):
                print(f"   ✅ Pragmatic Router routing successful")
            else:
                print(f"   ⚠️  Using fallback routing")
            
            print(f"   → Selected: {routing['agent']}")
            print(f"   → Reasoning: {routing['reasoning']}")
        
        # Call appropriate agent (with image if provided)
        # RULE: Image/Video → FIRST analyze with Gemini 2.5 Flash, THEN pass to code_switcher
        if image_path or image_base64:
            print(f"\n🖼️  Step 2a: Vision Analysis (Gemini 2.5 Flash)")
            print(f"🏷️  Agent Badge: Vision Analyzer (Multimodal Pragmatics)")
            print(f"   🤖 Model: Gemini 2.5 Flash (Google AI API)")
            # FIRST: Process image/video with vision analyzer
            vision_response = vision_analyzer.process(query, context, image_path=image_path, image_base64=image_base64)
            vision_analysis = ""
            if vision_response.get("ai_success"):
                vision_analysis = vision_response.get("analysis", "")
                print(f"   ✅ Vision Analysis (Gemini 2.5 Flash): {vision_analysis[:200]}...")
            else:
                print(f"   ⚠️  Vision analysis failed: {vision_response.get('error', 'Unknown error')}")
            
            # THEN: Process with code_switcher (cultural preservation) - include vision analysis in query
            print(f"\n🤖 Step 2b: Agent Processing (code_switcher) [Image detected → Cultural preservation]")
            print(f"🏷️  Agent Badge: Code Switcher (Myers-Scotton 1993 + Relevance Theory)")
            print(f"   🤖 Model: Kimi K2 Thinking")
            # Enhance query with vision analysis if available
            enhanced_query = query
            if vision_analysis:
                enhanced_query = f"{query}\n\n[Vision Analysis from Gemini 2.5 Flash]: {vision_analysis}"
            
            response = code_switcher.process(enhanced_query, context)
            # Combine vision analysis into response
            if isinstance(response, dict):
                response["vision_analysis"] = vision_analysis
                response["vision_ai_success"] = vision_response.get("ai_success", False)
        else:
            print(f"\n🤖 Step 2: Agent Processing ({routing['agent']})")
            # Show agent badge with theory citation
            agent_badges = {
                "code_switcher": "Code Switcher (Myers-Scotton 1993 + Relevance Theory)",
                "discourse_analyzer": "Discourse Analyzer (RST - Mann & Thompson 1988)",
                "semantic_parser": "Semantic Parser (Montague Semantics 1973)",
                "vision_analyzer": "Vision Analyzer (Multimodal Pragmatics)",
                "memory_curator": "Memory Curator (Frame Semantics - Fillmore)",
                "explainability_bridge": "Explainability Bridge (Abductive Reasoning)"
            }
            badge = agent_badges.get(routing["agent"], f"{routing['agent']} (Linguistics)")
            print(f"🏷️  Agent Badge: {badge}")
            # Show model name
            model_names = {
                "code_switcher": "Kimi K2 Thinking",
                "discourse_analyzer": "Kimi K2 Thinking",
                "semantic_parser": "Claude Sonnet 4.5",
                "vision_analyzer": "Gemini 2.5 Flash",
                "memory_curator": "Claude Haiku 4.5",
                "explainability_bridge": "Claude Haiku 4.5"
            }
            model_name = model_names.get(routing["agent"], "Unknown")
            print(f"   🤖 Model: {model_name}")
            response = call_agent(routing["agent"], query, context)
        
        # Check if agent AI call succeeded
        agent_ai_success = response.get("ai_success", False)
        ai_status["agent_ai_success"].append(agent_ai_success)
        
        if agent_ai_success:
            print(f"   ✅ Agent AI call successful")
        else:
            print(f"   ❌ Agent AI call failed or using fallback")
            if "error" in response:
                print(f"   Error: {response.get('error', 'Unknown')}")
        
        # Store response for next iteration
        previous_response = response
        
        # Evaluate if response is sufficient (with user rating if provided)
        print(f"\n📊 Step 3: RLAIF Evaluation")
        print(f"🏷️  Agent Badge: RLAIF Critic (Preference Learning)")
        evaluation = rlhf_critic.evaluate(
            response, 
            query, 
            overall_rating=overall_rating,
            goal_context=goal_context
        )
        
        sufficient = evaluation.get("sufficient", False)
        rating = evaluation.get("scores", {}).get("overall_rating", 0)
        print(f"   → Sufficient: {sufficient}")
        print(f"   → Overall Rating: {rating:.1f}/10")
        
        if sufficient and iteration >= 2:  # Require at least 2 iterations for proactive loop
            print(f"\n✅ Response is sufficient after {iteration} iterations! Stopping loop.")
            # Store in episodic memory
            print(f"💾 Storing in episodic memory...")
            print(f"🏷️  Agent Badge: Memory Curator (Frame Semantics - Fillmore)")
            memory_curator.store(query, str(response), routing)
            
            # Generate explanation
            print(f"💡 Generating explanation...")
            print(f"🏷️  Agent Badge: Explainability Bridge (Abductive Reasoning)")
            explanation = explainability_bridge.explain(query, routing, response)
            
            ai_status["final_status"] = "success"
            return response, routing, explanation, evaluation, ai_status
        
        # Continue loop for proactive refinement (x3 iterations)
        if iteration < max_iter:
            print(f"\n🔄 Step 4: Preparing for next iteration...")
            # Refine query with context for next iteration
            old_query = query
            refined_query = router.refine(query, str(response), context)
            refinement_success = (refined_query != old_query)
            ai_status["refinement_ai_success"].append(refinement_success)
            
            if refinement_success:
                print(f"   ✅ Query refined by Pragmatic Router for next iteration")
                print(f"   Old: {old_query}")
                print(f"   New: {refined_query}")
                query = refined_query
            else:
                print(f"   ⚠️  Query unchanged, will use previous response context")
            
            context.append({"query": query, "response": response})
    
    # Generate explanation after all iterations
    print(f"\n✅ Completed {max_iter} iterations. Generating final explanation...")
    
    # Ensure we have valid routing and response before generating explanation
    if not routing:
        routing = {"agent": "code_switcher", "reasoning": "Default routing", "ai_success": False}
    if not response:
        response = {"error": "No response generated", "ai_success": False}
    
    try:
        print(f"🏷️  Agent Badge: Explainability Bridge (Abductive Reasoning)")
        explanation = explainability_bridge.explain(query, routing, response)
    except Exception as e:
        print(f"⚠️  Explanation generation failed: {e}")
        explanation = f"Selected {routing.get('agent', 'unknown')} because: {routing.get('reasoning', 'default')}"
    
    try:
        evaluation = rlhf_critic.evaluate(response, query, overall_rating=overall_rating, goal_context=goal_context)
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        evaluation = {"sufficient": False, "scores": {"overall_rating": 0}}
    
    ai_status["final_status"] = "completed"
    return response, routing, explanation, evaluation, ai_status

def call_agent(agent_name: str, query: str, context: list, image_path: str = None, image_base64: str = None):
    """Route to specific agent based on pragmatic decision."""
    agents = {
        "code_switcher": code_switcher,
        "discourse_analyzer": discourse_analyzer,
        "semantic_parser": semantic_parser,
        "vision_analyzer": vision_analyzer,
    }
    
    agent = agents.get(agent_name)
    if agent:
        if agent_name == "vision_analyzer":
            return agent.process(query, context, image_path=image_path, image_base64=image_base64)
        return agent.process(query, context)
    
    return {"error": f"Unknown agent: {agent_name}"}

# Test queries for different agents
TEST_QUERIES = {
    "bilingual": "我需要讨论潜规则问题，这个在中国商业环境中很重要。Can you explain what this means in English and provide cultural context?",
    "discourse": "Analyze this meeting transcript for coherence relations: 'We need to finish the project. However, the budget is tight. Therefore, we should prioritize.'",
    "semantic": "Convert this sentence to lambda calculus: Every student passed the exam.",
    "vision": "Describe what's happening in this image in detail.",
    "simple": "What are the latest developments in AI agent frameworks?"
}

if __name__ == "__main__":
    print("=" * 70)
    print("Cognitive Architecture - Pragmatic Inference Engine")
    print("=" * 70)
    print("\nInput options:")
    print("1. Type 'voice' or 'audio' + path to audio file")
    print("2. Type 'image' or 'img' + path to image/video file")
    print("3. Type 'test' + query name (bilingual, discourse, semantic, vision, simple)")
    print("4. Type your query directly")
    print("\n" + "-" * 70)
    
    user_input = input("\nEnter query (or 'voice <path>' / 'image <path>' / 'test <name>'): ").strip()
    
    # Handle voice input
    image_path = None
    image_base64 = None
    
    if user_input.lower().startswith("voice ") or user_input.lower().startswith("audio "):
        audio_path = user_input.split(" ", 1)[1] if " " in user_input else input("Enter audio file path: ")
        print(f"\n🎤 Transcribing audio: {audio_path}")
        query = transcribe_audio(audio_path)
        if not query:
            print("❌ Failed to transcribe audio. Exiting.")
            sys.exit(1)
        print(f"✅ Transcribed: {query}\n")
    
    # Handle image input
    elif user_input.lower().startswith("image ") or user_input.lower().startswith("img "):
        image_path = user_input.split(" ", 1)[1] if " " in user_input else input("Enter image file path: ")
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            sys.exit(1)
        query = input("Enter query about the image: ").strip()
        if not query:
            query = "Describe what's happening in this image in detail."
        print(f"\n🖼️  Image: {image_path}")
        print(f"📝 Query: {query}\n")
    
    # Handle test queries
    elif user_input.lower().startswith("test "):
        test_name = user_input.split(" ", 1)[1] if " " in user_input else ""
        if test_name in TEST_QUERIES:
            query = TEST_QUERIES[test_name]
            print(f"\n🧪 Using test query: {test_name}")
            print(f"Query: {query}\n")
        else:
            print(f"❌ Unknown test query: {test_name}")
            print(f"Available: {', '.join(TEST_QUERIES.keys())}")
            sys.exit(1)
    
    # Direct text input
    else:
        query = user_input
    
    # Process query
    print("=" * 70)
    print("Processing query...")
    
    # Show current goal if set
    current_goal = goal_memory.get_current_goal()
    if current_goal:
        print(f"🎯 Current Goal: {current_goal}")
    
    print("=" * 70)
    
    response, routing, explanation, evaluation, ai_status = process_with_proactivity(
        query, 
        max_iter=2,  # x2 proactive iterations
        image_path=image_path,
        image_base64=image_base64
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # AI Status Summary
    print(f"\n🤖 AI CALL STATUS:")
    routing_success_rate = sum(ai_status["routing_ai_success"]) / len(ai_status["routing_ai_success"]) * 100 if ai_status["routing_ai_success"] else 0
    agent_success_rate = sum(ai_status["agent_ai_success"]) / len(ai_status["agent_ai_success"]) * 100 if ai_status["agent_ai_success"] else 0
    refinement_success_rate = sum(ai_status["refinement_ai_success"]) / len(ai_status["refinement_ai_success"]) * 100 if ai_status["refinement_ai_success"] else 0
    
    print(f"  Pragmatic Router: {routing_success_rate:.0f}% success ({sum(ai_status['routing_ai_success'])}/{len(ai_status['routing_ai_success'])})")
    print(f"  Agent Processing: {agent_success_rate:.0f}% success ({sum(ai_status['agent_ai_success'])}/{len(ai_status['agent_ai_success'])})")
    print(f"  Query Refinement: {refinement_success_rate:.0f}% success ({sum(ai_status['refinement_ai_success'])}/{len(ai_status['refinement_ai_success'])})")
    print(f"  Total Iterations: {ai_status['total_iterations']}")
    print(f"  Final Status: {ai_status.get('final_status', 'unknown')}")
    
    # Routing Info
    print(f"\n🔀 ROUTING DECISION:")
    print(f"  Agent: {routing['agent']}")
    print(f"  Reasoning: {routing['reasoning']}")
    if routing.get('ai_success'):
        print(f"  ✅ AI Routing (Pragmatic Router)")
    else:
        print(f"  ⚠️  Fallback Routing (keyword matching)")
    
    print(f"\n💡 EXPLANATION:")
    print(f"  {explanation}")
    
    print(f"\n📊 EVALUATION SCORES:")
    scores = evaluation.get("scores", {})
    print(f"  Overall Rating: {scores.get('overall_rating', 0):.1f}/10")
    print(f"  Pragmatic Coherence: {scores.get('pragmatic_coherence', 0):.2f}")
    print(f"  Cultural Appropriateness: {scores.get('cultural_appropriateness', 0):.2f}")
    print(f"  Syntactic Clarity: {scores.get('syntactic_clarity', 0):.2f}")
    print(f"  Speech Act Alignment: {scores.get('speech_act_alignment', 0):.2f}")
    
    print(f"\n📄 RESPONSE:")
    if isinstance(response, dict):
        if response.get("ai_success"):
            print(f"  ✅ AI Response (not fallback)")
        else:
            print(f"  ⚠️  Fallback Response")
        for key, value in response.items():
            if key != "ai_success" and key != "raw_response":
                print(f"  {key}: {value}")
    else:
        print(f"  {response}")
    
    print("\n" + "=" * 70)

