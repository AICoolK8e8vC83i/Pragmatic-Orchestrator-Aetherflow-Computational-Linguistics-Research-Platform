"""
Research Terminal X - Computational Linguistics Research Platform
Dark mode, monospace fonts, live metrics, theory-forward.
Style: MIT AI Lab terminal, Sparse but powerful, Theory citations visible
"""

import streamlit as st
import sys
import os
import tempfile
import base64
import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from typing import Optional, List, Dict
import time
import threading
import cv2  # For video frame extraction

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import process_with_proactivity, goal_memory, TEST_QUERIES, transcribe_audio
from agents.vision_analyzer import Agent_VisionAnalyzer
from message_history import MessageHistory

# ============================================================================
# STYLING - Research Terminal X Dark Mode
# ============================================================================

st.set_page_config(
    page_title="Research Terminal X",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode terminal aesthetic
st.markdown("""
<style>
    /* Dark mode background */
    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
    }
    
    /* Monospace fonts */
    .stMarkdown, .stText, .stCodeBlock {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    
    /* Terminal-style containers */
    .terminal-box {
        background-color: #1e1e1e;
        border: 1px solid #00bcd4;
        border-radius: 4px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #e0e0e0;
    }
    
    /* Agent badges */
    .agent-badge {
        display: inline-block;
        background-color: #1e1e1e;
        border: 1px solid #00bcd4;
        padding: 5px 10px;
        margin: 5px;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
    }
    
    /* Live log styling */
    .log-entry {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        padding: 2px 0;
        color: #00bcd4;
    }
    
    .log-entry.routing { color: #00bcd4; }
    .log-entry.thinking { color: #ff9800; }
    .log-entry.round { color: #4caf50; }
    .log-entry.error { color: #f44336; }
    
    /* Metrics cards */
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #00bcd4;
        padding: 10px;
        border-radius: 4px;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# AGENT BADGES WITH THEORY CITATIONS
# ============================================================================

AGENT_BADGES = {
    "meta_head": {
        "name": "Qwen 3 1.7B",
        "theory": "Searle (1969) + Grice (1975)",
        "color": "#00bcd4"
    },
    "code_switcher": {
        "name": "Kimi K2",
        "theory": "Myers-Scotton (1993) + Relevance Theory",
        "color": "#ff9800"
    },
    "discourse_analyzer": {
        "name": "Kimi K2",
        "theory": "RST (Mann & Thompson 1988)",
        "color": "#ff9800"
    },
    "semantic_parser": {
        "name": "Claude Sonnet 4.5",
        "theory": "Montague Semantics (1973)",
        "color": "#4caf50"
    },
    "vision_analyzer": {
        "name": "Gemini 2.5 Flash",
        "theory": "Multimodal Pragmatics",
        "color": "#9c27b0"
    },
    "rlhf_critic": {
        "name": "RLAIF (Qwen 3 1.7B)",
        "theory": "Preference Learning",
        "color": "#00bcd4"
    },
    "memory_curator": {
        "name": "Claude Haiku 4.5",
        "theory": "Frame Semantics (Fillmore)",
        "color": "#4caf50"
    },
    "explainability_bridge": {
        "name": "Claude Haiku 4.5",
        "theory": "Abductive Reasoning",
        "color": "#4caf50"
    }
}

# ============================================================================
# VIDEO PROCESSING UTILITIES
# ============================================================================

def extract_video_frames(video_path: str, fps: float = 1.0) -> List[str]:
    """
    Extract frames from video at specified FPS.
    Returns list of base64-encoded frame images.
    """
    try:
        import cv2
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return []
        
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(frame_rate / fps)  # Extract every N frames
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames.append(frame_base64)
            
            frame_count += 1
        
        cap.release()
        return frames
    except Exception as e:
        st.error(f"Video processing error: {e}")
        return []

def process_video_frames(video_path: str, query: str, fps: float = 1.0) -> List[Dict]:
    """
    Process video frames with Gemini 2.5 Flash.
    Returns list of analysis results per frame.
    """
    frames = extract_video_frames(video_path, fps)
    if not frames:
        return []
    
    vision_analyzer = Agent_VisionAnalyzer()
    results = []
    
    for i, frame_base64 in enumerate(frames):
        frame_query = f"Frame {i+1}/{len(frames)}: {query}"
        result = vision_analyzer.process(frame_query, None, image_base64=frame_base64)
        results.append({
            "frame": i + 1,
            "timestamp": i / fps,
            "analysis": result.get("analysis", ""),
            "ai_success": result.get("ai_success", False)
        })
    
    return results

# ============================================================================
# MEMORY STREAM UTILITIES
# ============================================================================

def get_memory_stream(limit: int = 20) -> pd.DataFrame:
    """Get episodic memory stream from SQLite. Falls back to top-rated RLHF annotations for demo."""
    try:
        conn = sqlite3.connect("episodic_memory.db")
        # Fix: table name is 'episodic_memory', not 'episodes'
        df = pd.read_sql(
            "SELECT * FROM episodic_memory ORDER BY timestamp DESC LIMIT ?",
            conn,
            params=(limit,)
        )
        conn.close()
        if not df.empty:
            return df
    except Exception as e:
        pass
    
    # Fallback: Use top-rated RLHF annotations (better for demo - shows human feedback learning)
    try:
        conn = sqlite3.connect("rlhf_annotations.db")
        # Get top-rated RLHF annotations sorted by overall rating
        df = pd.read_sql(
            """SELECT query, response, timestamp, overall_rating as rating,
                      pragmatic_coherence, cultural_appropriateness, syntactic_clarity, speech_act_alignment
               FROM rlhf_annotations 
               ORDER BY overall_rating DESC, timestamp DESC 
               LIMIT ?""",
            conn,
            params=(limit,)
        )
        conn.close()
        if not df.empty:
            return df
    except Exception as e:
        pass
    
    return pd.DataFrame()

def get_rlhf_annotations(limit: int = 20) -> pd.DataFrame:
    """Get RLHF annotations from SQLite."""
    try:
        conn = sqlite3.connect("rlhf_annotations.db")
        df = pd.read_sql(
            """SELECT query, overall_rating, pragmatic_coherence, cultural_appropriateness,
                      syntactic_clarity, speech_act_alignment, timestamp
               FROM rlhf_annotations ORDER BY timestamp DESC LIMIT ?""",
            conn,
            params=(limit,)
        )
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# ============================================================================
# RLHF SPIDER CHART
# ============================================================================

def create_rlhf_spider_chart(scores: Dict) -> plt.Figure:
    """Create polar/spider chart for RLHF scores."""
    categories = ['Overall', 'Pragmatic', 'Cultural', 'Clarity', 'Speech Act']
    
    # Normalize scores: overall_rating is 1-10, others are 0-1
    overall = scores.get('overall_rating', 0)
    if overall > 1.0:  # If it's in 1-10 scale, normalize
        overall = overall / 10.0
    elif overall < 0:
        overall = 0.0
    
    values = [
        overall,
        max(0.0, min(1.0, scores.get('pragmatic_coherence', 0))),  # Clamp to 0-1
        max(0.0, min(1.0, scores.get('cultural_appropriateness', 0))),
        max(0.0, min(1.0, scores.get('syntactic_clarity', 0))),
        max(0.0, min(1.0, scores.get('speech_act_alignment', 0)))
    ]
    
    # Close the polygon
    values += [values[0]]
    categories += [categories[0]]
    
    # Create polar plot
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True).tolist()
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='#00bcd4', label='RLHF Scores')
    ax.fill(angles, values, alpha=0.25, color='#00bcd4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=10, fontfamily='monospace')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, color='#1e1e1e', linestyle='--', alpha=0.5)
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    ax.tick_params(colors='#e0e0e0')
    
    return fig

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("🧠 Pragmatic Orchestrator")
st.markdown("**Pragma-Logos: Where Speech Acts Meet Software** | Computational Linguistics Research Platform")

# 3-Column Layout
col_left, col_center, col_right = st.columns([1, 2, 1])

# ============================================================================
# LEFT COLUMN: CONTROLS
# ============================================================================

with col_left:
    st.subheader("📁 Controls")
    
    # Goal Memory
    with st.expander("🎯 Goal Memory", expanded=True):
        current_goal = goal_memory.get_current_goal()
        new_goal = st.text_area(
            "Current Goal",
            value=current_goal or "",
            height=100,
            key="goal_input",
            label_visibility="collapsed"
        )
        if st.button("Update Goal", use_container_width=True):
            goal_memory.set_goal(new_goal)
            st.success("✅ Goal updated")
            st.rerun()
    
    # Multimodal Input
    st.subheader("📁 Multimodal Input")
    
    # Test Queries (to populate text box)
    with st.expander("🧪 Test Queries"):
        selected_test = st.selectbox(
            "Select test query to populate text box",
            ["None"] + list(TEST_QUERIES.keys()),
            label_visibility="collapsed"
        )
        if selected_test != "None":
            # Store test query in session state to populate text area
            if 'test_query_selected' not in st.session_state or st.session_state.get('last_test_selected') != selected_test:
                st.session_state['test_query_selected'] = TEST_QUERIES[selected_test]
                st.session_state['last_test_selected'] = selected_test
    
    # Text Input (editable for both text and multimodal)
    query = st.text_area(
        "Query / Prompt",
        height=100,
        placeholder="Enter your query or prompt here... (works for text, image, video, and audio)",
        value=st.session_state.get('test_query_selected', ''),
        key="main_query"
    )
    
    # Clear test query selection after it's been used
    if 'test_query_selected' in st.session_state and query != st.session_state['test_query_selected']:
        st.session_state.pop('test_query_selected', None)
    
    # Image Upload
    image_file = st.file_uploader(
        "📷 Upload Image",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        help="Chinese paper, document, or image for analysis"
    )
    
    # Video Upload
    video_file = st.file_uploader(
        "🎬 Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Video will be processed frame-by-frame (0.5-1s intervals)"
    )
    
    # Audio Upload / Voice Recording
    audio_file = st.file_uploader(
        "🎤 Upload Audio / Voice",
        type=["wav", "mp3", "m4a"],
        help="Audio will be transcribed using Whisper STT"
    )
    
    # Voice Recording (if available)
    if st.button("🎙️ Record Voice", use_container_width=True):
        st.info("Voice recording feature - use audio upload for now")
    
    # Agent Selection Dropdown
    st.subheader("🤖 Agent Selection")
    agent_options = [
        "Auto (Pragmatic Router)",
        "Agent_CodeSwitcher (Kimi K2)",
        "Agent_DiscourseAnalyzer (Kimi K2)",
        "Agent_SemanticParser (Claude Sonnet 4.5)",
        "Agent_MemoryCurator (Claude Haiku 4.5)",
        "Agent_RLAIF_Critic (Qwen 3 1.7B)"
    ]
    selected_agent = st.selectbox(
        "Select agent (or Auto for routing)",
        agent_options,
        help="Image/Audio → Always CodeSwitcher | Text → Choose based on query type"
    )
    
    # Process Button
    process_button = st.button("🚀 Process Query", type="primary", use_container_width=True)

# ============================================================================
# CENTER COLUMN: EXECUTION FLOW
# ============================================================================

with col_center:
    st.subheader("⚡ Execution Flow")
    
    # Live Terminal Log Container
    log_container = st.empty()
    
    # Results Container
    results_container = st.empty()
    
    # Initialize message history
    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = MessageHistory()
    
    if process_button:
        if not query and not image_file and not video_file and not audio_file:
            st.warning("⚠️ Please enter a query or upload a file")
        else:
            # Store user query in message history
            if query:
                st.session_state['message_history'].add_message(
                    "user",
                    query,
                    {"timestamp": time.time()}
                )
            
            # Initialize variables
            image_path = None
            image_base64 = None
            video_path = None
            transcribed_text = ""
            
            # Process audio
            if audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                    tmp_file.write(audio_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    transcribed_text = transcribe_audio(tmp_path)
                    if transcribed_text:
                        query = transcribed_text
                        st.success(f"✅ Transcribed: {transcribed_text}")
                except Exception as e:
                    st.error(f"❌ Transcription error: {e}")
                finally:
                    os.unlink(tmp_path)
            
            # Process image
            if image_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.name)[1]) as tmp_file:
                    tmp_file.write(image_file.read())
                    image_path = tmp_file.name
                
                image_file.seek(0)
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                st.image(image_file, caption=f"Processing: {image_file.name}", width=300)
            
            # Process video
            video_results = []
            if video_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_file:
                    tmp_file.write(video_file.read())
                    video_path = tmp_file.name
                
                st.info(f"🎬 Processing video: {video_file.name}")
                with st.spinner("Extracting frames and analyzing with Gemini 2.5 Flash..."):
                    video_query = query or "Describe what's happening in this video frame by frame."
                    video_results = process_video_frames(video_path, video_query, fps=1.0)
                
                if video_results:
                    st.success(f"✅ Processed {len(video_results)} frames")
                    # Concatenate all frame analyses
                    combined_analysis = "\n\n".join([
                        f"[Frame {r['frame']} @ {r['timestamp']:.1f}s]: {r['analysis']}"
                        for r in video_results
                    ])
                    # Use first frame as image for routing
                    if video_results:
                        # Extract first frame as base64 for routing
                        import cv2
                        cap = cv2.VideoCapture(video_path)
                        ret, frame = cap.read()
                        if ret:
                            _, buffer = cv2.imencode('.jpg', frame)
                            image_base64 = base64.b64encode(buffer).decode('utf-8')
                        cap.release()
            
            # Map selected agent to agent name
            agent_mapping = {
                "Auto (Pragmatic Router)": None,
                "Agent_CodeSwitcher (Kimi K2)": "code_switcher",
                "Agent_DiscourseAnalyzer (Kimi K2)": "discourse_analyzer",
                "Agent_SemanticParser (Claude Sonnet 4.5)": "semantic_parser",
                "Agent_MemoryCurator (Claude Haiku 4.5)": "memory_curator",
                "Agent_RLAIF_Critic (Qwen 3 1.7B)": None  # RLAIF is not a processing agent
            }
            manual_agent = agent_mapping.get(selected_agent, None)
            
            # Process query with live logging
            if query or image_file or video_file:
                execution_logs = []
                
                # Capture stdout for live terminal feed
                import io
                import sys
                from contextlib import redirect_stdout, redirect_stderr
                
                class StreamlitLogger:
                    """Capture stdout and display in Streamlit in real-time."""
                    def __init__(self, container):
                        self.container = container
                        self.logs = []
                        self.buffer = ""
                    
                    def write(self, text):
                        if text.strip():
                            self.logs.append(text.strip())
                            # Update log display
                            self.update_display()
                    
                    def flush(self):
                        pass
                    
                    def update_display(self):
                        with self.container.container():
                            st.markdown('<div class="terminal-box">', unsafe_allow_html=True)
                            for log in self.logs[-100:]:  # Show last 100 lines
                                # Categorize log messages
                                if "Routing" in log or "Qwen 3 1.7B" in log or "meta_head" in log:
                                    st.markdown(f'<div class="log-entry routing">🔧 {log}</div>', unsafe_allow_html=True)
                                elif "Thinking" in log or "Kimi K2" in log or "K2" in log:
                                    st.markdown(f'<div class="log-entry thinking">🤔 {log}</div>', unsafe_allow_html=True)
                                elif "Round" in log or "Iteration" in log or "🔄" in log:
                                    st.markdown(f'<div class="log-entry round">🔄 {log}</div>', unsafe_allow_html=True)
                                elif "Error" in log or "Failed" in log or "❌" in log:
                                    st.markdown(f'<div class="log-entry error">❌ {log}</div>', unsafe_allow_html=True)
                                elif "✅" in log or "Success" in log:
                                    st.markdown(f'<div class="log-entry" style="color: #4caf50;">✅ {log}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="log-entry">📊 {log}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                
                # Process with proactivity
                logger = StreamlitLogger(log_container)
                
                spinner_text = "Processing with manual agent selection..." if manual_agent else "Processing through pragmatic router..."
                with st.spinner(spinner_text):
                    try:
                        # Redirect stdout to our logger
                        old_stdout = sys.stdout
                        sys.stdout = logger
                        
                        # Process query (NO overall_rating here - rating happens AFTER response)
                        response, routing, explanation, evaluation, ai_status = process_with_proactivity(
                            query or "Analyze the uploaded media.",
                            max_iter=2,  # x2 proactive iterations
                            overall_rating=None,  # Rating happens AFTER response
                            image_path=image_path,
                            image_base64=image_base64,
                            manual_agent=manual_agent  # Manual agent selection (None = auto routing)
                        )
                        
                        # Restore stdout
                        sys.stdout = old_stdout
                        
                        # Store evaluation in session state for right column
                        st.session_state['last_evaluation'] = evaluation
                        st.session_state['last_ai_status'] = ai_status
                        st.session_state['last_routing'] = routing
                        st.session_state['last_response'] = response
                        st.session_state['last_explanation'] = explanation
                        st.session_state['last_video_results'] = video_results if video_results else []
                        
                        # Display results
                        with results_container.container():
                            st.subheader("📊 Results")
                            
                            # Agent Badge
                            agent_name = routing.get("agent", "unknown")
                            if agent_name in AGENT_BADGES:
                                badge = AGENT_BADGES[agent_name]
                                st.markdown(f"""
                                <div class="agent-badge" style="border-color: {badge['color']};">
                                    <strong>{badge['name']}</strong> | {badge['theory']}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # AI Status Metrics
                            col1, col2, col3 = st.columns(3)
                            routing_success = sum(ai_status["routing_ai_success"]) / len(ai_status["routing_ai_success"]) * 100 if ai_status["routing_ai_success"] else 0
                            agent_success = sum(ai_status["agent_ai_success"]) / len(ai_status["agent_ai_success"]) * 100 if ai_status["agent_ai_success"] else 0
                            
                            with col1:
                                st.metric("Routing AI", f"{routing_success:.0f}%", 
                                         "✅" if routing_success > 50 else "⚠️")
                            with col2:
                                st.metric("Agent AI", f"{agent_success:.0f}%",
                                         "✅" if agent_success > 50 else "⚠️")
                            with col3:
                                st.metric("Iterations", ai_status["total_iterations"])
                            
                            # Routing Decision
                            st.markdown("---")
                            st.subheader("🔀 Routing Decision")
                            if routing.get('ai_success'):
                                st.success(f"✅ AI Routing (Qwen 3 1.7B)")
                            else:
                                st.warning(f"⚠️ Fallback Routing")
                            
                            st.code(f"Agent: {routing['agent']}\nReasoning: {routing['reasoning']}", language="text")
                            
                            # Explanation
                            st.markdown("---")
                            st.subheader("💡 Explanation")
                            st.markdown(explanation)
                            
                            # Response (Full markdown, no truncation)
                            st.markdown("---")
                            st.subheader("📄 Response")
                            if isinstance(response, dict):
                                if response.get("ai_success"):
                                    st.success("✅ AI Response (not fallback)")
                                else:
                                    st.warning("⚠️ Fallback Response")
                                
                                # Display full response in markdown
                                # Convert JSON to markdown for Kimi K2 and Claude responses
                                agent_name = routing.get('agent', '')
                                is_kimi_or_claude = agent_name in ['code_switcher', 'discourse_analyzer', 'semantic_parser', 'memory_curator', 'explainability_bridge']
                                
                                response_text = ""
                                for key, value in response.items():
                                    if key not in ["ai_success", "raw_response", "error"]:
                                        if isinstance(value, str):
                                            response_text += f"### {key.replace('_', ' ').title()}\n\n{value}\n\n"
                                        else:
                                            # For Kimi K2 and Claude, convert JSON to formatted markdown
                                            if is_kimi_or_claude:
                                                json_str = json.dumps(value, indent=2, ensure_ascii=False)
                                                response_text += f"### {key.replace('_', ' ').title()}\n\n```json\n{json_str}\n```\n\n"
                                            else:
                                                response_text += f"### {key.replace('_', ' ').title()}\n\n{value}\n\n"
                                
                                if response_text:
                                    st.markdown(response_text)
                                    
                                    # Store in message history
                                    if 'message_history' not in st.session_state:
                                        st.session_state['message_history'] = MessageHistory()
                                    st.session_state['message_history'].add_message(
                                        "assistant",
                                        response_text,
                                        {"agent": agent_name, "routing": routing, "query": query}
                                    )
                                
                                if "error" in response:
                                    st.error(f"Error: {response['error']}")
                            else:
                                st.markdown(str(response))
                            
                            # Video Results (if processed)
                            if video_results:
                                st.markdown("---")
                                st.subheader("🎬 Video Frame Analysis")
                                for result in video_results:
                                    with st.expander(f"Frame {result['frame']} @ {result['timestamp']:.1f}s"):
                                        st.markdown(result['analysis'])
                            
                            # RLHF Rating Section (AFTER response)
                            st.markdown("---")
                            st.subheader("⭐ Rate Response (RLHF)")
                            st.markdown("**Manual rating is core to RLHF - adjust all sliders, then click Save:**")
                            
                            # Use a form to "freeze" the UI - nothing updates until form is submitted
                            # Create unique key based on response to prevent conflicts
                            response_id = str(hash(str(response)))[:10] if response else "default"
                            form_key = f"rlhf_form_{response_id}"
                            
                            with st.form(key=form_key, clear_on_submit=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    rlhf_overall = st.slider(
                                        "Overall Rating (1-10)",
                                        min_value=1,
                                        max_value=10,
                                        value=st.session_state.get(f"rlhf_overall_{response_id}", 7),
                                        step=1,
                                        key=f"rlhf_overall_{response_id}",
                                        help="Your overall judgment"
                                    )
                                    rlhf_speech_act = st.slider(
                                        "Speech Act Alignment (1-10)",
                                        min_value=1,
                                        max_value=10,
                                        value=st.session_state.get(f"rlhf_speech_act_{response_id}", 8),
                                        step=1,
                                        key=f"rlhf_speech_act_{response_id}",
                                        help="Does response match query's illocutionary force?"
                                    )
                                    rlhf_cultural = st.slider(
                                        "Cultural Appropriateness (1-10)",
                                        min_value=1,
                                        max_value=10,
                                        value=st.session_state.get(f"rlhf_cultural_{response_id}", 9),
                                        step=1,
                                        key=f"rlhf_cultural_{response_id}",
                                        help="Did it preserve cultural terms (e.g., 潜规则)?"
                                    )
                                with col2:
                                    rlhf_pragmatic = st.slider(
                                        "Pragmatic Coherence (1-10)",
                                        min_value=1,
                                        max_value=10,
                                        value=st.session_state.get(f"rlhf_pragmatic_{response_id}", 7),
                                        step=1,
                                        key=f"rlhf_pragmatic_{response_id}",
                                        help="Did it obey Grice's Maxims?"
                                    )
                                    rlhf_clarity = st.slider(
                                        "Syntactic Clarity (1-10)",
                                        min_value=1,
                                        max_value=10,
                                        value=st.session_state.get(f"rlhf_clarity_{response_id}", 6),
                                        step=1,
                                        key=f"rlhf_clarity_{response_id}",
                                        help="Is parse tree depth appropriate?"
                                    )
                                
                                # Form submit button - only this triggers processing
                                submitted = st.form_submit_button("💾 Save RLHF Rating", type="primary", use_container_width=True)
                            
                            # Only process when form is submitted (not on slider movement)
                            if submitted:
                                # Store full annotation
                                from backend.cognitive_architecture.agents.rlhf_critic import Agent_RLHF_Critic
                                critic = Agent_RLHF_Critic()
                                response_text = str(response.get("english_response") or response.get("analysis") or response)
                                current_goal = goal_memory.get_goal_context() or ""
                                critic._store_annotation(
                                    query or "Analyze the uploaded media.",
                                    response_text,
                                    rlhf_overall / 10.0,  # Normalize to 0-1
                                    rlhf_pragmatic / 10.0,
                                    rlhf_cultural / 10.0,
                                    rlhf_clarity / 10.0,
                                    rlhf_speech_act / 10.0,
                                    0,  # syntactic_complexity (can be computed later)
                                    "assertive",  # speech_act_type
                                    current_goal
                                )
                                # Store slider values in session state for persistence
                                st.session_state[f"rlhf_overall_{response_id}"] = rlhf_overall
                                st.session_state[f"rlhf_speech_act_{response_id}"] = rlhf_speech_act
                                st.session_state[f"rlhf_cultural_{response_id}"] = rlhf_cultural
                                st.session_state[f"rlhf_pragmatic_{response_id}"] = rlhf_pragmatic
                                st.session_state[f"rlhf_clarity_{response_id}"] = rlhf_clarity
                                
                                # Store evaluation in session state ONLY after form is submitted
                                st.session_state['last_evaluation'] = {
                                    "scores": {
                                        "overall_rating": rlhf_overall,
                                        "pragmatic_coherence": rlhf_pragmatic / 10.0,
                                        "cultural_appropriateness": rlhf_cultural / 10.0,
                                        "syntactic_clarity": rlhf_clarity / 10.0,
                                        "speech_act_alignment": rlhf_speech_act / 10.0
                                    }
                                }
                                # Mark that save was clicked (prevents auto-update on slider movement)
                                st.session_state['rlhf_saved'] = True
                                st.session_state['rlhf_saved_id'] = response_id
                                st.success(f"✅ Saved RLHF rating: Overall={rlhf_overall}/10, Speech Act={rlhf_speech_act}/10, Cultural={rlhf_cultural}/10, Pragmatic={rlhf_pragmatic}/10, Clarity={rlhf_clarity}/10")
                                st.rerun()
                        
                    except Exception as e:
                        sys.stdout = old_stdout  # Restore stdout on error
                        st.error(f"❌ Error: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language="python")
                    finally:
                        # Restore stdout
                        if 'old_stdout' in locals():
                            sys.stdout = old_stdout
                        # Cleanup temp files
                        if image_path and os.path.exists(image_path):
                            os.unlink(image_path)
                        if video_path and os.path.exists(video_path):
                            os.unlink(video_path)

# ============================================================================
# RIGHT COLUMN: METRICS & MEMORY
# ============================================================================

with col_right:
    st.subheader("📊 Metrics")
    
    # RLHF Spider Chart (ONLY show after Save button is clicked - form submission prevents auto-update)
    # Use a separate key to track if save was clicked
    if 'rlhf_saved' in st.session_state and st.session_state.get('rlhf_saved', False):
        if 'last_evaluation' in st.session_state:
            evaluation = st.session_state['last_evaluation']
            scores = evaluation.get("scores", {})
            if scores and any(v > 0 for v in scores.values() if isinstance(v, (int, float))):
                st.subheader("📈 RLHF Multi-Category Score")
                fig = create_rlhf_spider_chart(scores)
                st.pyplot(fig, use_container_width=True)
                
                # Detailed scores
                st.markdown("**Detailed Scores:**")
                st.json({
                    "Overall Rating": f"{scores.get('overall_rating', 0):.1f}/10",
                    "Pragmatic Coherence": f"{scores.get('pragmatic_coherence', 0):.2f}",
                    "Cultural Appropriateness": f"{scores.get('cultural_appropriateness', 0):.2f}",
                    "Syntactic Clarity": f"{scores.get('syntactic_clarity', 0):.2f}",
                    "Speech Act Alignment": f"{scores.get('speech_act_alignment', 0):.2f}"
                })
    
    # Message History (scrollable)
    st.subheader("📜 Message History")
    if 'message_history' in st.session_state:
        messages = st.session_state['message_history'].get_messages(limit=50)
        if messages:
            # Display in reverse chronological order (newest first)
            for msg in reversed(messages[-20:]):  # Show last 20
                timestamp = msg.get('timestamp', '')
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                metadata = msg.get('metadata', {})
                
                # Format timestamp
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = timestamp[:19] if len(timestamp) > 19 else timestamp
                
                # Color code by role
                if role == "user":
                    st.markdown(f'<div style="color: #00bcd4; font-family: monospace; font-size: 0.9em; margin: 5px 0;">[{time_str}] <strong>User:</strong> {content[:100]}{"..." if len(content) > 100 else ""}</div>', unsafe_allow_html=True)
                elif role == "assistant":
                    agent = metadata.get('agent', 'unknown')
                    st.markdown(f'<div style="color: #ff9800; font-family: monospace; font-size: 0.9em; margin: 5px 0;">[{time_str}] <strong>{agent}:</strong> {content[:100]}{"..." if len(content) > 100 else ""}</div>', unsafe_allow_html=True)
        else:
            st.info("No messages yet")
    else:
        st.info("Message history not initialized")
    
    # Memory Stream
    st.subheader("🧠 Episodic Memory Stream")
    memory_df = get_memory_stream(limit=20)  # Increased limit
    if not memory_df.empty:
        # Format for better display
        display_df = memory_df.copy()
        
        # Truncate long text for display
        if 'query' in display_df.columns:
            display_df['query'] = display_df['query'].apply(lambda x: (x[:80] + '...') if isinstance(x, str) and len(x) > 80 else x)
        if 'response' in display_df.columns:
            display_df['response'] = display_df['response'].apply(lambda x: (x[:100] + '...') if isinstance(x, str) and len(x) > 100 else x)
        
        # Select columns to display
        cols_to_show = []
        if 'query' in display_df.columns:
            cols_to_show.append('query')
        if 'response' in display_df.columns:
            cols_to_show.append('response')
        if 'rating' in display_df.columns:
            cols_to_show.append('rating')
        if 'timestamp' in display_df.columns:
            cols_to_show.append('timestamp')
        
        if cols_to_show:
            st.dataframe(
                display_df[cols_to_show].head(20),
                use_container_width=True,
                hide_index=True
            )
            # Show count
            st.caption(f"📊 Showing {len(display_df)} entries (sorted by {'rating' if 'rating' in display_df.columns else 'timestamp'})")
        else:
            st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)
    else:
        # Show helpful message
        st.info("💡 **No episodic memory entries yet.**\n\nMemory entries are created when the Memory Curator agent processes interactions. Try running a query to see episodic memory in action!")
    
    # RLHF Annotations
    st.subheader("⭐ RLHF Annotations")
    rlhf_df = get_rlhf_annotations(limit=10)
    if not rlhf_df.empty:
        st.dataframe(
            rlhf_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No RLHF annotations yet")

# Footer
st.markdown("---")
st.caption("Pragmatic Orchestrator | Pragma-Logos: Where Speech Acts Meet Software | Built with Searle (1969), Grice (1975)")

