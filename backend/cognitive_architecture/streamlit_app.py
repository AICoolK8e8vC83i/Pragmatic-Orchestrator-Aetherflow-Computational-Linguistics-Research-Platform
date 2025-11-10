"""
Streamlit Dashboard for Cognitive Architecture
Allows user to:
- Enter queries
- Set goals
- Rate responses (1-10)
- View evaluation scores
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import process_with_proactivity, goal_memory, TEST_QUERIES

st.set_page_config(page_title="Cognitive Architecture", page_icon="🧠", layout="wide")

st.title("🧠 Cognitive Architecture: Pragmatic Inference Engine")
st.markdown("**Theory**: Searle's Speech Act Theory + Grice's Maxims")

# Sidebar for goal management
with st.sidebar:
    st.header("🎯 Goal Memory")
    current_goal = goal_memory.get_current_goal()
    new_goal = st.text_area("Current Goal", value=current_goal, height=100)
    if st.button("Update Goal"):
        goal_memory.set_goal(new_goal)
        st.success("Goal updated!")
        st.rerun()
    
    st.markdown("---")
    st.header("📝 Test Queries")
    selected_test = st.selectbox("Select test query", ["None"] + list(TEST_QUERIES.keys()))
    if selected_test != "None":
        st.text_area("Query", value=TEST_QUERIES[selected_test], height=100, key="test_query")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Query Input")
    
    # Query input
    query = st.text_area(
        "Enter your query (or select test query from sidebar)",
        height=150,
        key="main_query"
    )
    
    # Multimodal input options
    col_audio, col_image = st.columns(2)
    
    with col_audio:
        audio_file = st.file_uploader("Upload audio (MP3, WAV)", type=["mp3", "wav"])
    
    with col_image:
        image_file = st.file_uploader("Upload image/video", type=["jpg", "jpeg", "png", "gif", "webp", "mp4", "avi", "mov"])
    
    image_path = None
    image_base64 = None
    
    if image_file:
        # Save uploaded image temporarily
        import tempfile
        import base64
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.name)[1]) as tmp_file:
            tmp_file.write(image_file.read())
            image_path = tmp_file.name
        
        # Also encode to base64
        image_file.seek(0)
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        st.success(f"✅ Image uploaded: {image_file.name}")
        
        if not query:
            query = "Describe what's happening in this image in detail."
    
    if audio_file:
        with st.spinner("Transcribing audio..."):
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name
            
            try:
                from main import transcribe_audio
                transcribed = transcribe_audio(tmp_path)
                if transcribed:
                    query = transcribed
                    st.success(f"Transcribed: {transcribed}")
                else:
                    st.error("Failed to transcribe audio")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)

with col2:
    st.header("⭐ Rating")
    overall_rating = st.slider(
        "Rate response quality (1-10)",
        min_value=1,
        max_value=10,
        value=7,
        step=1
    )
    st.caption("This rating will be used for RLHF evaluation")

# Process button
if st.button("🚀 Process Query", type="primary"):
    if not query:
        st.warning("Please enter a query or select a test query")
    else:
        with st.spinner("Processing query through pragmatic router..."):
            try:
                # Get image path if uploaded
                img_path = None
                img_base64 = None
                if image_file:
                    img_path = image_path
                    img_base64 = image_base64
                
                response, routing, explanation, evaluation, ai_status = process_with_proactivity(
                    query, 
                    max_iter=3,  # x3 proactive iterations
                    overall_rating=overall_rating,
                    image_path=img_path,
                    image_base64=img_base64
                )
                
                # Display results
                st.header("📊 Results")
                
                # AI Status
                st.subheader("🤖 AI Call Status")
                routing_success = sum(ai_status["routing_ai_success"]) / len(ai_status["routing_ai_success"]) * 100 if ai_status["routing_ai_success"] else 0
                agent_success = sum(ai_status["agent_ai_success"]) / len(ai_status["agent_ai_success"]) * 100 if ai_status["agent_ai_success"] else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Routing AI", f"{routing_success:.0f}%", 
                             "✅" if routing_success > 50 else "⚠️")
                with col2:
                    st.metric("Agent AI", f"{agent_success:.0f}%",
                             "✅" if agent_success > 50 else "⚠️")
                with col3:
                    st.metric("Iterations", ai_status["total_iterations"])
                
                # Routing info
                st.subheader("🔀 Routing Decision")
                if routing.get('ai_success'):
                    st.success(f"✅ AI Routing (Qwen 3 1.7B)")
                else:
                    st.warning(f"⚠️ Fallback Routing (keyword matching)")
                
                st.info(f"**Agent**: {routing['agent']}")
                st.caption(f"**Reasoning**: {routing['reasoning']}")
                st.caption(f"**Explanation**: {explanation}")
                
                # Evaluation scores
                st.subheader("📈 Evaluation Scores")
                scores = evaluation.get("scores", {})
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Overall", f"{scores.get('overall_rating', 0):.1f}/10")
                with col_b:
                    st.metric("Pragmatic", f"{scores.get('pragmatic_coherence', 0):.2f}")
                with col_c:
                    st.metric("Cultural", f"{scores.get('cultural_appropriateness', 0):.2f}")
                with col_d:
                    st.metric("Syntactic", f"{scores.get('syntactic_clarity', 0):.2f}")
                
                # Response
                st.subheader("📄 Response")
                if isinstance(response, dict):
                    if response.get("ai_success"):
                        st.success("✅ AI Response (not fallback)")
                    else:
                        st.warning("⚠️ Fallback Response")
                    for key, value in response.items():
                        if key not in ["ai_success", "raw_response", "error"]:
                            st.markdown(f"**{key}**: {value}")
                    if "error" in response:
                        st.error(f"Error: {response['error']}")
                else:
                    st.markdown(response)
                
                st.success("✅ Query processed successfully!")
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("Built with Searle's Speech Act Theory, Grice's Maxims, and computational linguistics")

