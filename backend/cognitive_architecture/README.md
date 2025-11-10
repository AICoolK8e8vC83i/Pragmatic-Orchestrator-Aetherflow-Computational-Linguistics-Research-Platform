# Pragmatic Orchestrator: Computational Linguistics Research Platform

> **Pragma-Logos: Where Speech Acts Meet Software**

A multi-agent AI system that operationalizes computational linguistics theory through pragmatic inference, multimodal analysis, and human feedback learning. Watch the [demo video](https://youtube.com/watch?v=YOUR_VIDEO_ID) to see it in action.

---

## 🎯 What Is This?

The Pragmatic Orchestrator isn't a chatbot wrapper—it's a research platform that applies speech act theory, pragmatic inference, and sociolinguistic analysis to AI agent routing. Each agent is grounded in established computational linguistics frameworks, making this a practical implementation of theoretical research.

**Key Features:**
- 🧠 **Pragmatic Routing**: Uses Searle's Speech Act Theory and Grice's Maxims to intelligently route queries
- 🖼️ **Multimodal Analysis**: Combines Google Cloud Vision API (OCR) with Gemini 2.5 Flash for detailed visual semantics
- 🔄 **Proactive Intelligence Loop**: 2-iteration refinement that chains context across responses
- 📊 **Human Feedback Learning**: Manual RLHF rating system with 5-category evaluation
- 💾 **Episodic Memory**: Persistent storage with Frame Semantics-based semantic role labeling

---

## 🧠 Theoretical Foundation

Each agent implements specific computational linguistics frameworks:

| Agent | Theory | Model | Service |
|-------|--------|-------|---------|
| **Pragmatic Router** | Searle (1969) + Grice (1975) | Qwen 3 1.7B | Ollama (Local) |
| **Code Switcher** | Myers-Scotton (1993) + Relevance Theory | Kimi K2 Thinking | Moonshot AI |
| **Discourse Analyzer** | RST (Mann & Thompson 1988) | Kimi K2 Thinking | Moonshot AI |
| **Semantic Parser** | Montague Semantics (1973) | Claude Sonnet 4.5 | AWS Bedrock |
| **Vision Analyzer** | Multimodal Pragmatics | Gemini 2.5 Flash + Cloud Vision API | Google AI + GCP |
| **Memory Curator** | Frame Semantics (Fillmore) | Claude Haiku 4.5 | AWS Bedrock |
| **RLAIF Critic** | Preference Learning | Qwen 3 1.7B | Ollama (Local) |
| **Explainability Bridge** | Abductive Reasoning | Claude Haiku 4.5 | AWS Bedrock |
| | | | *Note: Not in dropdown menu, automatically called by architecture* |

### Theoretical Citations

- **Searle, J. R. (1969)**: *Speech Acts: An Essay in the Philosophy of Language* — Classification of speech acts for routing decisions
- **Grice, H. P. (1975)**: Logic and conversation — Quality, Quantity, Relevance, and Manner maxims for response evaluation
- **Myers-Scotton, C. (1993)**: *Duelling Languages: Grammatical Structure in Codeswitching* — Matrix Language Frame Model for bilingual analysis
- **Sperber, D., & Wilson, D. (1986)**: *Relevance: Communication and Cognition* — Pragmatic inference for cultural term preservation
- **Mann, W. C., & Thompson, S. A. (1988)**: Rhetorical Structure Theory — Discourse coherence analysis
- **Montague, R. (1973)**: The proper treatment of quantification in ordinary English — Natural language to formal logic conversion
- **Fillmore, C. J.**: Frame Semantics — Semantic role labeling for episodic memory

---

## 🚀 Setup Guide

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- API accounts for:
  - AWS Bedrock (Claude models)
  - Google Cloud Platform (Gemini + Cloud Vision API)
  - Moonshot AI (Kimi K2)

### Step 1: Install Dependencies

```bash
# Clone repository
git clone <your-repo-url>
cd backend/cognitive_architecture

# Install Python packages
pip install -r requirements.txt
```

### Step 2: Set Up Ollama

```bash
# Start Ollama service
ollama serve

# Pull required models (in separate terminal)
ollama pull qwen3:1.7b      # For pragmatic routing
ollama pull gemma3:1b        # For parsing Qwen output
```

### Step 3: Configure API Keys

Create a `.env` file in the project root:

```bash
# AWS Bedrock (Claude models)
AWS_BEARER_TOKEN_BEDROCK=your_bearer_token_here
AWS_REGION=us-east-1

# Google AI API (Gemini 2.5 Flash)
GEMINI_API_KEY=your_gemini_api_key_here

# Moonshot AI (Kimi K2 Thinking)
MOONSHOT_API_KEY=your_moonshot_api_key_here

# Google Cloud Vision API (set via environment variable after downloading JSON)
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

#### Getting API Keys

**AWS Bedrock:**
1. Go to [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Navigate to "Model access" and enable Claude Sonnet 4.5 and Claude Haiku 4.5
3. Go to "API keys" or use IAM credentials
4. Create a bearer token or use AWS CLI to generate credentials
5. Copy the bearer token to `AWS_BEARER_TOKEN_BEDROCK` in `.env`

**Google AI API (Gemini):**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the API key to `GEMINI_API_KEY` in `.env`

**Moonshot AI:**
1. Go to [Moonshot AI Console](https://platform.moonshot.cn/)
2. Navigate to API keys section
3. Create a new API key
4. Copy to `MOONSHOT_API_KEY` in `.env`

**Google Cloud Vision API:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable "Cloud Vision API":
   - Navigate to "APIs & Services" > "Library"
   - Search for "Cloud Vision API"
   - Click "Enable"
4. Create Service Account:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Fill in details and create
5. Create and Download JSON Key:
   - Click on the service account you created
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Select "JSON" format
   - Download the JSON file (e.g., `your-project-credentials.json`)
6. Set Environment Variable:
   ```bash
   # Option 1: Export in your shell
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-project-credentials.json"
   
   # Option 2: Add to .env file (if your system supports it)
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-project-credentials.json
   
   # Option 3: Place JSON file in project root with keywords in filename
   # The system will auto-detect files with 'google', 'vision', 'gcp', 'cloud', 'service', or 'account' in the name
   ```
7. Place the JSON file in the project root (the system will auto-detect it if named appropriately)

**Note:** The system will automatically search for JSON files in the project root with keywords like `google`, `vision`, `gcp`, `cloud`, `service`, or `account` in the filename. If found, it will automatically set `GOOGLE_APPLICATION_CREDENTIALS`.

### Step 4: Verify Setup

```bash
# Test Ollama
ollama list  # Should show qwen3:1.7b and gemma3:1b

# Test Python installation
python -c "import google.generativeai; import google.cloud.vision; print('✅ All imports successful')"
```

---

## 🎮 Usage

### Streamlit Dashboard (Recommended)

```bash
streamlit run streamlit_dashboard.py
```

The dashboard provides:
- Interactive query input (text, image, audio, video)
- Live execution logging with agent badges
- RLHF rating system with spider chart visualization
- Episodic memory stream display
- Message history

### Command Line Interface

```bash
python main.py
```

**Input Options:**
- `voice /path/to/audio.mp3` — Transcribe and process audio
- `image /path/to/image.jpg` — Analyze image/video
- `test bilingual` — Use predefined test queries
- Direct text query — Type your query

### Programmatic Usage

```python
from main import process_with_proactivity

response, routing, explanation, evaluation, ai_status = process_with_proactivity(
    query="我需要讨论潜规则问题",
    max_iter=2
)

print(f"Selected agent: {routing['agent']}")
print(f"Response: {response}")
```

---

## 🏗️ Architecture

### Proactive Intelligence Loop

The system implements a 2-iteration proactive loop:

1. **Routing**: Pragmatic Router (Qwen 3 1.7B) analyzes query using Searle + Grice
2. **Processing**: Specialized agent processes based on linguistic features
3. **Evaluation**: RLAIF Critic evaluates response quality
4. **Refinement**: Query refined for next iteration with previous response context
5. **Storage**: Interaction stored in episodic memory with semantic role labels

### Agent Routing Rules

**Multimodal Input:**
- **Image/Video**: Vision Analyzer (Gemini + Cloud Vision) → Code Switcher
- **Audio**: MLX Whisper transcription → Code Switcher

**Text Input:**
- Chinese terms → `code_switcher`
- "analyze discourse" → `discourse_analyzer`
- "formal logic" → `semantic_parser`
- "memory" → `memory_curator`
- "rate this" → `rlhf_critic`

**Manual Selection:** Users can bypass routing via Streamlit UI.

---

## 📊 Features

### Multimodal Processing

**Image Analysis Pipeline:**
1. **Google Cloud Vision API**: OCR (text detection) + label detection
2. **Gemini 2.5 Flash**: Detailed visual analysis using OCR data as context
   - Character analysis (pinyin, meaning, stroke order, cultural significance)
   - Color analysis (shades, gradients, symbolic meanings)
   - Symbol analysis (seals, marks, decorative elements)
   - Visual description (composition, layout, brushwork techniques)
   - Fine details (ink variations, paper texture, aging effects)

**Video Processing:**
- Frame-by-frame extraction (OpenCV)
- Each frame analyzed through vision pipeline
- Temporal coherence analysis

**Audio Processing:**
- MLX Whisper transcription
- Automatic language detection
- Routes to Code Switcher for bilingual analysis

### Manual RLHF Rating System

After response generation, rate responses across 5 categories:
- **Overall Rating** (1-10)
- **Speech Act Alignment** (1-10)
- **Syntactic Clarity** (1-10)
- **Cultural Appropriateness** (1-10)
- **Pragmatic Coherence** (1-10)

Ratings stored in SQLite and visualized as spider chart. Top 2 rated examples used as few-shot prompts.

### Episodic Memory

All interactions stored in `episodic_memory.db` with:
- Semantic role labels (ARG0, ARG1, ARGM-TMP, ARGM-LOC)
- Routing information
- Timestamps
- Query and response pairs

Memory entries are never deleted, creating a persistent research log.

### Agent Badge System

During processing, each agent displays its theoretical citation:
```
🏷️  Agent Badge: Vision Analyzer (Multimodal Pragmatics)
↓
🏷️  Agent Badge: Code Switcher (Myers-Scotton 1993 + Relevance Theory)
```

---

## 🛠️ Technology Stack

### Cloud Services

- **AWS Bedrock**
  - Claude Sonnet 4.5: Semantic parsing (Montague Semantics)
  - Claude Haiku 4.5: Memory curation, explainability

- **Google AI API**
  - Gemini 2.5 Flash: Multimodal vision analysis

- **Google Cloud Vision API**
  - OCR with pixel-level accuracy
  - Label detection for image classification
  - Safe search detection

- **Moonshot AI**
  - Kimi K2 Thinking: Code-switching, discourse analysis

### Local Inference

- **Ollama**
  - Qwen 3 1.7B: Pragmatic routing
  - Gemma 3 1B: Parser for Qwen output

### Additional Tools

- **MLX Whisper**: Audio transcription
- **OpenCV**: Video frame extraction
- **Streamlit**: Interactive web dashboard
- **SQLite**: Episodic memory and RLHF annotations
- **PIL (Pillow)**: Image processing

---

## 📁 Project Structure

```
backend/cognitive_architecture/
├── agents/
│   ├── meta_head.py              # Pragmatic Router
│   ├── code_switcher.py          # Code Switcher
│   ├── discourse_analyzer.py     # Discourse Analyzer
│   ├── semantic_parser.py       # Semantic Parser
│   ├── vision_analyzer.py        # Vision Analyzer
│   ├── rlhf_critic.py           # RLAIF Critic
│   ├── memory_curator.py         # Memory Curator
│   ├── explainability_bridge.py  # Explainability Bridge
│   └── goal_memory.py            # Goal context management
├── main.py                       # Main orchestrator
├── streamlit_dashboard.py        # Interactive UI
├── requirements.txt              # Dependencies
└── README.md                    # This file
```

---

## 🔬 Research Applications

This platform demonstrates:

1. **Operationalizing Theory**: How computational linguistics frameworks can be implemented in AI systems
2. **Multimodal Understanding**: Combining structured OCR (Cloud Vision) with deep visual semantics (Gemini)
3. **Cultural Preservation**: Sociopragmatic analysis beyond translation
4. **Human Feedback Learning**: RLHF with 5-category evaluation
5. **Pragmatic Inference**: Speech act classification for intelligent routing

---

## 📈 Evaluation Metrics

### RLHF Annotations

Stored in `rlhf_annotations.db`:
- `overall_rating` (FLOAT)
- `pragmatic_coherence` (FLOAT)
- `cultural_appropriateness` (FLOAT)
- `syntactic_clarity` (FLOAT)
- `speech_act_alignment` (FLOAT)
- `syntactic_complexity` (INT)
- `speech_act_type` (TEXT)
- `goal_context` (TEXT)

### Episodic Memory

Stored in `episodic_memory.db`:
- Semantic role labels (JSON)
- Routing information (JSON)
- Query and response pairs
- Timestamps

---

## 📚 References

- Searle, J. R. (1969). *Speech Acts: An Essay in the Philosophy of Language*. Cambridge University Press.
- Grice, H. P. (1975). Logic and conversation. In P. Cole & J. L. Morgan (Eds.), *Syntax and Semantics 3: Speech Acts* (pp. 41-58). Academic Press.
- Myers-Scotton, C. (1993). *Duelling Languages: Grammatical Structure in Codeswitching*. Oxford University Press.
- Sperber, D., & Wilson, D. (1986). *Relevance: Communication and Cognition*. Harvard University Press.
- Mann, W. C., & Thompson, S. A. (1988). Rhetorical Structure Theory: Toward a functional theory of text organization. *Text*, 8(3), 243-281.
- Montague, R. (1973). The proper treatment of quantification in ordinary English. In J. Hintikka, J. Moravcsik, & P. Suppes (Eds.), *Approaches to Natural Language* (pp. 221-242). D. Reidel.
- Fillmore, C. J. (1982). Frame Semantics. In *Linguistics in the Morning Calm* (pp. 111-137). Hanshin Publishing.

---

## 📝 License

MIT

---

## 🙏 Acknowledgments

Built for computational linguistics research, demonstrating how theoretical frameworks can be operationalized in practical AI systems.

**Pragma-Logos: Where Speech Acts Meet Software** 🧠⚙️

---

## 🎥 Demo Video

Watch the system in action: [YouTube Demo](https://youtube.com/watch?v=YOUR_VIDEO_ID)

*Note: Replace `YOUR_VIDEO_ID` with your actual YouTube video ID once uploaded.*
