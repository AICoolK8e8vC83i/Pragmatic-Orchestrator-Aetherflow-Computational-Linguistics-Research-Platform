# AETHERFLOW Setup Guide

## Quick Start

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp backend/.env.example backend/.env
```

Required:
- `OPENAI_API_KEY` - Get from https://platform.openai.com/

Optional:
- `PERPLEXITY_API_KEY` - For research agent (get from https://www.perplexity.ai/)
- `API_KEY` - Custom API key for authentication
- `OBSIDIAN_API_URL` - Default is `http://localhost:27123`

### 3. Run Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend will be available at `http://localhost:8000`

### 4. Frontend Setup

```bash
cd frontend
npm install
npm start
```

React app will start on `http://localhost:3000`

### 5. Electron Dashboard (Optional)

```bash
cd frontend
npm run electron-dev
```

## Obsidian Setup

1. Install Obsidian REST API plugin:
   - Open Obsidian
   - Settings → Community Plugins → Browse
   - Search for "REST API" plugin
   - Install and enable

2. Configure plugin:
   - Port: `27123` (default)
   - Enable CORS for localhost

3. Test connection:
   ```bash
   curl http://localhost:27123/ping
   ```

## Docker Deployment

```bash
docker-compose up -d
```

## Architecture Notes

- **Backend**: FastAPI with LangGraph orchestrator
- **Agents**: Task, Obsidian, Research
- **Memory**: Chroma vector store (local, can migrate to GCP later)
- **Frontend**: React + Electron

## Troubleshooting

### Obsidian Agent not working
- Ensure Obsidian REST API plugin is running
- Check `OBSIDIAN_API_URL` in `.env`
- Test with: `curl http://localhost:27123/ping`

### OpenAI API errors
- Verify `OPENAI_API_KEY` is set correctly
- Check API quota/balance

### Research Agent not working
- Perplexity API key is optional
- If not set, research features will be disabled

## Next Steps

1. Add more agent capabilities
2. Integrate with Google Calendar API
3. Deploy to AWS ECS/Fargate
4. Migrate Chroma to GCP Cloud Run
5. Add authentication UI

