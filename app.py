import os
import json
import shutil
import logging
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from agents.nodes import (
    MeetingState,
    transcription_node,
    summary_concept_node,
    task_milestone_node,
    mcp_export_node,
    get_whisper,
    llm
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("FastAPI")

# =====================================================================
# 🕸️ LangGraph Multi-Agent Workflow (Parallel Fan-Out)
# =====================================================================
graph_builder = StateGraph(MeetingState)

graph_builder.add_node("transcriber", transcription_node)
graph_builder.add_node("summarizer", summary_concept_node)
graph_builder.add_node("task_planner", task_milestone_node)
graph_builder.add_node("mcp_exporter", mcp_export_node)

graph_builder.set_entry_point("transcriber")

# Parallel Fan-Out: Both Summarizer & Task Planner run simultaneously
graph_builder.add_edge("transcriber", "summarizer")
graph_builder.add_edge("transcriber", "task_planner")

# Fan-In: Re-converge before exporting
graph_builder.add_edge("summarizer", "mcp_exporter")
graph_builder.add_edge("task_planner", "mcp_exporter")
graph_builder.add_edge("mcp_exporter", END)

app_graph = graph_builder.compile()

# =====================================================================
# 🚀 FastAPI Lifespan & App Setup
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_whisper()
    try:
        await llm.ainvoke("Warmup ping")
    except Exception as e:
        logger.warning(f"⚠️ Ollama warm-up ping failed: {e}")
    yield

app = FastAPI(title="AI Meeting Assistant", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    question: str
    transcript: str

# =====================================================================
# 🌐 UI & API Endpoints
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Formatted UI with marked.js, SSE progress, and Chat Interface."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Meeting Assistant</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1e293b; background: #f8fafc; line-height: 1.6; }
            .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 28px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 24px; }
            .input-group { margin: 20px 0; display: flex; gap: 12px; align-items: center; }
            button { background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
            button:hover { background: #1d4ed8; }
            button:disabled { background: #94a3b8; cursor: not-allowed; }
            .btn-secondary { background: #475569; margin-right: 8px; }
            .btn-secondary:hover { background: #334155; }
            #status-log { margin-top: 15px; font-family: monospace; background: #1e293b; color: #10b981; padding: 16px; border-radius: 8px; font-size: 14px; }
            .result-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-top: 8px; }
            .chat-box { display: flex; gap: 10px; margin-top: 15px; }
            .chat-input { flex-grow: 1; padding: 10px; border: 1px solid #cbd5e1; border-radius: 6px; }
            .hidden { display: none; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🎙️ AI Meeting Assistant</h2>
            <div class="input-group">
                <input type="file" id="fileInput" accept=".mp3,.wav,.m4a,.txt" />
                <button id="uploadBtn" onclick="processFile()">Process Meeting</button>
            </div>
            <div id="status-log" class="hidden"></div>
        </div>

        <div id="resultsContainer" class="card hidden">
            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                <span style="background: #dcfce7; color: #15803d; padding: 4px 12px; border-radius: 12px; font-weight: 600;">✅ Processing Complete</span>
                <div>
                    <button class="btn-secondary" onclick="copySummary()">📋 Copy Summary</button>
                    <button onclick="downloadMarkdown()">💾 Download .md</button>
                </div>
            </div>
            
            <h3>📌 Executive Summary</h3>
            <div id="summaryBox" class="result-box"></div>

            <h3>🎯 Tasks & Milestones</h3>
            <div id="tasksBox" class="result-box"></div>
            
            <p style="color: #64748b; font-size: 14px; margin-top: 10px;" id="mcpBox"></p>

            <hr style="margin: 30px 0; border: 0; border-top: 1px solid #e2e8f0;">

            <!-- RAG Chat Feature -->
            <h3>💬 Chat with this Meeting</h3>
            <p style="font-size: 14px; color: #64748b;">Ask specific follow-up questions about the transcript.</p>
            <div id="chatHistory" class="result-box hidden"></div>
            <div class="chat-box">
                <input type="text" id="chatInput" class="chat-input" placeholder="e.g., What did Alex say about the budget?" onkeypress="handleEnter(event)">
                <button onclick="askQuestion()" id="chatBtn">Ask</button>
            </div>
        </div>

        <script>
            let rawSummary = "", rawTasks = "", rawTranscript = "", exportFile = "";

            async function processFile() {
                const fileInput = document.getElementById('fileInput');
                const btn = document.getElementById('uploadBtn');
                const log = document.getElementById('status-log');
                
                if (!fileInput.files[0]) return alert("Please select a file.");

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                btn.disabled = true;
                log.classList.remove('hidden');
                document.getElementById('resultsContainer').classList.add('hidden');
                log.innerHTML = "⏳ Uploading & Initializing LangGraph Pipeline...<br>";

                try {
                    const response = await fetch("/api/process_stream", { method: "POST", body: formData });
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\\n\\n');
                        buffer = lines.pop();

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = JSON.parse(line.replace('data: ', ''));
                                
                                if (data.type === 'progress') {
                                    log.innerHTML += `> ${data.msg}<br>`;
                                } else if (data.type === 'complete') {
                                    log.innerHTML += `> 🏁 Workflow Complete!<br>`;
                                    
                                    rawSummary = data.summary;
                                    rawTasks = data.tasks;
                                    rawTranscript = data.transcript;
                                    exportFile = data.export_filename;

                                    document.getElementById('summaryBox').innerHTML = marked.parse(rawSummary);
                                    document.getElementById('tasksBox').innerHTML = marked.parse(rawTasks);
                                    document.getElementById('mcpBox').innerText = data.mcp;
                                    
                                    document.getElementById('resultsContainer').classList.remove('hidden');
                                }
                            }
                        }
                    }
                } catch (err) {
                    log.innerHTML += `<span style="color: #ef4444;">❌ Error: ${err.message}</span>`;
                } finally {
                    btn.disabled = false;
                }
            }

            function copySummary() {
                navigator.clipboard.writeText(`# Summary\\n${rawSummary}\\n\\n# Tasks\\n${rawTasks}`);
                alert("Copied to clipboard!");
            }

            function downloadMarkdown() {
                const blob = new Blob([`# Digest\\n\\n## Summary\\n${rawSummary}\\n\\n## Tasks\\n${rawTasks}`], { type: 'text/markdown' });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = exportFile || "meeting.md";
                a.click();
            }

            // --- RAG Chat Logic ---
            function handleEnter(e) { if (e.key === 'Enter') askQuestion(); }
            
            async function askQuestion() {
                const input = document.getElementById('chatInput');
                const btn = document.getElementById('chatBtn');
                const history = document.getElementById('chatHistory');
                const q = input.value.trim();
                
                if(!q || !rawTranscript) return;

                history.classList.remove('hidden');
                history.innerHTML += `<b>You:</b> ${q}<br>`;
                input.value = "";
                btn.disabled = true;

                try {
                    const res = await fetch("/api/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question: q, transcript: rawTranscript })
                    });
                    const data = await res.json();
                    history.innerHTML += `<b>AI:</b> ${marked.parseInline(data.answer)}<br><hr style="border:0; border-top: 1px solid #e2e8f0; margin:10px 0;">`;
                } catch (e) {
                    history.innerHTML += `<span style="color:red">Error: ${e.message}</span><br>`;
                } finally {
                    btn.disabled = false;
                    history.scrollTop = history.scrollHeight;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/process_stream")
async def process_meeting_stream(file: UploadFile = File(...)):
    """Handles file uploads and streams execution updates (SSE) to unblock event loop."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ext = os.path.splitext(file.filename)[1].lower()
    initial_state = {
        "audio_path": file_path if ext in [".mp3", ".wav", ".m4a"] else None,
        "transcript_path": file_path if ext in [".txt", ".md"] else None,
        "transcript": "", "summary": "", "tasks_and_milestones": "", "mcp_status": "", "export_filename": ""
    }

    async def event_generator():
        final_state = {}
        try:
            # LangGraph native unblocked async streaming
            async for chunk in app_graph.astream(initial_state):
                for node_name, state_update in chunk.items():
                    final_state.update(state_update)
                    
                    if node_name == "transcriber":
                        msg = "🟢 Step 1/3: Audio/Text transcribed successfully."
                    elif node_name in ["summarizer", "task_planner"]:
                        msg = f"🟡 Step 2/3: Parallel branch finished [{node_name}]."
                    elif node_name == "mcp_exporter":
                        msg = "🟢 Step 3/3: Exported markdown digest file."
                    else:
                        msg = f"Node [{node_name}] executed."

                    yield f"data: {json.dumps({'type': 'progress', 'msg': msg})}\n\n"

            yield f"data: {json.dumps({'type': 'complete', 'summary': final_state.get('summary', ''), 'tasks': final_state.get('tasks_and_milestones', ''), 'mcp': final_state.get('mcp_status', ''), 'transcript': final_state.get('transcript', ''), 'export_filename': final_state.get('export_filename', 'meeting.md')})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'msg': str(e)})}\n\n"
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/chat")
async def chat_with_meeting(req: ChatRequest):
    """RAG Endpoint: Answer follow-up questions without re-running the graph."""
    prompt = ChatPromptTemplate.from_template(
        "You are an AI meeting assistant. Answer the user's question based strictly on the following transcript.\n\n"
        "TRANSCRIPT:\n{transcript}\n\n"
        "QUESTION: {question}\n\n"
        "ANSWER:"
    )
    chain = prompt | llm
    res = await chain.ainvoke({"transcript": req.transcript, "question": req.question})
    return {"answer": res.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)