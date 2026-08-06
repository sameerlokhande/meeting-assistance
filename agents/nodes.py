import os
import sys
import time
import logging
import asyncio
from datetime import datetime
from typing import TypedDict, List, Optional

# =====================================================================
# 🛠️ FFmpeg Auto-Fix for Windows
# =====================================================================
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    ffmpeg_target = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_target) and os.path.exists(ffmpeg_exe):
        import shutil
        shutil.copy(ffmpeg_exe, ffmpeg_target)
except Exception:
    pass

from faster_whisper import WhisperModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger("MeetingAgents")
logger.setLevel(logging.INFO)

# =====================================================================
# 📋 State Definition
# =====================================================================
class MeetingState(TypedDict):
    audio_path: Optional[str]
    transcript_path: Optional[str]
    transcript: str
    summary: str
    key_concepts: List[str]
    tasks_and_milestones: str
    mcp_status: str
    export_filename: str  # Added for unique output naming

# =====================================================================
# ⚡ Model Singletons & Optimization Config
# =====================================================================
llm = ChatOllama(
    model="qwen2.5:0.5b",
    temperature=0.1,
    num_thread=os.cpu_count() or 4,
    timeout=60.0
)

_whisper_model = None

def get_whisper() -> WhisperModel:
    """Lazy loader and singleton for faster-whisper (INT8 on CPU)."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("[Agent: Transcriber] ⏳ Loading faster-whisper (tiny, int8)...")
        # Optimization: Swapped openai-whisper for faster-whisper CTranslate2
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("[Agent: Transcriber] ✅ faster-whisper model loaded.")
    return _whisper_model

# =====================================================================
# 🧩 Line-Based Overlapping Text Splitter
# =====================================================================
def chunk_text(text: str, max_chars: int = 8000, overlap_chars: int = 800) -> List[str]:
    """Splits long transcripts using line-aware sliding window with context overlap."""
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        if current_length + len(line) > max_chars and current_chunk:
            full_chunk_text = "".join(current_chunk).strip()
            if full_chunk_text:
                chunks.append(full_chunk_text)
            
            overlap_chunk = []
            overlap_len = 0
            for prev_line in reversed(current_chunk):
                if overlap_len + len(prev_line) <= overlap_chars:
                    overlap_chunk.insert(0, prev_line)
                    overlap_len += len(prev_line)
                else:
                    break
            
            current_chunk = overlap_chunk
            current_length = overlap_len

        current_chunk.append(line)
        current_length += len(line)

    if current_chunk:
        final_text = "".join(current_chunk).strip()
        if final_text:
            chunks.append(final_text)

    return chunks

# =====================================================================
# 🤖 Agent Nodes
# =====================================================================

async def transcription_node(state: MeetingState) -> dict:
    """Agent 1: Reads text file or transcribes audio using faster-whisper."""
    start_time = time.time()
    logger.info("[Node: transcription_node] 🚀 Execution started")
    
    transcript = ""
    if state.get("transcript_path"):
        try:
            with open(state["transcript_path"], "r", encoding="utf-8") as f:
                transcript = f.read()
        except Exception as err:
            transcript = f"Transcription error: Could not read file - {err}"

    elif state.get("audio_path"):
        try:
            whisper_instance = get_whisper()
            segments, info = whisper_instance.transcribe(state["audio_path"], beam_size=1)
            transcript = " ".join([segment.text for segment in segments]).strip()
        except Exception as err:
            transcript = f"Transcription error: {err}"
    else:
        transcript = "Transcription error: No valid input provided."

    return {"transcript": transcript}


async def summarize_chunk_async(chain, chunk: str, idx: int, total: int) -> str:
    res = await chain.ainvoke({"text_chunk": chunk})
    return res.content


async def summary_concept_node(state: MeetingState) -> dict:
    """Agent 2: Async MapReduce Summarizer."""
    transcript = state.get("transcript", "")
    if not transcript or transcript.startswith("Transcription error"):
        return {"summary": "No valid transcript available."}

    chunks = chunk_text(transcript, max_chars=8000, overlap_chars=800)

    map_prompt = ChatPromptTemplate.from_template(
        "Summarize the key points, topics, and decisions from this portion of the transcript:\n\n{text_chunk}\n\nSummary:"
    )
    map_chain = map_prompt | llm

    tasks = [summarize_chunk_async(map_chain, chunk, i, len(chunks)) for i, chunk in enumerate(chunks)]
    chunk_summaries = await asyncio.gather(*tasks)

    if len(chunk_summaries) > 1:
        reduce_prompt = ChatPromptTemplate.from_template(
            "Synthesize these section summaries into a single, cohesive Executive Summary (3-5 bullet points):\n\n{combined}\n\nExecutive Summary:"
        )
        reduce_chain = reduce_prompt | llm
        combined_text = "\n\n---\n\n".join(chunk_summaries)
        final_res = await reduce_chain.ainvoke({"combined": combined_text})
        summary_result = final_res.content
    else:
        summary_result = chunk_summaries[0]

    return {"summary": summary_result}


async def task_milestone_node(state: MeetingState) -> dict:
    """Agent 3: Actionable Task & Milestone Planner."""
    transcript = state.get("transcript", "")
    if not transcript or transcript.startswith("Transcription error"):
        return {"tasks_and_milestones": "No valid transcript available."}

    context_text = transcript[:12000] if len(transcript) > 12000 else transcript

    prompt = ChatPromptTemplate.from_template(
        "Analyze this transcript and extract key actionable tasks and milestones:\n\n{context}\n\n"
        "Output format:\n"
        "### 🎯 Key Milestones & Action Items\n"
        "- **Milestone**: Description\n"
        "  - [ ] Action item"
    )
    chain = prompt | llm
    res = await chain.ainvoke({"context": context_text})

    return {"tasks_and_milestones": res.content}


async def mcp_export_node(state: MeetingState) -> dict:
    """Agent 4: Markdown Exporter with Unique Naming."""
    # Optimization: Unique Output Naming to prevent overwrites
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"meeting_{timestamp}.md"
    
    content = (
        f"# Meeting Digest ({timestamp})\n\n"
        f"## Summary\n{state.get('summary', 'N/A')}\n\n"
        f"## Action Items & Milestones\n{state.get('tasks_and_milestones', 'N/A')}\n"
    )
    
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, export_filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    mcp_status = f"Exported successfully to {file_path}"
    return {"mcp_status": mcp_status, "export_filename": export_filename}