import os
import re
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# Load .env
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env. Add OPENAI_API_KEY=... and restart.")

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
APP_TITLE = "PersonaAI = Your Identity. Your Brand."

OUTPUT_FOLDER = "generated_sites"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
INDEX_HTML_PATH = os.path.join(OUTPUT_FOLDER, "index.html")

SYSTEM_PROMPT = """
You are PersonaAI Coach — "Your Identity. Your Brand."
You help professionals elevate career identity, personal brand, and website presence.

Rules:
- Ask focused, friendly questions about role, achievements, goals.
- Use uploaded resume/profile as context but do not repeat it verbatim.
- Give actionable suggestions (bullets ok).
- Avoid inventing credentials/employers/dates.
"""

WEBSITE_SYSTEM_PROMPT = """
You are PersonaAI Website Generator.

Goal:
- Use chat history + optional resume/profile text
- Output a COMPLETE single-file professional personal website as HTML with embedded CSS
- Clean, modern, responsive, professional

Sections:
- Hero (Name, Title, Tagline)
- About
- Experience / Projects highlights
- Skills
- Contact

Rules:
- Return ONLY HTML starting with <!DOCTYPE html> and ending with </html>
- No external CSS frameworks
- Do NOT invent employers/dates/awards not present in provided content.
"""

# -------------------------
# Helpers
# -------------------------
def clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def file_to_text(file_path: str) -> str:
    """
    file_path comes from gr.File(type="filepath") and is a string path.
    Supports: .txt, .md, .pdf, .docx
    """
    if not file_path:
        return ""

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return clean_text(f.read())

        if ext == ".docx":
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text and p.text.strip()])
            return clean_text(text)

        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            pages = [(page.extract_text() or "") for page in reader.pages]
            return clean_text("\n".join(pages))

    except Exception:
        return ""

    return ""


def normalize_history(history_messages):
    """
    Gradio 6 Chatbot(type="messages") expects a list[dict] with:
      {"role": "user"|"assistant", "content": "text"}
    This function filters/normalizes to keep only valid items.
    """
    out = []
    for m in history_messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content.strip()})
    return out


def history_to_transcript(history_messages):
    lines = []
    for m in normalize_history(history_messages):
        if m["role"] == "user":
            lines.append(f"User: {m['content']}")
        else:
            lines.append(f"Assistant: {m['content']}")
    return "\n".join(lines)


# -------------------------
# OpenAI calls
# -------------------------
def persona_chat(user_message, history_messages, uploaded_file_path):
    history_messages = normalize_history(history_messages)
    file_text = file_to_text(uploaded_file_path)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if file_text:
        messages.append({
            "role": "system",
            "content": "Additional user context from uploaded resume/profile:\n" + file_text[:6000]
        })

    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_message})

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=600,
    )
    return completion.choices[0].message.content.strip()


def generate_website(history_messages, uploaded_file_path):
    history_messages = normalize_history(history_messages)

    convo_text = history_to_transcript(history_messages)
    file_text = file_to_text(uploaded_file_path)

    messages = [
        {"role": "system", "content": WEBSITE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Chat transcript:\n{convo_text}\n\nResume/Profile text:\n{file_text[:8000]}"},
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.6,
        max_tokens=2200,
    )

    html = (completion.choices[0].message.content or "").strip()

    with open(INDEX_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    return INDEX_HTML_PATH


# -------------------------
# Gradio handlers
# -------------------------
def handle_submit(user_message, history_messages, uploaded_file_path):
    history_messages = normalize_history(history_messages)

    if not user_message or not user_message.strip():
        return "", history_messages

    reply = persona_chat(user_message.strip(), history_messages, uploaded_file_path)

    history_messages.append({"role": "user", "content": user_message.strip()})
    history_messages.append({"role": "assistant", "content": reply})

    return "", history_messages


# -------------------------
# UI
# -------------------------
with gr.Blocks(title="PersonaAI") as demo:
    gr.Markdown(f"# {APP_TITLE}")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="PersonaAI Chat", height=480)
            msg = gr.Textbox(
                label="Your message",
                placeholder="Say hello or ask about career branding, resume, LinkedIn summary, website ideas...",
                lines=3,
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                gen_site_btn = gr.Button("Generate Website", variant="secondary")

        with gr.Column(scale=1):
            uploaded_file = gr.File(
                label="Upload resume/profile (PDF/DOCX/TXT/MD)",
                file_types=[".pdf", ".docx", ".txt", ".md"],
            )
            download_html = gr.File(label="Download Website HTML")

    submit_btn.click(
        fn=handle_submit,
        inputs=[msg, chatbot, uploaded_file],
        outputs=[msg, chatbot],
    )

    gen_site_btn.click(
        fn=generate_website,
        inputs=[chatbot, uploaded_file],
        outputs=[download_html],
    )

if __name__ == "__main__":
    demo.launch()
