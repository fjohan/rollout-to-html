#!/usr/bin/env python3
"""
Convert Codex rollout/session files into a standalone transcript-style HTML page.

The renderer groups content into four visual classes:
1. user prompts
2. assistant replies
3. assistant thinking / progress output
4. code / tool / command output

Thinking and code sections are collapsed by default.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any


TEXT_KEYS = (
    "text",
    "content",
    "message",
    "output",
    "value",
    "body",
    "stdout",
    "stderr",
    "input",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Codex rollout/session file as a transcript-style HTML page."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Path to the rollout/session file. Omit to use the newest local Codex session for the current directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path (defaults to <input>.html)",
    )
    parser.add_argument(
        "--title",
        help="Optional page title override",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        help="Directory to match when auto-finding a rollout (defaults to the current working directory)",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path.home() / ".codex" / "sessions",
        help="Codex sessions directory to search when input is omitted",
    )
    parser.add_argument(
        "--list-matches",
        action="store_true",
        help="List matching local rollout files for the selected cwd and exit",
    )
    parser.add_argument(
        "--copy-rollout",
        nargs="?",
        const=Path("."),
        type=Path,
        help="Copy the matched rollout file into this directory (or the provided directory) before rendering",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def read_session_cwd(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as file:
            first_line = file.readline()
        record = json.loads(first_line)
    except (OSError, json.JSONDecodeError):
        return None

    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    cwd = payload.get("cwd")
    return cwd if isinstance(cwd, str) and cwd else None


def find_rollouts_for_cwd(target_cwd: Path, sessions_dir: Path) -> list[Path]:
    target = str(resolve_path(target_cwd))
    root = resolve_path(sessions_dir)
    if not root.exists():
        return []

    matches: list[Path] = []
    for path in root.rglob("rollout-*.jsonl"):
        session_cwd = read_session_cwd(path)
        if session_cwd and str(resolve_path(Path(session_cwd))) == target:
            matches.append(path)
    return sorted(matches, key=lambda item: item.name)


def default_output_path(source: Path, input_was_auto: bool) -> Path:
    if input_was_auto:
        return Path.cwd() / f"{source.stem}.html"
    return source.with_suffix(source.suffix + ".html")


def load_input(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        return []

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    records: list[Any] = []
    jsonl_ok = True
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            jsonl_ok = False
            break

    if jsonl_ok and records:
        return records

    return {"role": "assistant", "channel": "final", "content": raw}


def parse_json_maybe(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def iter_message_candidates(obj: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if any(key in node for key in ("role", "channel", "type", "content", "message", "output")):
                found.append(node)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(obj)
    return found


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [flatten_text(item) for item in value]
        return "\n".join(part for part in parts if part.strip())
    if isinstance(value, dict):
        if value.get("type") in {"input_text", "output_text", "text", "reasoning", "summary"}:
            return flatten_text(value.get("text") or value.get("content"))
        for key in TEXT_KEYS:
            if key in value:
                text = flatten_text(value[key])
                if text.strip():
                    return text
        if "args" in value:
            return flatten_text(value["args"])
        if "arguments" in value:
            return flatten_text(value["arguments"])
        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value)


def should_skip_instruction_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    prefixes = (
        "<permissions instructions>",
        "<collaboration_mode>",
        "<environment_context>",
        "<skills_instructions>",
        "<apps_instructions>",
        "<plugins_instructions>",
        "# AGENTS.md instructions",
    )
    return stripped.startswith(prefixes)


def format_tool_call(name: str, arguments: str) -> str:
    args_obj = parse_json_maybe(arguments)
    if isinstance(args_obj, dict):
        interesting = []
        for key in ("cmd", "workdir", "justification", "shell", "yield_time_ms"):
            if key in args_obj:
                interesting.append(f"{key}: {args_obj[key]}")
        if interesting:
            return f"{name}\n" + "\n".join(interesting)
    return f"{name}\n{arguments}".strip()


def looks_like_final_assistant_message(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    starters = (
        "implemented",
        "added",
        "updated",
        "fixed",
        "created",
        "completed",
        "changed",
    )
    lowered = stripped.lower()
    return lowered.startswith(starters) or "validation:" in lowered or "what changed" in lowered


def extract_from_jsonl_records(records: list[Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    for record in records:
        if not isinstance(record, dict):
            continue

        top_type = record.get("type")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue

        payload_type = payload.get("type")

        if top_type == "event_msg" and payload_type == "user_message":
            text = flatten_text(payload.get("message")).strip()
            if text and not should_skip_instruction_text(text):
                messages.append({"kind": "user", "label": "User", "text": text})
            continue

        if top_type == "event_msg" and payload_type == "agent_message":
            text = flatten_text(payload.get("message")).strip()
            if not text:
                continue
            phase = str(payload.get("phase", "")).lower()
            kind = "assistant" if phase == "final_answer" or (not phase and looks_like_final_assistant_message(text)) else "thinking"
            label = "Codex" if kind == "assistant" else "Thinking"
            messages.append({"kind": kind, "label": label, "text": text})
            continue

        if top_type == "event_msg" and payload_type == "agent_reasoning":
            text = flatten_text(payload.get("text")).strip()
            if text:
                messages.append({"kind": "thinking", "label": "Thinking", "text": text})
            continue

        if top_type == "response_item" and payload_type == "reasoning":
            text = flatten_text(payload.get("summary")).strip()
            if text:
                messages.append({"kind": "thinking", "label": "Thinking", "text": text})
            continue

        if top_type == "response_item" and payload_type == "message":
            role = str(payload.get("role", "")).lower()
            phase = str(payload.get("phase", "")).lower()
            text = flatten_text(payload.get("content")).strip()
            if not text or should_skip_instruction_text(text):
                continue
            if role == "user":
                messages.append({"kind": "user", "label": "User", "text": text})
                continue
            if role == "assistant" and phase == "final_answer":
                messages.append({"kind": "assistant", "label": "Codex", "text": text})
            elif role == "assistant" and phase == "commentary":
                messages.append({"kind": "thinking", "label": "Thinking", "text": text})
            continue

        if top_type == "response_item" and payload_type == "function_call":
            name = str(payload.get("name", "tool"))
            arguments = flatten_text(payload.get("arguments")).strip()
            text = format_tool_call(name, arguments)
            messages.append({"kind": "code", "label": f"Code Output ({name})", "text": text})
            continue

        if top_type == "response_item" and payload_type == "function_call_output":
            call_id = str(payload.get("call_id", "")).strip()
            output = flatten_text(payload.get("output")).strip()
            if output:
                label = f"Code Output ({call_id})" if call_id else "Code Output"
                messages.append({"kind": "code", "label": label, "text": output})
            continue

        if top_type == "response_item" and payload_type == "custom_tool_call":
            name = str(payload.get("name", "tool"))
            tool_input = flatten_text(payload.get("input")).strip()
            if tool_input:
                messages.append({"kind": "code", "label": f"Code Output ({name})", "text": tool_input})
            continue

        if top_type == "response_item" and payload_type == "custom_tool_call_output":
            output = flatten_text(payload.get("output")).strip()
            parsed = parse_json_maybe(output)
            if isinstance(parsed, dict) and "output" in parsed:
                output = flatten_text(parsed.get("output")).strip()
            if output:
                messages.append({"kind": "code", "label": "Code Output", "text": output})
            continue

    compacted: list[dict[str, str]] = []
    for entry in messages:
        if compacted and compacted[-1]["kind"] == entry["kind"] and compacted[-1]["text"] == entry["text"]:
            continue
        if compacted and compacted[-1]["kind"] == entry["kind"] and compacted[-1]["label"] == entry["label"]:
            compacted[-1]["text"] += "\n\n" + entry["text"]
        else:
            compacted.append(entry)
    return compacted


def classify_message(item: dict[str, Any]) -> str | None:
    role = str(item.get("role", "")).lower()
    channel = str(item.get("channel", "")).lower()
    msg_type = str(item.get("type", "")).lower()
    name = str(item.get("name", "")).lower()

    if role == "user":
        return "user"

    if role in {"tool", "function"}:
        return "code"

    if role == "assistant":
        if channel == "final":
            return "assistant"
        if channel in {"analysis", "commentary", "summary"}:
            return "thinking"

    if msg_type in {"tool_result", "tool_call", "function_call", "function_result", "command"}:
        return "code"
    if msg_type in {"reasoning", "analysis", "thought", "commentary"}:
        return "thinking"
    if msg_type in {"message", "assistant_message"} and role == "assistant":
        return "assistant"

    if channel in {"analysis", "commentary"}:
        return "thinking"
    if channel == "final":
        return "assistant"

    if name in {"exec_command", "write_stdin", "apply_patch"}:
        return "code"

    text = flatten_text(item).strip()
    if not text:
        return None

    if role == "assistant":
        return "assistant"

    return None


def extract_label(item: dict[str, Any], kind: str) -> str:
    if kind == "user":
        return "User"
    if kind == "assistant":
        return "Codex"
    if kind == "thinking":
        channel = str(item.get("channel", "")).lower()
        if channel:
            return f"Thinking ({channel})"
        return "Thinking"
    if kind == "code":
        msg_type = item.get("type") or item.get("name") or item.get("channel")
        if msg_type:
            return f"Code Output ({msg_type})"
        return "Code Output"
    return "Message"


def normalize_messages(data: Any) -> list[dict[str, str]]:
    if isinstance(data, list) and data and all(isinstance(item, dict) and "timestamp" in item and "type" in item for item in data):
        extracted = extract_from_jsonl_records(data)
        if extracted:
            return extracted

    raw_items: list[dict[str, Any]]
    if isinstance(data, list):
        raw_items = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict) and isinstance(data.get("messages"), list):
        raw_items = [item for item in data["messages"] if isinstance(item, dict)]
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        raw_items = [item for item in data["items"] if isinstance(item, dict)]
    else:
        raw_items = iter_message_candidates(data)

    normalized: list[dict[str, str]] = []
    for item in raw_items:
        kind = classify_message(item)
        if not kind:
            continue

        text = (
            flatten_text(item.get("content"))
            or flatten_text(item.get("message"))
            or flatten_text(item.get("output"))
            or flatten_text(item)
        ).strip()
        if not text:
            continue

        normalized.append(
            {
                "kind": kind,
                "label": extract_label(item, kind),
                "text": text,
            }
        )

    compacted: list[dict[str, str]] = []
    for entry in normalized:
        if compacted and compacted[-1]["kind"] == entry["kind"] and compacted[-1]["label"] == entry["label"]:
            compacted[-1]["text"] += "\n\n" + entry["text"]
        else:
            compacted.append(entry)
    return compacted


def render_message(entry: dict[str, str]) -> str:
    kind = entry["kind"]
    label = html.escape(entry["label"])
    body = html.escape(entry["text"])
    collapsible = kind in {"thinking", "code"}

    if collapsible:
        return f"""
        <details class="message {kind}">
          <summary><span class="label">{label}</span></summary>
          <pre>{body}</pre>
        </details>
        """.strip()

    return f"""
    <section class="message {kind}">
      <div class="label">{label}</div>
      <pre>{body}</pre>
    </section>
    """.strip()


def build_html(title: str, messages: list[dict[str, str]]) -> str:
    body = "\n".join(render_message(message) for message in messages)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --border: #d8cdbd;
      --text: #1e1a16;
      --muted: #6b6259;
      --user: #dbeafe;
      --assistant: #e7f7df;
      --thinking: #fef3c7;
      --code: #ece7ff;
      --shadow: 0 12px 30px rgba(56, 43, 27, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.75), transparent 30%),
        linear-gradient(180deg, #f3efe5 0%, #ede7db 100%);
    }}
    .page {{
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 16px 48px;
    }}
    .header {{
      margin-bottom: 20px;
      padding: 20px 22px;
      background: rgba(255, 253, 248, 0.8);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0 0 8px;
      font-family: "IBM Plex Serif", Georgia, serif;
      font-size: clamp(1.6rem, 3vw, 2.3rem);
    }}
    .subtle {{
      margin: 0;
      color: var(--muted);
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 18px 0 24px;
    }}
    button {{
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
      box-shadow: var(--shadow);
    }}
    .transcript {{
      display: grid;
      gap: 14px;
    }}
    .is-hidden {{
      display: none !important;
    }}
    .message {{
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--panel);
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    .message.user {{ border-left: 8px solid #60a5fa; background: linear-gradient(180deg, var(--user), #f8fbff); }}
    .message.assistant {{ border-left: 8px solid #65a30d; background: linear-gradient(180deg, var(--assistant), #fbfff8); }}
    .message.thinking {{ border-left: 8px solid #d97706; background: linear-gradient(180deg, var(--thinking), #fffdf4); }}
    .message.code {{ border-left: 8px solid #7c3aed; background: linear-gradient(180deg, var(--code), #faf8ff); }}
    .label {{
      display: block;
      padding: 14px 18px 0;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .message pre {{
      margin: 0;
      padding: 12px 18px 18px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.95rem;
      line-height: 1.5;
    }}
    details > summary {{
      list-style: none;
      cursor: pointer;
      padding: 0 0 14px;
    }}
    details > summary::-webkit-details-marker {{ display: none; }}
    details > summary .label::after {{
      content: " (collapsed)";
      color: var(--muted);
      font-weight: 500;
    }}
    details[open] > summary .label::after {{
      content: " (expanded)";
    }}
    @media (max-width: 640px) {{
      .page {{ padding: 20px 10px 36px; }}
      .header {{ padding: 16px; }}
      .message pre {{ font-size: 0.9rem; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="header">
      <h1>{html.escape(title)}</h1>
      <p class="subtle">Transcript-style rendering of a Codex rollout file.</p>
    </header>
    <div class="toolbar">
      <button type="button" data-view="conversation">Conversation Only</button>
      <button type="button" data-view="user">User Only</button>
      <button type="button" data-target="thinking">Toggle Thinking</button>
      <button type="button" data-target="code">Toggle Code Output</button>
    </div>
    <section class="transcript">
      {body or '<section class="message assistant"><div class="label">Codex</div><pre>No transcript content was detected.</pre></section>'}
    </section>
  </main>
  <script>
    const viewButtons = document.querySelectorAll('[data-view]');
    const allMessages = document.querySelectorAll('.message');
    let activeView = 'all';

    function applyView(nextView) {{
      activeView = activeView === nextView ? 'all' : nextView;
      allMessages.forEach((node) => {{
        const hideForConversation = activeView === 'conversation' &&
          (node.classList.contains('thinking') || node.classList.contains('code'));
        const hideForUser = activeView === 'user' && !node.classList.contains('user');
        node.classList.toggle('is-hidden', hideForConversation || hideForUser);
      }});
      viewButtons.forEach((viewButton) => {{
        viewButton.classList.toggle('is-active', viewButton.dataset.view === activeView);
      }});
    }}

    document.querySelectorAll('.toolbar button').forEach((button) => {{
      button.addEventListener('click', () => {{
        if (button.dataset.target) {{
          const kind = button.dataset.target;
          document.querySelectorAll(`details.${{kind}}`).forEach((node) => {{
            node.open = !node.open;
          }});
          return;
        }}

        if (button.dataset.view) {{
          applyView(button.dataset.view);
        }}
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    input_was_auto = args.input is None

    if input_was_auto:
        target_cwd = args.cwd or Path.cwd()
        matches = find_rollouts_for_cwd(target_cwd, args.sessions_dir)
        if args.list_matches:
            if not matches:
                print(f"No rollout files found for cwd: {resolve_path(target_cwd)}")
                return 1
            for match in matches:
                print(match)
            return 0
        if not matches:
            print(f"No rollout files found for cwd: {resolve_path(target_cwd)}")
            print(f"Searched: {resolve_path(args.sessions_dir)}")
            return 1
        source = matches[-1]
        print(f"Using rollout: {source}")
    else:
        source = args.input

    if args.copy_rollout is not None:
        copy_target = args.copy_rollout
        if copy_target.suffix != ".jsonl":
            copy_target = copy_target / source.name
        copy_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copy_target)
        print(f"Copied rollout to {copy_target}")

    output = args.output or default_output_path(source, input_was_auto)
    title = args.title or f"Codex Transcript: {source.name}"

    data = load_input(source)
    messages = normalize_messages(data)
    rendered = build_html(title, messages)

    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
