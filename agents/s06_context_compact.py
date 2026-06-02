#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
"""
s06_context_compact.py - Compact

三层上下文压缩管线，目的是让 agent 可以长时间运行而不耗尽上下文：

    每一轮交互：
    +------------------+
    | 工具调用的结果   |
    +------------------+
        |
        v
    [层1: micro_compact]        （每轮静默运行）
      将较旧的非 `read_file` 工具结果替换为占位符：
      "[Previous: used {tool_name}]"，只保留最近的若干条原始结果
        |
        v
    [检查：tokens > THRESHOLD?]
       |               |
       no              yes
       |               |
       v               v
    继续          [层2: auto_compact]
          将完整的对话写入 `.transcripts/`，请求 LLM 做总结，
          用摘要替代原始消息以释放上下文空间。
            |
            v
        [层3: compact 工具]
          模型触发 `compact` 工具进行手动压缩（立即运行 auto_compact）。

关键思想：agent 能够“有策略地忘记”，只保留对继续任务必要的信息，从而无限期运行。
本文件在关键步骤打印中文日志，便于理解上下文压缩的运行过程与原理。
"""

import json
import os
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 系统提示，用于 LLM 的 system 角色
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

# 触发自动压缩的 token 阈值、转录目录、保留最近条数等配置
THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3
PRESERVE_RESULT_TOOLS = {"read_file"}


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    # 层1：微压缩（micro_compact）
    # 说明：此函数遍历 messages，收集所有工具调用结果（type == "tool_result"），
    # 并将最早的那些非 read_file 的、大体积的结果替换为占位符。保留最近 KEEP_RECENT 条。
    print("[micro_compact] 开始检查工具结果，寻找可压缩项...")
    # Collect (msg_index, part_index, tool_result_dict) for all tool_result entries
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    print(f"[micro_compact] 找到 {len(tool_results)} 个工具结果；保留最近 {KEEP_RECENT} 条")
    if len(tool_results) <= KEEP_RECENT:
        print("[micro_compact] 无需压缩，条目数量在保留阈值内。")
        return messages
    # Find tool_name for each result by matching tool_use_id in prior assistant messages
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    # assistant 内容中可能包含工具调用描述（tool_use）
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
                        print(f"[micro_compact] 记录工具映射：id={block.id} name={block.name}")
    # Clear old results (keep last KEEP_RECENT). Preserve read_file outputs because
    # they are reference material; compacting them forces the agent to re-read files.
    to_clear = tool_results[:-KEEP_RECENT]
    cleared = 0
    for _, _, result in to_clear:
        # 只压缩较长的文本结果；短文本跳过
        if not isinstance(result.get("content"), str) or len(result["content"]) <= 100:
            continue
        tool_id = result.get("tool_use_id", "")
        tool_name = tool_name_map.get(tool_id, "unknown")
        if tool_name in PRESERVE_RESULT_TOOLS:
            print(f"[micro_compact] 保留来自工具 `{tool_name}` 的结果（未压缩）")
            continue
        result["content"] = f"[Previous: used {tool_name}]"
        cleared += 1
    print(f"[micro_compact] 完成：已将 {cleared} 条工具结果替换为占位符。")
    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    # 层2：自动压缩（auto_compact）
    # 说明：将完整对话写入磁盘，然后请求 LLM 生成摘要，最后用单条 user 消息替代原始对话
    print("[auto_compact] 开始自动压缩：保存转录并请求 LLM 生成摘要...")
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[auto_compact] 转录已保存: {transcript_path}")

    # 截取最近一段会话文本传给 LLM（避免过长）
    conversation_text = json.dumps(messages, default=str)[-80000:]
    print("[auto_compact] 向 LLM 发送摘要请求（max_tokens=2000）...")
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": (
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text)}],
        max_tokens=2000,
    )
    # next() 用于从 response.content 中提取第一个具有 text 属性的块作为摘要；如果没有则使用默认文本
    summary = next((block.text for block in response.content if hasattr(block, "text")), "")
    if not summary:
        summary = "（LLM 未生成摘要）"
    else:
        print("[auto_compact] 已收到摘要（片段）：", summary[:300].replace('\n', ' '))

    # 用单条压缩消息替代整个对话以释放上下文
    compressed = [{"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"}]
    print("[auto_compact] 自动压缩完成：对话已被摘要替代。")
    return compressed


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        print(f"[run_bash] 执行命令: {command}，返回长度={len(out)}")
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        content = "\n".join(lines)[:50000]
        print(f"[run_read] 读取文件: {path}，字节数={len(content)}")
        return content
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        print(f"[run_write] 写入文件: {path}，字节数={len(content)}")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        print(f"[run_edit] 在 {path} 中替换文本（首次出现）")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact":    lambda **kw: "Manual compression requested.",
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


def agent_loop(messages: list):
    # 主循环：每次与 LLM 交互前先做微压缩；如超出阈值则自动压缩；若 LLM 请求工具则处理工具调用并把结果追加回对话
    while True:
        print("\n[agent_loop] 新一轮 LLM 调用前，执行 micro_compact() 清理旧结果")
        micro_compact(messages)
        token_est = estimate_tokens(messages)
        print(f"[agent_loop] 估算 tokens={token_est}（阈值={THRESHOLD}）")
        # Layer 2: auto_compact if token estimate exceeds threshold
        if token_est > THRESHOLD:
            print("[agent_loop] 触发 auto_compact（上下文过大）")
            messages[:] = auto_compact(messages)

        print("[agent_loop] 向 LLM 发送请求，等待响应（含可能的工具调用）...")
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        print(f"[agent_loop] 收到 LLM 响应，stop_reason={getattr(response, 'stop_reason', None)}")

        # 将 LLM 的原始返回当作 assistant 内容追加到对话
        messages.append({"role": "assistant", "content": response.content})

        # 如果 LLM 没有请求工具使用，则这是最终回复，结束本次循环返回
        if response.stop_reason != "tool_use":
            print("[agent_loop] LLM 未请求工具，返回最终回复。")
            return

        # 处理工具调用结果并把 tool_result 加回对话
        results = []
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                print(f"[agent_loop] 处理工具调用：name={block.name} id={block.id}")
                if block.name == "compact":
                    manual_compact = True
                    output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    except Exception as e:
                        output = f"Error: {e}"
                # 打印工具输出的前200字符以便快速查看
                print(f"> {block.name} 输出预览:\n{str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        # 把工具结果作为下一条 user 消息追加，从而让 LLM 在后续轮次接着处理
        messages.append({"role": "user", "content": results})

        # 如果收到 compact 工具调用，则立即进行自动压缩并返回（模拟模型触发的手动压缩）
        if manual_compact:
            print("[agent_loop] 收到手动 compact 工具，执行自动压缩并结束本次循环")
            messages[:] = auto_compact(messages)
            return


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
