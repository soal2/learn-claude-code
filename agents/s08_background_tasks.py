#!/usr/bin/env python3
# Harness: background execution -- the model thinks while the harness waits.
"""
s08_background_tasks.py - Background Tasks

Run commands in background threads. A notification queue is drained
before each LLM call to deliver results.

    Main thread                Background thread
    +-----------------+        +-----------------+
    | agent loop      |        | task executes   |
    | ...             |        | ...             |
    | [LLM call] <---+------- | enqueue(result) |
    |  ^drain queue   |        +-----------------+
    +-----------------+

    Timeline:
    Agent ----[spawn A]----[spawn B]----[other work]----
                 |              |
                 v              v
              [A runs]      [B runs]        (parallel)
                 |              |
                 +-- notification queue --> [results injected]

Key insight: "Fire and forget -- the agent doesn't block while the command runs."
"""

import os
import subprocess
import threading
import uuid
from pathlib import Path

import httpx
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
# client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use background_run for long-running commands."


def log_info(message: str):
    """统一输出中文运行日志，方便观察后台任务的完整生命周期。"""
    print(f"[后台任务] {message}")

def build_anthropic_client() -> Anthropic:
    proxy_keys = [
        key for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
        if os.getenv(key)
    ]
    if proxy_keys:
        log_info(f"检测到代理环境变量：{proxy_keys}，Anthropic 客户端将忽略系统代理")
    return Anthropic(
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        http_client=httpx.Client(trust_env=False),
    )


client = build_anthropic_client()
log_info("Anthropic 客户端已初始化完成")

# -- BackgroundManager: threaded execution + notification queue --
class BackgroundManager:
    # 管理后台任务的生命周期：启动线程、捕获输出、维护状态、提供查询接口、排队通知结果。
    def __init__(self):
        log_info("初始化后台任务管理器")
        self.tasks = {}  # task_id -> {status, result, command}
        self._notification_queue = []  # completed task results
        self._lock = threading.Lock()

    # 启动后台线程，立即返回任务ID。
    # 线程执行入口：运行子进程、捕获输出、写入通知队列。
    def run(self, command: str) -> str:
        """启动后台线程，立即返回任务ID。"""
        task_id = str(uuid.uuid4())[:8]
        log_info(f"收到后台任务创建请求，准备启动任务 {task_id}，命令：{command}")
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}
        # 启动后台线程执行命令，线程入口为 self._execute，传入 task_id 和 command 参数，设置为守护线程。
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        thread.start()
        log_info(f"后台线程已启动，任务 {task_id} 正在独立执行")
        return f"Background task {task_id} started: {command[:80]}"

    # 线程执行入口：运行子进程、捕获输出、写入通知队列。
    def _execute(self, task_id: str, command: str):
        """线程执行入口：运行子进程、捕获输出、写入通知队列。"""
        log_info(f"任务 {task_id} 开始执行，后台线程进入子进程阶段")
        try:
            r = subprocess.run(
                command, shell=True, cwd=WORKDIR,
                capture_output=True, text=True, timeout=300
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
            log_info(f"任务 {task_id} 执行完成，退出码：{r.returncode}")
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
            log_info(f"任务 {task_id} 执行超时")
        except Exception as e:
            output = f"Error: {e}"
            status = "error"
            log_info(f"任务 {task_id} 执行异常：{e}")
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "result": (output or "(no output)")[:500],
            })
        log_info(f"任务 {task_id} 的结果已放入通知队列，等待下一轮主循环读取")

    def check(self, task_id: str = None) -> str:
        """查询单个任务状态，或列出全部任务。"""
        log_info("收到后台任务查询请求")
        if task_id:
            log_info(f"查询指定任务：{task_id}")
            t = self.tasks.get(task_id)
            if not t:
                log_info(f"未找到任务：{task_id}")
                return f"Error: Unknown task {task_id}"
            return f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}"
        lines = []
        for tid, t in self.tasks.items():
            lines.append(f"{tid}: [{t['status']}] {t['command'][:60]}")
        log_info(f"返回全部任务列表，共 {len(lines)} 个任务")
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """取出并清空所有已完成任务的通知。"""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        if notifs:
            log_info(f"主循环准备读取 {len(notifs)} 条后台任务通知")
        return notifs


BG = BackgroundManager()


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    log_info(f"准备执行阻塞式 bash 命令：{command}")
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        log_info("检测到危险命令，已阻止执行")
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        log_info(f"阻塞式 bash 命令执行完成，返回码：{r.returncode}")
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        log_info("阻塞式 bash 命令执行超时")
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    log_info(f"读取文件请求：path={path}, limit={limit}")
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        log_info(f"文件读取完成：{path}，返回 {len(lines)} 行")
        return "\n".join(lines)[:50000]
    except Exception as e:
        log_info(f"文件读取失败：{path}，原因：{e}")
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    log_info(f"写入文件请求：path={path}，内容长度={len(content)}")
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        log_info(f"文件写入完成：{path}")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        log_info(f"文件写入失败：{path}，原因：{e}")
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    log_info(f"编辑文件请求：path={path}")
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            log_info(f"编辑失败：未找到目标文本，path={path}")
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        log_info(f"文件编辑完成：{path}")
        return f"Edited {path}"
    except Exception as e:
        log_info(f"文件编辑失败：{path}，原因：{e}")
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":             lambda **kw: run_bash(kw["command"]),
    "read_file":        lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":       lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":        lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "background_run":   lambda **kw: BG.run(kw["command"]),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command (blocking).",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "background_run", "description": "Run command in background thread. Returns task_id immediately.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "check_background", "description": "Check background task status. Omit task_id to list all.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
]


def agent_loop(messages: list):
    log_info(f"进入一轮 agent_loop，当前消息数：{len(messages)}")
    while True:
        # 在每次 LLM 调用前，先把后台任务完成通知取出来注入上下文。
        notifs = BG.drain_notifications()
        if notifs and messages:
            log_info("将后台任务完成结果注入给模型，供下一轮推理使用")
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs
            )
            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})
        log_info(f"准备发起 LLM 请求，累计消息数：{len(messages)}")
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        log_info(f"LLM 响应返回，stop_reason={response.stop_reason}")
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            log_info("本轮未继续调用工具，agent_loop 结束")
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                log_info(f"模型请求调用工具：{block.name}")
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"[工具输出] {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        log_info(f"工具执行完成，准备把 {len(results)} 条 tool_result 回传给模型")
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    log_info("后台任务示例程序已启动")
    log_info("输入 q、exit 或直接回车可以退出")
    history = []
    while True:
        try:
            query = input("\033[36ms08 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            log_info("收到中断信号，准备退出程序")
            break
        if query.strip().lower() in ("q", "exit", ""):
            log_info(f"收到退出指令：{query!r}")
            break
        log_info(f"用户输入：{query}")
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        log_info("本轮交互结束，等待下一次输入")
        print()
