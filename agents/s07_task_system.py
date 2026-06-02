#!/usr/bin/env python3
# Harness: persistent tasks -- goals that outlive any single conversation.
"""
s07_task_system.py - Tasks

Tasks persist as JSON files in .tasks/ so they survive context compression.
Each task has a dependency graph (blockedBy).

    .tasks/
      task_1.json  {"id":1, "subject":"...", "status":"completed", ...}
      task_2.json  {"id":2, "blockedBy":[1], "status":"pending", ...}
      task_3.json  {"id":3, "blockedBy":[2], ...}

    Dependency resolution:
    +----------+     +----------+     +----------+
    | task 1   | --> | task 2   | --> | task 3   |
    | complete |     | blocked  |     | blocked  |
    +----------+     +----------+     +----------+
         |                ^
         +--- completing task 1 removes it from task 2's blockedBy

Key insight: "State that survives compression -- because it's outside the conversation."
"""

import json
import os
import subprocess
from pathlib import Path

import httpx
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
MODEL = os.environ["MODEL_ID"]
TASKS_DIR = WORKDIR / ".tasks"

SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."


def log_info(message: str):
    """统一输出中文运行日志，方便观察任务系统的完整执行过程。"""
    print(f"[任务系统] {message}")


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


# -- TaskManager: CRUD with dependency graph, persisted as JSON files --
# 任务管理器：CRUD（创建、读取、更新、删除）与依赖关系图，持久化为JSON文件
class TaskManager:
    # 初始化任务管理器，确保任务目录存在，并计算下一个可用的任务ID
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        log_info(f"任务目录已准备就绪：{self.dir}")
        self._next_id = self._max_id() + 1
        log_info(f"下一个可创建的任务ID：{self._next_id}")

    # 获取当前最大的任务ID，以便为新任务分配一个新的唯一ID
    def _max_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        max_id = max(ids) if ids else 0
        log_info(f"扫描现有任务文件，当前最大任务ID：{max_id}")
        return max_id

    # 从文件系统加载任务数据，返回一个字典表示任务的属性
    def _load(self, task_id: int) -> dict:
        path = self.dir / f"task_{task_id}.json"
        log_info(f"准备加载任务文件：{path}")
        if not path.exists():
            log_info(f"任务文件不存在：{path}")
            raise ValueError(f"Task {task_id} not found")
        task = json.loads(path.read_text())
        log_info(f"任务加载完成：#{task_id} {task.get('subject', '')}")
        return task

    # 将任务数据保存到文件系统中，以JSON格式存储，文件名包含任务ID
    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        log_info(f"写入任务文件：{path}")
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))
        log_info(f"任务已保存：#{task['id']}，状态：{task['status']}，依赖：{task.get('blockedBy', [])}")

    # 创建一个新任务，分配一个唯一ID，并保存到文件系统中，返回任务的JSON表示
    def create(self, subject: str, description: str = "") -> str:
        log_info(f"开始创建任务：{subject}")
        task = {
            "id": self._next_id, "subject": subject, "description": description,
            "status": "pending", "blockedBy": [], "owner": "",
        }
        self._save(task)
        self._next_id += 1
        log_info(f"任务创建完成：#{task['id']}，下一次创建将使用ID：{self._next_id}")
        return json.dumps(task, indent=2, ensure_ascii=False)

    # 获取一个任务的详细信息，返回任务的JSON表示
    def get(self, task_id: int) -> str:
        log_info(f"读取任务详情：#{task_id}")
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    # 更新一个任务的状态或依赖关系，支持修改状态、添加或移除依赖，保存更改并返回更新后的任务JSON表示
    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, remove_blocked_by: list = None) -> str:
        log_info(f"开始更新任务：#{task_id}")
        task = self._load(task_id)
        # 如果提供了新的状态，验证状态是否合法并更新任务状态；如果状态变为完成，调用清理依赖的方法；如果提供了要添加的依赖，更新blockedBy列表；如果提供了要移除的依赖，从blockedBy列表中删除对应项；最后保存任务并返回更新后的JSON表示
        if status:
            log_info(f"准备修改任务状态：#{task_id} -> {status}")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                log_info(f"任务已完成，开始清理被该任务阻塞的其他任务：#{task_id}")
                self._clear_dependency(task_id)
        if add_blocked_by:
            log_info(f"为任务 #{task_id} 新增依赖：{add_blocked_by}")
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if remove_blocked_by:
            log_info(f"为任务 #{task_id} 移除依赖：{remove_blocked_by}")
            task["blockedBy"] = [x for x in task["blockedBy"] if x not in remove_blocked_by]
        self._save(task)
        log_info(f"任务更新完成：#{task_id}")
        return json.dumps(task, indent=2, ensure_ascii=False)

    # 当一个任务被标记为完成时，调用此方法从所有其他任务的blockedBy列表中移除该任务ID，以解除依赖关系
    def _clear_dependency(self, completed_id: int):
        """Remove completed_id from all other tasks' blockedBy lists."""
        log_info(f"清理依赖关系，移除所有任务中的完成任务ID：{completed_id}")
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                log_info(f"任务 #{task['id']} 解除依赖：移除 {completed_id}")
                task["blockedBy"].remove(completed_id)
                self._save(task)

    # 列出所有任务，按照ID排序，并以文本形式返回每个任务的状态、主题和依赖关系摘要
    def list_all(self) -> str:
        log_info("开始列出所有任务")
        tasks = []
        files = sorted(
            self.dir.glob("task_*.json"),
            key=lambda f: int(f.stem.split("_")[1])
        )
        for f in files:
            log_info(f"读取任务列表文件：{f}")
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            log_info("当前没有任何任务")
            return "No tasks."
        lines = []
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        log_info(f"任务列表读取完成，共 {len(tasks)} 个任务")
        return "\n".join(lines)


TASKS = TaskManager(TASKS_DIR)


# -- Base tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    log_info(f"解析路径：{p} -> {path}")
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    log_info(f"准备执行 bash 命令：{command}")
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        log_info("检测到危险命令，已阻止执行")
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        log_info(f"bash 命令执行完成，返回码：{r.returncode}")
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        log_info("bash 命令执行超时")
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        log_info(f"读取文件请求：path={path}, limit={limit}")
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        log_info(f"文件读取完成：{path}，返回行数：{len(lines)}")
        return "\n".join(lines)[:50000]
    except Exception as e:
        log_info(f"文件读取失败：{path}，原因：{e}")
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        log_info(f"写入文件请求：path={path}，内容长度：{len(content)}")
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        log_info(f"文件写入完成：{path}")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        log_info(f"文件写入失败：{path}，原因：{e}")
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        log_info(f"编辑文件请求：path={path}")
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            log_info(f"编辑失败，未找到要替换的文本：{path}")
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        log_info(f"文件编辑完成：{path}")
        return f"Edited {path}"
    except Exception as e:
        log_info(f"文件编辑失败：{path}，原因：{e}")
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("removeBlockedBy")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
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
    {"name": "task_create", "description": "Create a new task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update a task's status or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "removeBlockedBy": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status summary.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full details of a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


def agent_loop(messages: list):
    log_info("进入对话循环，准备向模型请求下一步动作")
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        log_info(f"模型返回，stop_reason={response.stop_reason}")
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            log_info("模型未请求工具，当前轮结束")
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                log_info(f"开始执行工具：{block.name}，输入：{block.input}")
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                    log_info(f"工具执行异常：{block.name}，原因：{e}")
                print(f"> {block.name}:")
                print(str(output)[:200])
                log_info(f"工具执行完成：{block.name}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    log_info("任务系统脚本已启动")
    history = []
    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            log_info("收到退出信号，准备结束程序")
            break
        if query.strip().lower() in ("q", "exit", ""):
            log_info("用户输入退出指令，准备结束程序")
            break
        log_info(f"收到用户输入：{query}")
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        log_info("本轮输出完成")
        print()
