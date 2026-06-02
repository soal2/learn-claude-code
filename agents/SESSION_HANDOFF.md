# Session Handoff Document
## Agent Development Project — /Users/eversse/Documents/codes/self-learning/learn-claude-code/agents/

### Project Overview
This is a progressive, 12-session tutorial building a self-evolving AI agent framework in Python. Each session file (`s01` through `s12`) builds on the previous, culminating in `s_full.py` — a complete, runnable multi-agent system with autonomous capabilities.

---

### Architecture Summary

#### Core Loop (s01, s_full)
- `while True:` loop with `input()` for user messages
- LLM called via `litellm.completion()` (supports OpenAI, Claude, Gemini via API keys)
- `tool_choice: "auto"` — the LLM decides when to use tools
- Conversation history passed each turn as `messages`
- Exit when LLM responds with no `tool_calls`

#### Tool System (s02)
- Tools defined as Python dicts with `name`, `description`, `parameters` (JSON Schema)
- `handle_tool_call()` dispatches via `if/elif` to actual Python functions
- Tools: `read_file`, `write_file`, `edit_file`, `bash`, `compact`, `web_search`, `add_subtask`, `delegate_to_agent`, `spawn_agent`, `list_agents`, `send_message`, `web_search`, `list_web_pages`

#### Context Compaction (s03)
- User types "compact" → sends summarization prompt to LLM
- Replaces full history with `{"role":"user","content":"[CONTEXT COMPACTED]\n" + summary}`

#### Sub-Agent / Thread Model (s04, s11)
- Agents have: `id`, `name`, `system_prompt`, `conversation` (message list), `tools`, `status`, `handoff`
- Sub-agents run in background threads via `threading.Thread`
- `run_agent_turn()` is the inner loop — runs until no tool_calls, max 10 iterations
- Results returned via `agent.handoff["result"]`

#### Team Protocol (s05, s10)
- `spawn_agent()` — creates an agent and optionally starts it
- `delegate_to_agent()` — sends a task to an existing agent, waits for result
- `add_subtask()` — for synchronous nested delegation (blocks until complete)
- `send_message()` — async message between agents (checked each turn)
- `list_agents()` — shows all agents and their status
- JSON-based protocol files in `agents/.agents/` directory
- `load_protocol()` / `save_protocol()` for persistence

#### Auto-Evolution (s08, s_full)
- `check_evolution()` reads from `agents/.auto_evolution/prompts/`
- On "NEW" marker → generates new tool, writes to `.tools/`, marks "DONE"
- On "ADD" marker → adds new tool name to `active_tools`, writes to `.tool_manifest`
- Evolved tools auto-imported at startup via `importlib`

#### Human-in-the-Loop (s09, s10)
- `ask_human` tool — pauses agent, writes to `.pending/`, waits for `approval.txt`
- `ask_user()` function — direct user input for tool confirmation
- `request_approval()` — writes decision request to `.pending/`
- `check_pending_decisions()` — checks if decisions were approved

#### Directory Isolation (s12)
- Each agent gets its own git worktree: `agents/.worktrees/{agent_id}`
- Created via `git worktree add` from main repo
- Agent operates only in its isolated directory
- `cleanup_worktree()` removes worktree on completion
- `is_path_safe()` — validates paths before write_file/bash operations

#### Task Isolation (s12)
- `TaskManager` class with dependency graph (`Task` dataclass)
- Tasks have: id, description, status (pending→in_progress→completed/failed), dependencies, result
- Topological sort for execution order
- Parallel execution via `ThreadPoolExecutor` with `max_workers=3`
- `create_epic()` — auto-creates task graphs with dependencies

---

### Complete Tool Inventory (s_full.py)

| Tool | Category | Description |
|------|----------|-------------|
| `read_file` | File Ops | Read file contents |
| `write_file` | File Ops | Write content to file |
| `edit_file` | File Ops | Replace exact text |
| `bash` | System | Run shell commands |
| `compact` | Context | Summarize conversation |
| `web_search` | Research | Search web (DuckDuckGo) |
| `list_web_pages` | Research | List discovered pages |
| `add_subtask` | Task Mgmt | Create synchronous sub-task |
| `delegate_to_agent` | Team | Send task to existing agent |
| `spawn_agent` | Team | Create new agent |
| `list_agents` | Team | List all agents |
| `send_message` | Team | Async inter-agent messaging |
| `ask_human` | Human-ITL | Request human input/approval |
| `create_epic` | Task Mgmt | Create epic with auto-dependencies |
| `add_dependency` | Task Mgmt | Add task dependency |
| `execute_epic` | Task Mgmt | Execute task graph |
| `list_tasks` | Task Mgmt | List all tasks |

---

### Auto-Evolution Protocol

**Directory:** `agents/.auto_evolution/`
```
prompts/
  new_tools.txt     — Write "NEW: tool_name" + instructions, agent generates code
  enhance_tools.txt — Write "ADD: tool_name" to activate a generated tool
  team_growth.txt   — Future: auto-spawn specialized agents
```

**Tool Storage:** `agents/.tools/`
```
tool_manifest.json  — {"active_tools": ["read_file", "write_file", ...]}
{tool_name}.py      — Generated tool files
```

**File Protocol:** Each agent in `.agents/{agent_id}/`
```
{agent_id}.json     — Agent state and conversation
requests.txt        — Incoming task requests
results.txt         — Task results
messages.txt        — Inter-agent messages
```

---

### Key Design Patterns

1. **Progressive Complexity**: s01→s12 each add one concept
2. **Single-Threaded Core**: Main loop is `while True`, agents are threads
3. **File-Based Communication**: Agents communicate via `.agents/` directory
4. **Self-Referential**: The agent can read/modify its own source code
5. **Safety by Convention**: `is_path_safe()` + `.block` markers, not enforced OS-level

---

### Running the System
```bash
cd /Users/eversse/Documents/codes/self-learning/learn-claude-code/agents/
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."  # Optional, for Claude
export GEMINI_API_KEY="..."     # Optional, for Gemini
python3 s_full.py
```

### LLM Model Configuration
- Default: `openai/gpt-4o-mini` (cheapest)
- Can change in `call_llm()`: `"anthropic/claude-sonnet-4-20250514"`, `"gemini/gemini-2.0-flash"`
- Requires corresponding API key set as env var

---

### Current File Sizes
- `s_full.py` — 437 lines (complete system)
- `s12_*.py` — ~150-200 lines each (focused tutorials)
- `utils.py` — 113 lines (shared: printl, save, check_api_keys)
- `.tool_manifest` — 55 lines (tool descriptions)
- `hello.py` — 12 lines (basic demo)

### Important Notes for Next Session
- The codebase is a **learning tutorial**, not production software
- `s_full.py` is the **canonical reference** implementation
- `s04` has a known bug: `check_pending_decisions()` referenced but not defined
- File-based inter-agent communication can be racy (no file locking)
- Git worktrees require the parent repo to be a git repository
- Some tools in `s_full.py` are stubs (web_search depends on duckduckgo-search)
- The auto-evolution system writes files to `.tools/` but requires manual activation
