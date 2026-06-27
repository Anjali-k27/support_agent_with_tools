# Enterprise AI Support Platform

A 12-session, ground-up build of a production-grade AI support agent using **LangGraph**, **Gemini 2.5 Flash**, and **FastAPI**. Each session adds a real capability on top of the previous one — no code is ever removed, only extended.

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd phase3-session2

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Google API key
echo "GOOGLE_API_KEY=your-key-here" > .env

# 5. Verify the agent works from CLI
python support_agent.py

# 6. Start the API server
python api.py

# 7. Open the UI
open http://localhost:8000
```

---

## Project Structure

```
phase3-session2/
├── support_agent.py   # LangGraph backend — all agent logic lives here
├── api.py             # FastAPI server — exposes /api/run, /api/stream, /api/verify
├── index.html         # Dark glassmorphism UI — served statically by FastAPI
├── requirements.txt   # All dependencies pinned
├── .env               # GOOGLE_API_KEY (git-ignored)
└── .venv/             # Virtual environment (git-ignored)
```

---

## Architecture

```
User Ticket
     │
     ▼
classify_node  ──── Gemini classifies into 4 categories
     │
     ▼
route_by_category
     ├── billing   ──► agent_node ◄──── (Session 2+)
     ├── technical ──► agent_node
     ├── fraud     ──► fraud_handler   (stub → replaced Session 9)
     └── general   ──► general_handler (permanent)

agent_node  ──── Gemini + bound tools
     │
     ▼
route_after_agent
     ├── tool_calls? ──► tool_node ──► agent_node  (loop)
     └── no tools    ──► respond_node ──► END
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/run` | Run a ticket synchronously. Returns full result + tool call log. |
| `POST` | `/api/stream` | Run a ticket with SSE streaming. Each graph node fires an event. |
| `POST` | `/api/verify` | Run the session verification suite. Returns structured pass/fail report. |
| `GET`  | `/health` | Server health check. Returns session number and tool count. |

### POST /api/run — Request
```json
{ "ticket": "My account C-1002 shows past due. Check it." }
```

### POST /api/run — Response
```json
{
  "category": "billing",
  "final_response": "Your account C-1002 is Past Due...",
  "is_safe": true,
  "pii_detected": false,
  "iteration_count": 0,
  "raw_input": "My account C-1002 shows past due. Check it.",
  "tool_calls_log": [
    {
      "tool_name": "get_customer_details",
      "args": { "customer_id": "C-1002" },
      "call_id": "...",
      "result": {
        "name": "Arjun Mehta",
        "billing_status": "Past Due",
        "outstanding_balance": 998.0,
        ...
      }
    }
  ]
}
```

---

## Tools (Session 2)

### `get_customer_details(customer_id: str)`
Looks up a customer in the mock CRM. Returns billing status, subscription tier, last payment, outstanding balance, and recent transactions. Strips internal/noisy fields before returning.

**Call when:** Any billing, payment, or account query.  
**Format:** `customer_id` must be `C-XXXX` (e.g. `C-1001`). Ask the user if not provided — never guess.

### `search_knowledge_base(query: str)`
Keyword-searches the internal KB. Returns matching troubleshooting articles.

**Call when:** Any technical issue, before responding.  
**Format:** Natural language query with specific keywords/error codes.

---

## Mock Data

### CRM Records
| ID | Name | Tier | Status | Balance |
|----|------|------|--------|---------|
| C-1001 | Priya Sharma | Enterprise | Active | $0 |
| C-1002 | Arjun Mehta | Pro | Past Due | $998 |
| C-1003 | Kavya Nair | Starter | Active | $0 |

### Knowledge Base Topics
`api` · `login` · `billing` · `update` · `2fa` · `sdk`

---

## State Schema (`SupportState`)

All 17 fields are defined in Session 1 and never removed. Each session populates more of them.

| Field | Type | Populated |
|-------|------|-----------|
| `raw_input` | str | Session 1 |
| `sanitized_input` | str | Session 6 |
| `category` | str | Session 1 |
| `messages` | list | Session 2 |
| `customer_data` | dict | Session 2 |
| `tool_results` | list | Session 2 |
| `pii_detected` | bool | Session 6 |
| `injection_detected` | bool | Session 6 |
| `is_safe` | bool | Session 1 |
| `system_summary` | str | Session 5 |
| `iteration_count` | int | Session 3 |
| `internal_notes` | list | Session 8 |
| `delegation_count` | int | Session 8 |
| `next_worker` | str | Session 8 |
| `github_draft` | dict | Session 10 |
| `github_issue_url` | str | Session 11 |
| `final_response` | str | Session 1 |

---

## Session Roadmap

### ✅ Session 1 — The Blueprint
**What was built:** Foundational skeleton.
- `SupportState` TypedDict (17 fields, all 12 sessions scoped)
- `classify_node` — Gemini-powered 4-category classifier with 3-layer defense
- `route_by_category` — deterministic conditional router
- 4 handler stubs (technical, billing, fraud, general)
- `run_ticket()` and `stream_ticket()` entry points
- FastAPI backend + dark glassmorphism UI

**Key lesson:** LangGraph graph compilation, conditional edges, state schema design.

---

### ✅ Session 2 — Tool Binding & Execution ← **You are here**
**What was built:** Real tool calling infrastructure.
- `MOCK_CRM` — 3 customer records with needed + noisy field separation
- `MOCK_KB` — 6 knowledge base articles
- `get_customer_details` tool — 3-layer: arg validation → CRM lookup → field filtering
- `search_knowledge_base` tool — keyword matching with fallback guidance
- `agent_node` — `llm.bind_tools(TOOLS)` with explicit tool usage rules in system prompt
- `route_after_agent` — pure Python: tool_calls present → loop, else → respond
- `respond_node` — extracts final AIMessage content (handles Gemini structured blocks)
- `tool_node` — `ToolNode(TOOLS)` executing both tools
- Graph rebuilt: 8 nodes, billing+technical → agent_node, agent↔tool loop
- UI: 🔧 Tool Call Panel, richer timeline labels, updated verification table

**Key lesson:** `bind_tools()`, `ToolNode`, agent-tool loop wiring, docstring-as-schema.

---

### 🔒 Session 3 — The ReAct Architecture
**What gets added:**
- `MAX_ITERATIONS = 5` circuit breaker — prevents infinite loops
- `CONTEXT_THRESHOLD = 12` rolling window — trims message history before LLM calls
- `build_escalation_response()` — graceful user-facing message when circuit breaker fires
- `trim_context()` — keeps first user message + last N, prevents context explosion
- `get_tool_fingerprint()` — detects duplicate tool calls (same name + args)
- `agent_node` gains 3 safety layers around existing logic (circuit breaker, trim, dedup)
- `check_fraud_signals` tool + `fraud_handler` replaced with real tool-calling node
- `iteration_count` field in `SupportState` starts being incremented

**Key lesson:** ReAct reasoning loop safety, context window management, escalation patterns.

---

### 🔒 Session 4 — The Memory
**What gets added:**
- LangGraph `SqliteSaver` checkpointer — thread persistence across server restarts
- `thread_id` in `run_ticket()` and `stream_ticket()` — each customer conversation has its own thread
- Multi-turn conversation support — agent can reference prior messages
- `/api/history/{thread_id}` endpoint — fetch conversation history
- UI conversation history panel

**Key lesson:** LangGraph checkpointers, thread IDs, stateful conversations.

---

### 🔒 Session 5 — Context Compression
**What gets added:**
- `system_summary` field populated — LLM-generated rolling summary of long conversations
- Summarization node fires when message count exceeds threshold
- Summary injected as SystemMessage at top of context instead of full history

**Key lesson:** Long-context management, summarization patterns.

---

### 🔒 Session 6 — The Shield (Safety Layer)
**What gets added:**
- `pii_detected` — regex + LLM scan for emails, phone numbers, credit cards
- `injection_detected` — prompt injection classifier
- `is_safe` gate — unsafe tickets short-circuit to a safety response node
- PII redacted from `sanitized_input` before it reaches any tool

**Key lesson:** Input validation, PII handling, adversarial robustness.

---

### 🔒 Session 7 — Structured Outputs
**What gets added:**
- Pydantic response schemas — every handler returns validated structured output
- `response_format` enforcement — no free-form text from billing/technical agents
- Typed tool return schemas

**Key lesson:** LLM output reliability, Pydantic validation, structured generation.

---

### 🔒 Session 8 — The Supervisor
**What gets added:**
- Supervisor agent — decides which specialist worker to delegate to
- `delegation_count` and `next_worker` fields used
- `internal_notes` scratchpad shared across parallel workers
- Worker agents: billing-specialist, technical-specialist

**Key lesson:** Multi-agent orchestration, supervisor-worker pattern.

---

### 🔒 Session 9 — The Swarm (Parallel Agents)
**What gets added:**
- `fraud_handler` fully replaced — parallel fraud analysis agents
- `check_fraud_signals` tool (from Session 3 stub) fully wired
- Multiple fraud sub-agents run in parallel: transaction analysis, account history, geo-risk
- Results merged before final response

**Key lesson:** Parallel graph execution, `Send()` API, result aggregation.

---

### 🔒 Session 10 — Write Access
**What gets added:**
- GitHub issue creation tool — agent can draft and propose GitHub issues
- `github_draft` field populated by agent
- Human-in-the-loop approval before issue is actually created
- `/api/approve` endpoint

**Key lesson:** Write-capable agents, human approval gates, side effects.

---

### 🔒 Session 11 — The Gatekeeper (Human Approval)
**What gets added:**
- LangGraph `interrupt()` — graph pauses at approval node
- Full approval/rejection flow with reason capture
- `github_issue_url` populated after approval
- UI approval modal with diff view

**Key lesson:** LangGraph interrupts, human-in-the-loop architecture.

---

### 🔒 Session 12 — The Auditor (Observability)
**What gets added:**
- Full audit timeline — every node execution logged with timestamps
- LangGraph time-travel — replay any prior state snapshot
- `/api/audit/{thread_id}` endpoint — full execution history
- UI audit timeline panel with state diff viewer

**Key lesson:** LangGraph replay, observability, production debugging patterns.

---

## Running the Verification Suite

```bash
# From CLI
python support_agent.py

# From the UI
# 1. Start the server: python api.py
# 2. Open http://localhost:8000
# 3. Scroll to "Session 2 — Verification Test"
# 4. Click "▶ Run Verification Test"

# From the API
curl -X POST http://localhost:8000/api/verify
```

### Session 2 Verification Checks
| # | Check | Pass Criteria |
|---|-------|---------------|
| 1 | Billing ticket with customer ID | `get_customer_details` called + non-empty response |
| 2 | Technical issue | `search_knowledge_base` called + non-empty response |
| 3 | Billing without customer ID | Agent asks for ID, no blind tool call, non-empty response |
| 4 | Invalid customer ID | Tool returns error, agent responds gracefully, no crash |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google AI Studio API key for Gemini 2.5 Flash |

Get a key at: [aistudio.google.com](https://aistudio.google.com)

---

## Dependencies (key packages)

| Package | Purpose |
|---------|---------|
| `langgraph` | Agent graph orchestration |
| `langchain-core` | Tool decorators, message types |
| `langchain-google-genai` | Gemini LLM integration |
| `fastapi` | REST + SSE API server |
| `uvicorn` | ASGI server |
| `python-dotenv` | `.env` file loading |
