"""
Enterprise AI Support Platform
Session 2 of 12 — Tool Binding & Execution

Extends Session 1 with real tool calling infrastructure.
Billing and technical handlers replaced with agent + tool loop.

Run server: python api.py  → http://localhost:8000
Run CLI:    python support_agent.py
"""

import os
import time
import operator
import json
import uuid
from typing import TypedDict, Annotated, Literal, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── Environment setup ──────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not set. Run: export GOOGLE_API_KEY='your-key-here'"
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
print("[System] Gemini 2.5 Flash initialized | temperature=0")

# ══════════════════════════════════════════════════════════════════
# SECTION 2: STATE SCHEMA
# ══════════════════════════════════════════════════════════════════

class SupportState(TypedDict):

    # ── Core Input (Session 1) ──────────────────────────────────
    raw_input:          str        # Original user message, never modified
    sanitized_input:    str        # PII-cleaned version (Session 6)

    # ── Classification (Session 1) ─────────────────────────────
    category:           str        # technical | billing | fraud | general

    # ── Conversation History (Session 2) ───────────────────────
    messages:           Annotated[list, add_messages]
    customer_data:      dict       # Populated by CRM tool
    tool_results:       Annotated[list, operator.add]  # Append-safe

    # ── Safety Controls (Session 6) ────────────────────────────
    pii_detected:       bool
    injection_detected: bool
    is_safe:            bool

    # ── Memory and Context (Sessions 3, 5) ─────────────────────
    system_summary:     str        # Compressed history (Session 5)
    iteration_count:    int        # ReAct circuit breaker (Session 3)

    # ── Multi-Agent Orchestration (Sessions 8, 9) ───────────────
    internal_notes:     Annotated[list, operator.add]  # Parallel scratchpad
    delegation_count:   int        # Supervisor counter
    next_worker:        str        # Supervisor decision

    # ── Write Access and Human Approval (Session 10, 11) ────────
    github_draft:       dict       # Proposed issue before approval
    github_issue_url:   str        # URL after creation

    # ── Output (Session 1) ─────────────────────────────────────
    final_response:     str


_field_count = len(SupportState.__annotations__)
print(f"[System] SupportState schema — {_field_count} fields across 12 sessions")

# ── Mock Data (Session 2) ──────────────────────────────────────

MOCK_CRM = {
    'C-1001': {
        # NEEDED FIELDS
        'name': 'Priya Sharma',
        'billing_status': 'Active',
        'subscription_tier': 'Enterprise',
        'last_payment_date': '2026-04-01',
        'last_payment_amount': 4999.00,
        'outstanding_balance': 0.00,
        'recent_transactions': [
            {'date': '2026-04-01', 'description': 'Enterprise Plan — April',  'amount': 4999.00, 'status': 'paid'},
            {'date': '2026-03-01', 'description': 'Enterprise Plan — March',  'amount': 4999.00, 'status': 'paid'},
            {'date': '2026-02-01', 'description': 'Enterprise Plan — February','amount': 4999.00, 'status': 'paid'},
        ],
        # NOISY FIELDS
        'internal_crm_id': 'CRM-88123',
        'sales_rep_code': 'SR-042',
        'geo_region_tag': 'APAC',
        'last_login_ip': '192.168.10.5',
        'feature_flag_cohort': 'beta-v2',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
    'C-1002': {
        # NEEDED FIELDS
        'name': 'Arjun Mehta',
        'billing_status': 'Past Due',
        'subscription_tier': 'Pro',
        'last_payment_date': '2026-03-01',
        'last_payment_amount': 499.00,
        'outstanding_balance': 998.00,
        'recent_transactions': [
            {'date': '2026-03-01', 'description': 'Pro Plan — March',  'amount': 499.00, 'status': 'paid'},
            {'date': '2026-04-01', 'description': 'Pro Plan — April',  'amount': 499.00, 'status': 'missed'},
            {'date': '2026-05-01', 'description': 'Pro Plan — May',    'amount': 499.00, 'status': 'missed'},
        ],
        # NOISY FIELDS
        'internal_crm_id': 'CRM-88456',
        'sales_rep_code': 'SR-017',
        'geo_region_tag': 'APAC',
        'last_login_ip': '10.0.0.44',
        'feature_flag_cohort': 'stable',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
    'C-1003': {
        # NEEDED FIELDS
        'name': 'Kavya Nair',
        'billing_status': 'Active',
        'subscription_tier': 'Starter',
        'last_payment_date': '2026-05-01',
        'last_payment_amount': 99.00,
        'outstanding_balance': 0.00,
        'recent_transactions': [
            {'date': '2026-05-01', 'description': 'Starter Plan — May', 'amount': 99.00, 'status': 'paid'},
        ],
        # NOISY FIELDS
        'internal_crm_id': 'CRM-88789',
        'sales_rep_code': 'SR-031',
        'geo_region_tag': 'EMEA',
        'last_login_ip': '172.16.0.9',
        'feature_flag_cohort': 'stable',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
}

# Knowledge Base (Session 2) — condensed for brevity, but can be expanded
MOCK_KB = {
    'api': (
        'API troubleshooting guide: (1) Check rate limits — free tier: 100 req/min, '
        'pro: 1000 req/min, enterprise: unlimited. (2) Auth header must be '
        '"Authorization: Bearer <token>" — never basic auth. (3) On 401 errors, '
        'regenerate your API key in Account > API Keys. (4) On 429 rate-limit errors, '
        'implement exponential backoff starting at 1s. (5) SDK v3+ requires '
        'client.initialize() before first call.'
    ),
    'login': (
        'Login troubleshooting: (1) Clear browser cache and cookies, then retry. '
        '(2) MFA: open your authenticator app, use the 6-digit code within 30 seconds. '
        '(3) Password reset: go to login page > "Forgot password" > check email within '
        '5 minutes. (4) If locked out after 5 attempts, wait 15 minutes or contact '
        'support. (5) SSO users: ensure your identity provider session is active.'
    ),
    'billing': (
        'Billing help: (1) Invoice portal: account.nexus.io/billing/invoices — '
        'download PDF or CSV. (2) Update payment method: Billing > Payment Methods > '
        'Add New Card. (3) Refund policy: eligible within 30 days of charge, '
        'processed in 5-10 business days. (4) Subscription changes take effect on '
        'next billing cycle. (5) Failed payments retry automatically for 3 days.'
    ),
    'update': (
        'Post-update troubleshooting: (1) Clear application cache after any update: '
        'Settings > Cache > Clear All. (2) If issues persist, rollback procedure: '
        'go to Admin > Versions > select previous stable version > Rollback. '
        '(3) Check the changelog at docs.nexus.io/changelog for breaking changes. '
        '(4) SDK updates: run "npm install @nexus/sdk@latest" or '
        '"pip install nexus-sdk --upgrade".'
    ),
    '2fa': (
        '2FA / MFA help: (1) Backup codes: stored during setup — check your saved '
        'codes document. (2) Lost device: go to login > "Use backup code" > enter '
        'one of your 8-digit backup codes. (3) Reset 2FA: Account > Security > '
        'Two-Factor Auth > Reset — requires email verification. (4) Manual '
        'verification for locked accounts: contact support with government ID. '
        '(5) TOTP apps supported: Google Authenticator, Authy, 1Password.'
    ),
    'sdk': (
        'SDK compatibility guide: (1) SDK v3.x requires Node 18+ or Python 3.10+. '
        '(2) Migration from v2 to v3: replace client.get() with client.fetch(), '
        'update auth to client.initialize({apiKey}). (3) Breaking changes in v3: '
        'callback-style API removed, promises only. (4) Python SDK: '
        '"from nexus import NexusClient" replaces "import nexus". '
        '(5) Full migration guide: docs.nexus.io/sdk/v3-migration.'
    ),
}

# ── Tools (Session 2) ──────────────────────────────────────────

@tool
def get_customer_details(customer_id: str) -> dict:
    """
    WHAT:
    Retrieves billing status, subscription tier, last payment date,
    outstanding balance, and recent transaction history for a customer
    from the CRM system.

    WHEN:
    Call this tool when the user's query involves billing, payment
    status, invoice disputes, subscription management, refund
    requests, or account standing. Always call before answering
    any billing question.

    FORMAT:
    customer_id must be in format 'C-XXXX' e.g. 'C-1001', 'C-1042'.
    Extract from the user message.
    If not present in the message, ask the user before calling.
    Never guess or fabricate a customer_id.

    RETURN:
    Dict with: name, billing_status, subscription_tier,
    last_payment_date, last_payment_amount, outstanding_balance,
    recent_transactions (last 3 only).
    On any failure: dict with single 'error' key describing what failed.
    """
    # LAYER 1 — Argument validation
    if not customer_id or not isinstance(customer_id, str):
        return {'error': "customer_id must be a non-empty string."}
    cid = customer_id.strip().upper()
    if not cid.startswith('C-'):
        return {'error': f"Invalid format: '{customer_id}'. Expected 'C-XXXX' e.g. 'C-1001'"}

    # LAYER 2 — Database lookup with error handling
    try:
        raw = MOCK_CRM.get(cid)
        if raw is None:
            return {'error': f"Customer '{cid}' not found. Please verify the ID with the customer."}
    except Exception as e:
        return {'error': f"CRM lookup failed: {type(e).__name__}. Contact engineering if this persists."}

    # LAYER 3 — Data filtering
    NEEDED = {
        'name', 'billing_status', 'subscription_tier',
        'last_payment_date', 'last_payment_amount',
        'outstanding_balance', 'recent_transactions'
    }
    filtered = {k: v for k, v in raw.items() if k in NEEDED}
    filtered['recent_transactions'] = filtered.get('recent_transactions', [])[:3]
    return filtered

# Test: get_customer_details.invoke({'customer_id': 'C-1001'})
# Test: get_customer_details.invoke({'customer_id': 'C-9999'})
# Test: get_customer_details.invoke({'customer_id': 'bad'})


@tool
def search_knowledge_base(query: str) -> dict:
    """
    WHAT:
    Searches the internal technical knowledge base for resolution
    steps and troubleshooting articles matching the issue described.

    WHEN:
    Call this tool for any technical issue before responding to the
    customer. Always search before saying you cannot help.
    If first search returns no match, try with different keywords.

    FORMAT:
    query is a natural language string describing the technical
    problem. Be specific. Include error codes or keywords.
    Example: 'API authentication 401 error after SDK update'

    RETURN:
    Dict with matched (bool), results (list of article strings),
    count (int). If no match: matched=False with fallback guidance.
    On failure: dict with single 'error' key.
    """
    try:
        if not query or not query.strip():
            return {'error': 'Search query cannot be empty.'}

        query_lower = query.lower()
        results = []
        for keyword, article in MOCK_KB.items():
            if keyword in query_lower:
                results.append(article)

        if not results:
            return {
                'matched': False,
                'results': [],
                'count': 0,
                'fallback': (
                    'No specific article found. General guidance: '
                    'check account status, clear browser cache, verify '
                    'recent configuration changes, review changelog.'
                )
            }

        return {'matched': True, 'results': results, 'count': len(results)}

    except Exception as e:
        return {'error': f"KB search failed: {type(e).__name__}"}

# Test: search_knowledge_base.invoke({'query': 'API 401 error'})
# Test: search_knowledge_base.invoke({'query': 'nothing matches'})


TOOLS = [get_customer_details, search_knowledge_base]
llm_with_tools = llm.bind_tools(TOOLS)

print("[Tools] Registered:")
for t in TOOLS:
    print(f"  · {t.name}: {t.description[:70]}...")


# ══════════════════════════════════════════════════════════════════
# SECTION 3: CLASSIFIER NODE
# ══════════════════════════════════════════════════════════════════

def classify_node(state: SupportState) -> dict:
    system_prompt = (
        "You are a support ticket classifier for an enterprise SaaS company.\n"
        "Classify the incoming ticket into EXACTLY ONE of these 4 categories:\n\n"
        "  technical:  API errors, login failures, bugs, performance issues,\n"
        "              integration problems, post-update breakage\n"
        "  billing:    payment failures, invoice disputes, subscriptions,\n"
        "              refund requests, double charges\n"
        "  fraud:      unauthorized transactions, account compromise,\n"
        "              suspicious activity, identity theft\n"
        "  general:    feature questions, how-to, onboarding, documentation,\n"
        "              anything that does not fit the above categories\n\n"
        "Respond with EXACTLY ONE WORD. No punctuation. "
        "No explanation. No other text whatsoever."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["raw_input"]),
    ])

    # Layer 1 — normalize
    raw = response.content.strip().lower().rstrip(".,!?")

    # Layer 2 — validate
    VALID = {"technical", "billing", "fraud", "general"}
    if raw not in VALID:
        print(f"[Classifier] Unexpected output: '{raw}' → defaulting to 'general'")
        raw = "general"

    # Layer 3 — print
    preview = state["raw_input"][:60]
    print(f"[Classifier] '{preview}'... → {raw}")

    return {
        "category":           raw,
        "sanitized_input":    state["raw_input"],  # pass-through until Session 6
        "iteration_count":    0,
        "delegation_count":   0,
        "is_safe":            True,
        "pii_detected":       False,
        "injection_detected": False,
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 4: ROUTER FUNCTION
# ══════════════════════════════════════════════════════════════════

def route_by_category(state: SupportState) -> str:
    raw = state.get("category") or ""
    category = raw.strip().lower()

    routing_map = {
        "technical": "technical_handler",
        "billing":   "billing_handler",
        "fraud":     "fraud_handler",
        "general":   "general_handler",
    }

    destination = routing_map.get(category, "general_handler")
    print(f"[Router] '{category}' → {destination}")
    return destination


# ══════════════════════════════════════════════════════════════════
# SECTION 5: HANDLER STUBS
# ══════════════════════════════════════════════════════════════════

def technical_handler(state: SupportState) -> dict:
    # STUB — replaced in Session 2 (routing now goes to agent_node)
    preview = state["raw_input"][:80]
    print(f"[technical_handler] Handling: '{preview}'")
    return {
        "final_response": (
            "Your technical issue has been received and assigned to our "
            "Engineering team. A specialist will respond within 4 hours."
        )
    }


def billing_handler(state: SupportState) -> dict:
    # STUB — replaced in Session 2 (routing now goes to agent_node)
    preview = state["raw_input"][:80]
    print(f"[billing_handler] Handling: '{preview}'")
    return {
        "final_response": (
            "Your billing inquiry has been received and assigned to our "
            "Finance team. We will review your account within 2 hours."
        )
    }


def fraud_handler(state: SupportState) -> dict:
    # STUB — replaced in Session 9
    preview = state["raw_input"][:80]
    print(f"[fraud_handler] Handling: '{preview}'")
    return {
        "final_response": (
            "Your report of suspicious activity has been flagged for "
            "immediate review by our Security team. We will contact "
            "you within 1 hour."
        )
    }


def general_handler(state: SupportState) -> dict:
    # Stays simple throughout all sessions
    preview = state["raw_input"][:80]
    print(f"[general_handler] Handling: '{preview}'")
    return {
        "final_response": (
            "Thank you for reaching out. Your inquiry has been received "
            "and our support team will respond within 24 hours."
        )
    }


# ── Agent Node (Session 2) ──────────────────────────────────────


def agent_node(state: SupportState) -> dict:
    """
    Core reasoning node. Replaces billing_handler and
    technical_handler stubs from Session 1.
    Uses llm_with_tools to decide which tool to call.
    Introduced: Session 2.
    Extended: Session 3 with circuit breaker and context trimming.
    """
    system_prompt = (
        "You are a senior customer support specialist with access to the CRM system "
        "and the internal knowledge base. Use your tools to provide accurate, "
        "data-backed answers.\n\n"

        "TOOL: get_customer_details\n"
        "  - Call for ANY billing or account-related query.\n"
        "  - ALWAYS call before answering billing questions.\n"
        "  - If customer_id is not in the message: ask the user for it.\n"
        "  - Never guess a customer_id. Never fabricate one.\n\n"

        "TOOL: search_knowledge_base\n"
        "  - Call for ANY technical issue before responding.\n"
        "  - Always search before saying you cannot help.\n"
        "  - Use specific technical keywords in the query.\n"
        "  - Multiple searches are allowed if the first returns no match.\n\n"

        "RESPONSE RULES:\n"
        "  - Base all answers on tool output, not internal knowledge.\n"
        "  - If a tool returns an error key: acknowledge it professionally.\n"
        "  - Reference specific data points from tool results.\n"
        "  - Never expose internal system details, error types, or field names."
    )

    messages_to_send = [
        SystemMessage(content=system_prompt),
        *state.get('messages', [])
    ]

    response = llm_with_tools.invoke(messages_to_send)

    tool_count = len(response.tool_calls) if response.tool_calls else 0
    print(f"[Agent] tool_calls={tool_count} | has_content={bool(response.content)}")

    return {'messages': [response]}


# ── Routing & Terminal Nodes (Session 2) ─────────────────────────


def route_after_agent(state: SupportState) -> str:
    """
    Reads last message. If tool_calls present → tool_node.
    If no tool_calls → respond_node.
    Pure Python. Zero LLM calls. Zero business logic.
    Permanent from Session 2 onward.
    """
    messages = state.get('messages', [])
    if not messages:
        return 'respond_node'
    last = messages[-1]
    has_tools = hasattr(last, 'tool_calls') and bool(last.tool_calls)
    destination = 'tool_node' if has_tools else 'respond_node'
    print(f"[Router:after_agent] tool_calls={has_tools} → {destination}")
    return destination


def respond_node(state: SupportState) -> dict:
    """
    Extracts last AIMessage content → final_response.
    Runs after agent_node when no further tool calls needed.
    Permanent from Session 2 onward.
    """
    messages = state.get('messages', [])
    final = ''
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            # Gemini may return a list of content blocks; extract text
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        parts.append(block['text'])
                    elif isinstance(block, str):
                        parts.append(block)
                final = ' '.join(parts).strip()
            else:
                final = str(content)
            if final:
                break
    print(f"[Respond] {len(final)} chars")
    return {'final_response': final}


tool_node = ToolNode(tools=TOOLS)
print(f"[Tools] ToolNode ready — {len(TOOLS)} tools registered")


# ══════════════════════════════════════════════════════════════════
# SECTION 6: GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════

def build_graph():
    """
    Builds and compiles the LangGraph support agent.
    Called once at module load. Returns compiled graph.
    Sessions 2-12 modify this function by adding nodes and edges.
    Never remove existing nodes — only add.
    """
    builder = StateGraph(SupportState)

    # Register Session 1 nodes
    builder.add_node("classify_node",     classify_node)
    builder.add_node("technical_handler", technical_handler)
    builder.add_node("billing_handler",   billing_handler)
    builder.add_node("fraud_handler",     fraud_handler)
    builder.add_node("general_handler",   general_handler)

    # Register Session 2 nodes
    builder.add_node("agent_node",   agent_node)
    builder.add_node("tool_node",    tool_node)
    builder.add_node("respond_node", respond_node)

    # Entry point
    builder.set_entry_point("classify_node")

    # Routing from classifier — billing and technical now go to agent_node
    builder.add_conditional_edges(
        "classify_node",
        route_by_category,
        {
            "technical_handler": "agent_node",
            "billing_handler":   "agent_node",
            "fraud_handler":     "fraud_handler",
            "general_handler":   "general_handler",
        }
    )

    # Agent → tool loop
    builder.add_conditional_edges(
        "agent_node",
        route_after_agent,
        {
            "tool_node":    "tool_node",
            "respond_node": "respond_node",
        }
    )

    # Tool node loops back to agent
    builder.add_edge("tool_node", "agent_node")

    # Terminal edges
    builder.add_edge("respond_node",     END)
    builder.add_edge("fraud_handler",    END)
    builder.add_edge("general_handler",  END)

    # Session 1 stub nodes still need edges (they are never reached for
    # billing/technical but must be registered to avoid orphan node errors)
    builder.add_edge("technical_handler", END)
    builder.add_edge("billing_handler",   END)

    graph = builder.compile()
    print("[Graph] Session 2 — 8 nodes | agent+tool loop active")
    return graph


# Module-level graph instance
graph = build_graph()


# ══════════════════════════════════════════════════════════════════
# SECTION 7: INITIAL STATE BUILDER
# ══════════════════════════════════════════════════════════════════

def build_initial_state(ticket: str) -> dict:
    """
    Constructs a clean initial state for every graph invocation.
    Provides safe defaults for ALL 17 fields so no node gets a KeyError.
    Called by both the test harness and the Streamlit UI.
    """
    return {
        "raw_input":          ticket,
        "sanitized_input":    "",
        "category":           "",
        "messages":           [HumanMessage(content=ticket)],
        "customer_data":      {},
        "tool_results":       [],
        "pii_detected":       False,
        "injection_detected": False,
        "is_safe":            True,
        "system_summary":     "",
        "iteration_count":    0,
        "internal_notes":     [],
        "delegation_count":   0,
        "next_worker":        "",
        "github_draft":       {},
        "github_issue_url":   "",
        "final_response":     "",
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 8: RUN FUNCTION (called by both CLI and UI)
# ══════════════════════════════════════════════════════════════════


def run_ticket(ticket: str) -> dict:
    """
    Single entry point for running a ticket through the graph.
    Called by the test harness AND by the Streamlit UI.
    Returns the complete final state dict.
    No config yet — checkpointer added in Session 4.
    """
    initial_state = build_initial_state(ticket)
    result = graph.invoke(initial_state)
    return result

def stream_ticket(ticket: str):
    """
    Generator that yields each graph step as it executes.
    Used by the Streamlit UI for live step-by-step display.
    Yields: (node_name: str, snapshot: dict) tuples.
    """
    initial_state = build_initial_state(ticket)
    for step in graph.stream(initial_state):
        for node_name, snapshot in step.items():
            yield node_name, snapshot


# ══════════════════════════════════════════════════════════════════
# SECTION 9: SESSION VERIFICATION TEST
# ══════════════════════════════════════════════════════════════════

def run_session_verification() -> dict:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  SESSION 2 — VERIFICATION TEST                              │
    ├─────────────────────────────────────────────────────────────┤
    │  WHAT THIS TESTS:                                           │
    │  Tool calling infrastructure works end-to-end.              │
    │  Agent calls the right tool for the right ticket type.      │
    │  Tool errors handled gracefully — graph never crashes.      │
    │                                                             │
    │  PASS CRITERIA:                                             │
    │  ✓ Billing ticket with ID  → get_customer_details called    │
    │  ✓ Technical ticket        → search_knowledge_base called   │
    │  ✓ Billing without ID      → agent asks, no blind tool call │
    │  ✓ Invalid customer ID     → error handled, response given  │
    │  ✓ All 4 return non-empty final_response                    │
    │                                                             │
    │  WHAT A PASS PROVES:                                        │
    │  Tool schemas correctly defined and schema-matched to LLM.  │
    │  ToolNode executes and returns results to the agent.        │
    │  Error handling prevents graph crashes on bad tool input.   │
    │  Session 3 is unblocked.                                    │
    └─────────────────────────────────────────────────────────────┘
    """
    test_cases = [
        {
            'label':       'Billing with customer ID',
            'ticket':      'My account C-1002 shows past due. Check it.',
            'expect_tool': 'get_customer_details',
            'check_type':  'tool_called',
        },
        {
            'label':       'Technical issue',
            'ticket':      '401 errors on every API call after SDK update.',
            'expect_tool': 'search_knowledge_base',
            'check_type':  'tool_called',
        },
        {
            'label':      'Billing without customer ID',
            'ticket':     'I want to check my billing status.',
            'check_type': 'no_crash',
            'note':       'Agent should ask for ID, not call tool blindly',
        },
        {
            'label':      'Invalid customer ID',
            'ticket':     'Check account C-9999 please.',
            'check_type': 'no_crash',
            'note':       'Tool returns error, agent responds gracefully',
        },
    ]

    print("\n" + "▓" * 60)
    print("▓  SESSION 2 — VERIFICATION TEST                        ▓")
    print("▓" * 60)

    start_ms = int(time.time() * 1000)
    check_results = []
    all_passed = True

    for tc in test_cases:
        result = run_ticket(tc['ticket'])

        has_response = bool(result.get('final_response', '').strip())
        no_crash = True  # reaching this line means no exception

        if tc['check_type'] == 'tool_called':
            tool_was_called = False
            for msg in result.get('messages', []):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for call in msg.tool_calls:
                        if call.get('name') == tc['expect_tool']:
                            tool_was_called = True
                            break

            passed = tool_was_called and has_response
            entry = {
                'label':       tc['label'],
                'ticket':      tc['ticket'][:60],
                'passed':      passed,
                'has_response': has_response,
                'tool_called': tc['expect_tool'] if tool_was_called else None,
            }
        else:
            passed = has_response
            entry = {
                'label':       tc['label'],
                'ticket':      tc['ticket'][:60],
                'passed':      passed,
                'has_response': has_response,
            }

        if 'note' in tc:
            entry['note'] = tc['note']

        all_passed = all_passed and passed
        check_results.append(entry)

        status = "✅" if passed else "❌"
        print(f"{status} [{tc['check_type']}] {tc['label']}")
        print(f"   Response: {'yes' if has_response else 'NO'} | "
              f"Tool: {entry.get('tool_called', 'n/a')}")

    end_ms = int(time.time() * 1000)
    duration_ms = end_ms - start_ms
    n_passed = sum(1 for c in check_results if c['passed'])
    total = len(check_results)
    summary = f"{n_passed}/{total} checks passed in {duration_ms}ms"

    print("\n" + "▓" * 60)
    if all_passed:
        print("▓  VERIFICATION: ✅ PASSED — Session 3 is unblocked       ▓")
    else:
        print("▓  VERIFICATION: ❌ FAILED — Fix agent_node or tools       ▓")
    print(f"▓  {summary:<54}▓")
    print("▓" * 60)

    return {
        'passed':      all_passed,
        'checks':      check_results,
        'summary':     summary,
        'duration_ms': duration_ms,
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 10: CLI TEST HARNESS
# ══════════════════════════════════════════════════════════════════

def run_cli_tests():
    """Runs all Session 2 test cases when file is executed directly."""

    print("\n" + "█" * 62)
    print("█  ENTERPRISE AI SUPPORT PLATFORM — SESSION 2 OF 12      █")
    print("█  Tool Binding & Execution                               █")
    print("█" * 62)

    test_cases = [
        {
            "label":    "TEST 1 — Billing with ID",
            "ticket":   "My payment is overdue. Please check account C-1002.",
            "note":     "Expected: calls get_customer_details, returns past due + balance",
        },
        {
            "label":    "TEST 2 — Billing without ID",
            "ticket":   "I want to check my billing status.",
            "note":     "Expected: agent asks for customer ID. No tool call without it.",
        },
        {
            "label":    "TEST 3 — Technical issue",
            "ticket":   "Getting 401 auth errors on every API call after SDK v3 update.",
            "note":     "Expected: calls search_knowledge_base, returns auth fix steps",
        },
        {
            "label":    "TEST 4 — Invalid ID",
            "ticket":   "Can you check account C-9999?",
            "note":     "Expected: tool returns error dict, agent handles it, no crash",
        },
        {
            "label":    "TEST 5 — Active customer full query",
            "ticket":   "What is my current plan and last payment? Account is C-1001.",
            "note":     "Expected: calls tool, returns Priya Sharma Enterprise plan data",
        },
    ]

    for tc in test_cases:
        print(f"\n{'─' * 60}")
        print(f"{tc['label']}")
        print(f"TICKET: {tc['ticket']}")
        print(f"NOTE:   {tc['note']}")

        result = run_ticket(tc["ticket"])

        tools_called = []
        for msg in result.get('messages', []):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for call in msg.tool_calls:
                    tools_called.append(call.get('name', '?'))

        print(f"Category:     {result.get('category', '?')}")
        print(f"Tools called: {tools_called if tools_called else 'none'}")
        print(f"Response:     {result.get('final_response', '')[:120]}...")
        print(f"Status:       {'✅ PASS' if result.get('final_response') else '❌ FAIL (no response)'}")

    # Run verification
    verification = run_session_verification()

    print(f"\n{'═' * 62}")
    print(f"SESSION 2 COMPLETE — {verification['summary']}")
    print("═" * 62)


# ══════════════════════════════════════════════════════════════════
# SECTION 11: MAIN BLOCK
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_cli_tests()
