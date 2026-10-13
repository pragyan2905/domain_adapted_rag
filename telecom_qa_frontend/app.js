/* ===================================================================
   TelecomRAG — app.js  (v5)
   Uses SSE streaming endpoint for real-time pipeline visualization.
   =================================================================== */

const API_BASE = 'http://localhost:8001';

// ── State ────────────────────────────────────────────────────────────
let sessionId = localStorage.getItem('telecomrag_session') || null;
let isProcessing = false;

// ── DOM refs ─────────────────────────────────────────────────────────
const queryInput      = document.getElementById('queryInput');
const submitBtn       = document.getElementById('submitBtn');
const chatArea        = document.getElementById('chatArea');
const welcomeScreen   = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const pipelineNav     = document.getElementById('pipelineNav');
const healthIndicator = document.getElementById('healthIndicator');
const sessionBadge    = document.getElementById('sessionIdText');
const newChatBtn      = document.getElementById('newChatBtn');
const clearSessionLink = document.getElementById('clearSessionLink');
const sidebarToggle       = document.getElementById('sidebarToggle');
const topbarSidebarToggle = document.getElementById('topbarSidebarToggle');
const sidebar             = document.querySelector('.sidebar');

// ── Pipeline step definitions (for the sidebar nav) ──────────────────
const PIPELINE_STEPS = [
    { key: 'Step 01: Understanding',        label: 'Query Understanding' },
    { key: 'Step 02: Representation',       label: 'Embedding (BGE-M3)' },
    { key: 'Step 03: Hybrid Retrieval',     label: 'Hybrid Retrieval' },
    { key: 'Step 04: RRF Fusion',           label: 'RRF Fusion' },
    { key: 'Step 05: Filtering',            label: 'Threshold Filtering' },
    { key: 'Step 06: Cross-Encoder Rerank', label: 'Cross-Encoder Rerank' },
    { key: 'Step 07: Evidence Selection',   label: 'Evidence Selection' },
    { key: 'Step 08: Validation',           label: 'LLM-as-Judge Validation' },
    { key: 'Step 09: Web Fallback',         label: 'Web Fallback' },
    { key: 'Step 10: Abstention Gate',      label: 'Abstention Gate' },
    { key: 'Step 11: Generation (LLM)',     label: 'LLM Generation' },
    { key: 'Step 12: Guardrails',           label: 'Guardrails' },
];

// ── Init ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateSessionBadge();
    checkHealth();
    renderPipelineNav('idle');

    // Auto-resize textarea
    queryInput.addEventListener('input', () => {
        queryInput.style.height = 'auto';
        queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
    });

    // Submit on Enter (Shift+Enter = newline)
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    });

    submitBtn.addEventListener('click', handleSubmit);
    newChatBtn.addEventListener('click', startNewSession);
    clearSessionLink.addEventListener('click', (e) => { e.preventDefault(); startNewSession(); });

    function toggleSidebar() {
        if (window.innerWidth <= 768) {
            sidebar.classList.toggle('mobile-open');
        } else {
            sidebar.classList.toggle('collapsed');
        }
    }

    if (sidebarToggle) sidebarToggle.addEventListener('click', toggleSidebar);
    if (topbarSidebarToggle) topbarSidebarToggle.addEventListener('click', toggleSidebar);
});

// ── Session management ────────────────────────────────────────────────
function startNewSession() {
    sessionId = null;
    localStorage.removeItem('telecomrag_session');
    messagesContainer.innerHTML = '';
    welcomeScreen.style.display = '';
    renderPipelineNav('idle');
    updateSessionBadge();
}

function updateSessionBadge() {
    sessionBadge.textContent = sessionId ? `Session: ${sessionId.slice(0, 8)}…` : 'No active session';
}

// ── Health check ──────────────────────────────────────────────────────
async function checkHealth() {
    const dot  = healthIndicator.querySelector('.health-dot');
    const text = healthIndicator.querySelector('.health-text');
    try {
        const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
        const data = await r.json();
        if (data.status === 'healthy') {
            dot.className = 'health-dot healthy';
            text.textContent = `Healthy · ${data.qdrant_chunks ?? '?'} chunks`;
        } else {
            dot.className = 'health-dot degraded';
            text.textContent = 'Services degraded';
        }
    } catch {
        dot.className = 'health-dot error';
        text.textContent = 'Backend unreachable';
    }
}

// ── Pipeline nav renderer ─────────────────────────────────────────────
function renderPipelineNav(mode, stepStates = {}) {
    if (mode === 'idle') {
        pipelineNav.innerHTML = `
            <div class="pipeline-step-nav idle">
                <span class="step-dot"></span>
                <span class="step-label">Awaiting query…</span>
            </div>`;
        return;
    }

    pipelineNav.innerHTML = PIPELINE_STEPS.map((s, i) => {
        const state = stepStates[s.key] || 'pending';
        const meta  = stepStates[s.key + '_meta'] || '';
        return `
            <div class="pipeline-step-nav ${state}" id="nav-step-${i}">
                <span class="step-dot"></span>
                <span class="step-label">${s.label}</span>
                ${meta ? `<span class="step-meta">${meta}</span>` : ''}
            </div>`;
    }).join('');
}

// ── Submit handler ────────────────────────────────────────────────────
async function handleSubmit() {
    const query = queryInput.value.trim();
    if (!query || isProcessing) return;

    isProcessing = true;
    submitBtn.disabled = true;
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Hide welcome, show messages
    welcomeScreen.style.display = 'none';

    // Append user message
    appendMessage('user', query);

    // Start pipeline nav
    const stepStates = {};
    PIPELINE_STEPS.forEach(s => { stepStates[s.key] = 'pending'; });
    renderPipelineNav('active', stepStates);

    // Typing indicator (assistant bubble)
    const thinkingBubble = appendThinkingBubble();

    try {
        await streamQuery(query, stepStates, thinkingBubble);
    } catch (err) {
        thinkingBubble.remove();
        appendErrorMessage(err.message || 'Unknown error occurred.');
    } finally {
        isProcessing = false;
        submitBtn.disabled = false;
        checkHealth();
    }
}

// ── Streaming query ───────────────────────────────────────────────────
async function streamQuery(query, stepStates, thinkingBubble) {
    const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: sessionId }),
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // incomplete line goes back to buffer

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;

            try {
                const event = JSON.parse(jsonStr);
                handleStreamEvent(event, stepStates, thinkingBubble);
            } catch { /* ignore parse errors */ }
        }
    }
}

// ── Stream event handler ──────────────────────────────────────────────
function handleStreamEvent(event, stepStates, thinkingBubble) {
    switch (event.type) {
        case 'start':
            break;

        case 'cache_hit':
            updateStepState(stepStates, 'cache', 'done');
            break;

        case 'step_progress':
            updateStepStateByName(stepStates, event.step, 'running');
            break;

        case 'step_done': {
            const trace = event.trace || {};
            const metaText = trace.duration_ms != null ? `${trace.duration_ms}ms` : '';
            updateStepStateByName(stepStates, event.step, 'done', metaText);
            break;
        }

        case 'step_skipped':
            updateStepStateByName(stepStates, event.step, 'skipped');
            break;

        case 'error':
            thinkingBubble.remove();
            appendErrorMessage(event.message);
            renderPipelineNav('idle');
            break;

        case 'complete': {
            const result = event.result;
            // Save session
            if (result.session_id) {
                sessionId = result.session_id;
                localStorage.setItem('telecomrag_session', sessionId);
                updateSessionBadge();
            }
            // Replace thinking bubble with real answer
            thinkingBubble.remove();
            appendAssistantMessage(result);
            // Mark all done steps
            PIPELINE_STEPS.forEach(s => {
                if (stepStates[s.key] === 'running') stepStates[s.key] = 'done';
            });
            renderPipelineNav('active', stepStates);
            scrollToBottom();
            break;
        }
    }
}

// ── Step state helpers ────────────────────────────────────────────────
function updateStepStateByName(stepStates, stepName, state, meta = '') {
    // Find the closest matching key
    const key = PIPELINE_STEPS.find(s => stepName.includes(s.key) || s.key.includes(stepName))?.key
             || stepName;
    stepStates[key] = state;
    if (meta) stepStates[key + '_meta'] = meta;
    renderPipelineNav('active', stepStates);
}

function updateStepState(stepStates, stepKey, state) {
    stepStates[stepKey] = state;
    renderPipelineNav('active', stepStates);
}

// ── Message renderers ─────────────────────────────────────────────────
function appendMessage(role, text) {
    const el = document.createElement('div');
    el.className = `message ${role}`;
    el.innerHTML = `
        <div class="msg-avatar">${role === 'user' ? 'U' : '🛰️'}</div>
        <div class="msg-bubble">
            <div class="msg-text">${role === 'user' ? escapeHtml(text) : marked.parse(text)}</div>
        </div>`;
    messagesContainer.appendChild(el);
    scrollToBottom();
    return el;
}

function appendThinkingBubble() {
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
        <div class="msg-avatar">🛰️</div>
        <div class="msg-bubble">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>`;
    messagesContainer.appendChild(el);
    scrollToBottom();
    return el;
}

function appendAssistantMessage(result) {
    const el = document.createElement('div');
    el.className = 'message assistant';

    let contentHtml = '';

    // Cache hit badge
    if (result.pipeline_trace?.cache_hit) {
        contentHtml += `<div class="cache-hit-banner">
            ⚡ <strong>Semantic cache hit</strong> — response served instantly
        </div>`;
    }

    // Abstention
    if (result.is_abstained) {
        contentHtml += `<div class="abstention-banner">
            🚫 <div><strong>System Abstained</strong><br>${escapeHtml(result.abstention_reason || 'Insufficient evidence to answer safely.')}</div>
        </div>`;
    } else {
        // Warning banners
        if (result.guardrails?.answered_from_general_knowledge) {
            contentHtml += `<div class="warning-banner">⚠️ Answer generated from model's general knowledge — no 3GPP evidence retrieved.</div>`;
        }
        if (result.guardrails?.web_fallback_used) {
            contentHtml += `<div class="warning-banner">🌐 Evidence augmented via web search (local DB insufficient).</div>`;
        }

        // Main answer
        let answerText = result.answer || '';
        // Replace UUIDs in answer with [1], [2], etc.
        const citationMap = {};
        (result.citations || []).forEach((cit, i) => { citationMap[cit] = i + 1; });
        Object.entries(citationMap).forEach(([uuid, num]) => {
            answerText = answerText.replaceAll(uuid, `[${num}]`);
        });

        contentHtml += `<div class="msg-text">${marked.parse(answerText)}</div>`;

        // Confidence meter
        const conf = result.confidence_score || 0;
        const confPct = Math.round(conf * 100);
        if (conf > 0 && !result.guardrails?.answered_from_general_knowledge) {
            contentHtml += `<div class="confidence-row">
                Confidence
                <div class="confidence-bar"><div class="confidence-fill" style="width:${confPct}%"></div></div>
                <span>${confPct}%</span>
            </div>`;
        }

        // Citations
        if (result.citations?.length > 0) {
            contentHtml += `<div class="citations-section">
                <div class="citations-title">Citations</div>
                <div class="citation-pills">
                    ${result.citations.map((c, i) => `<span class="citation-pill" title="${c}">[${i+1}] ${c.length > 36 ? c.slice(0,36)+'…' : c}</span>`).join('')}
                </div>
            </div>`;
        }

        // Guardrails strip
        const gr = result.guardrails || {};
        const grBadges = [
            { label: '✓ Citations', cls: gr.unresolved_citations?.length ? 'fail' : 'pass',
              text: gr.unresolved_citations?.length ? `${gr.unresolved_citations.length} hallucinated` : 'All grounded' },
            { label: '✓ Numbers', cls: gr.ungrounded_numbers?.length ? 'warn' : 'pass',
              text: gr.ungrounded_numbers?.length ? `${gr.ungrounded_numbers.length} ungrounded` : 'Grounded' },
            { label: '⚡ Version', cls: gr.version_conflict_detected ? 'warn' : 'pass',
              text: gr.version_conflict_detected ? 'Cross-release conflict' : 'No conflict' },
            { label: '🌐 Source', cls: gr.web_fallback_used ? 'warn' : 'pass',
              text: gr.web_fallback_used ? 'Web + DB' : '3GPP DB only' },
        ];
        contentHtml += `<div class="guardrails-strip">
            ${grBadges.map(b => `<span class="gr-badge ${b.cls}" title="${b.text}">${b.label} · ${b.text}</span>`).join('')}
        </div>`;
    }

    // Pipeline Funnel
    const trace = result.pipeline_trace;
    if (trace?.steps?.length > 0) {
        const stepRows = trace.steps.map(s => {
            const countText = s.input_count !== s.output_count
                ? `${s.input_count} → ${s.output_count}`
                : `${s.output_count}`;
            return `<div class="funnel-step">
                <span class="funnel-step-name">${s.step_name}</span>
                <span class="funnel-step-count">${countText}</span>
                <span class="funnel-step-time">${s.duration_ms}ms</span>
            </div>`;
        }).join('');

        contentHtml += `<div class="pipeline-funnel">
            <div class="funnel-title">Pipeline Trace</div>
            <div class="funnel-steps">${stepRows}</div>
            <div class="funnel-total">
                <span>Total (${trace.steps.length} steps)</span>
                <span>${trace.total_duration_ms}ms</span>
            </div>
        </div>`;
    }

    el.innerHTML = `
        <div class="msg-avatar">🛰️</div>
        <div class="msg-bubble">${contentHtml}</div>`;

    messagesContainer.appendChild(el);
    scrollToBottom();

    // Animate confidence bar after paint
    requestAnimationFrame(() => {
        const fill = el.querySelector('.confidence-fill');
        if (fill) { fill.style.width = fill.style.width; }
    });
}

function appendErrorMessage(msg) {
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
        <div class="msg-avatar">🛰️</div>
        <div class="msg-bubble">
            <div class="abstention-banner">
                ⚠️ <div><strong>Error</strong><br>${escapeHtml(msg)}<br><small>Ensure the backend is running on port 8001.</small></div>
            </div>
        </div>`;
    messagesContainer.appendChild(el);
    scrollToBottom();
}

// ── Suggestion chips ──────────────────────────────────────────────────
function submitSuggestion(btn) {
    queryInput.value = btn.textContent;
    handleSubmit();
}

// ── Utilities ─────────────────────────────────────────────────────────
function scrollToBottom() {
    chatArea.scrollTop = chatArea.scrollHeight;
}

function escapeHtml(str) {
    const d = document.createElement('div');
    d.appendChild(document.createTextNode(str));
    return d.innerHTML;
}
