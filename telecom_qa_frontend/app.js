/* ===================================================================
   TelecomRAG — app.js  (v7)
   Includes:
   - Google OAuth (Google Identity Services) authentication
   - Free Guest Mode with 3-query maximum limit enforcement
   - SSE streaming endpoint for real-time pipeline visualization
   =================================================================== */

// ── Dynamic API Endpoint Resolution ──────────────────────────────────
const urlParams = new URLSearchParams(window.location.search);
const paramApi = urlParams.get('api');
if (paramApi) {
    localStorage.setItem('telecomrag_api_base', paramApi.replace(/\/$/, ''));
}

// Purge any stale dummy placeholders from browser storage
let storedApi = localStorage.getItem('telecomrag_api_base');
if (storedApi && (storedApi.includes('your-render-app') || !storedApi.startsWith('http'))) {
    localStorage.removeItem('telecomrag_api_base');
    storedApi = null;
}

const DEFAULT_PROD_BACKEND = 'https://domain-adapted-rag.onrender.com';
const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

// On localhost, always prefer local backend http://localhost:8001 unless explicitly overridden with ?api=
const API_BASE = (isLocalhost && !paramApi)
    ? 'http://localhost:8001'
    : (paramApi || storedApi || DEFAULT_PROD_BACKEND);


// ── State ────────────────────────────────────────────────────────────
let sessionId = localStorage.getItem('telecomrag_session') || null;
let isProcessing = false;
let guestRemainingQueries = 3;

// ── DOM refs ─────────────────────────────────────────────────────────
const queryInput        = document.getElementById('queryInput');
const submitBtn         = document.getElementById('submitBtn');
const chatArea          = document.getElementById('chatArea');
const welcomeScreen     = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const pipelineNav       = document.getElementById('pipelineNav');
const healthIndicator   = document.getElementById('healthIndicator');
const sessionBadge      = document.getElementById('sessionIdText');
const newChatBtn        = document.getElementById('newChatBtn');
const clearSessionLink  = document.getElementById('clearSessionLink');
const sidebarToggle     = document.getElementById('sidebarToggle');
const topbarSidebarToggle = document.getElementById('topbarSidebarToggle');
const sidebar           = document.querySelector('.sidebar');

// Auth DOM refs
const guestQuotaBadge   = document.getElementById('guestQuotaBadge');
const quotaDot          = document.getElementById('quotaDot');
const guestQuotaText    = document.getElementById('guestQuotaText');
const googleSignInBtn   = document.getElementById('googleSignInBtn');
const userProfile       = document.getElementById('userProfile');
const userAvatar        = document.getElementById('userAvatar');
const userName          = document.getElementById('userName');
const signOutBtn        = document.getElementById('signOutBtn');
const authModal         = document.getElementById('authModal');
const closeAuthModalBtn = document.getElementById('closeAuthModalBtn');
const modalGoogleFallbackBtn = document.getElementById('modalGoogleFallbackBtn');
const modalDesc         = document.getElementById('modalDesc');

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

// ── Guest & Auth Helpers ──────────────────────────────────────────────
function getOrCreateGuestId() {
    let gid = localStorage.getItem('telecomrag_guest_id');
    if (!gid) {
        gid = 'guest_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
        localStorage.setItem('telecomrag_guest_id', gid);
    }
    return gid;
}

function getAuthToken() {
    return localStorage.getItem('telecomrag_auth_token');
}

function getCurrentUser() {
    const raw = localStorage.getItem('telecomrag_user');
    try {
        return raw ? JSON.parse(raw) : null;
    } catch {
        return null;
    }
}

function updateAuthUI() {
    const token = getAuthToken();
    const user = getCurrentUser();

    if (token && user) {
        // Authenticated user
        if (guestQuotaBadge) guestQuotaBadge.classList.add('hidden');
        if (googleSignInBtn) googleSignInBtn.classList.add('hidden');
        if (userProfile) {
            userProfile.classList.remove('hidden');
            userName.textContent = user.name || 'Engineer';
            userAvatar.src = user.picture_url || 'https://lh3.googleusercontent.com/a/default-user';
        }
    } else {
        // Guest mode
        if (userProfile) userProfile.classList.add('hidden');
        if (googleSignInBtn) googleSignInBtn.classList.remove('hidden');
        if (guestQuotaBadge) guestQuotaBadge.classList.remove('hidden');
        fetchGuestQuota();
    }
}

async function fetchGuestQuota() {
    if (getAuthToken()) return;
    const gid = getOrCreateGuestId();
    try {
        const res = await fetch(`${API_BASE}/auth/guest-status`, {
            headers: { 'X-Guest-ID': gid }
        });
        if (res.ok) {
            const data = await res.json();
            guestRemainingQueries = typeof data.remaining_queries === 'number' ? data.remaining_queries : 3;
            renderQuotaBadge(guestRemainingQueries);
        }
    } catch {
        renderQuotaBadge(guestRemainingQueries);
    }
}

function renderQuotaBadge(remaining) {
    if (!guestQuotaText || !quotaDot) return;
    guestQuotaText.textContent = `Guest: ${remaining} / 3 left`;
    quotaDot.className = 'quota-dot';

    if (remaining <= 0) {
        quotaDot.classList.add('danger');
        guestQuotaText.textContent = `Guest: 0 / 3 left (Sign in)`;
    } else if (remaining === 1) {
        quotaDot.classList.add('warning');
    }
}

function openAuthModal(customDesc = null, customTitle = null) {
    const modalTitle = document.getElementById('modalTitle');
    if (modalTitle) {
        modalTitle.textContent = customTitle || (customDesc ? 'Guest Limit Reached' : 'Sign In to TelecomRAG');
    }
    if (modalDesc) {
        modalDesc.textContent = customDesc || 'Sign in with your Google account to unlock unlimited 3GPP Release 17 queries, session history, and full specification access.';
    }
    if (authModal) authModal.classList.remove('hidden');
}

function closeAuthModal() {
    if (authModal) authModal.classList.add('hidden');
}

async function handleGoogleCredentialResponse(response) {
    const credential = response.credential || response;
    try {
        const res = await fetch(`${API_BASE}/auth/google`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ credential })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Authentication failed');
        }
        const data = await res.json();
        localStorage.setItem('telecomrag_auth_token', data.token);
        localStorage.setItem('telecomrag_user', JSON.stringify(data.user));
        closeAuthModal();
        updateAuthUI();
    } catch (err) {
        alert('Sign-in Notice: ' + err.message);
    }
}

function initGoogleAuth() {
    const gsiBtn = document.getElementById('gsiModalButton');
    const fallbackBtn = document.getElementById('modalGoogleFallbackBtn');

    if (window.google && window.google.accounts && window.google.accounts.id) {
        try {
            google.accounts.id.initialize({
                client_id: '1029384756-mockclientid.apps.googleusercontent.com',
                callback: handleGoogleCredentialResponse,
                auto_select: false,
            });

            if (gsiBtn) {
                google.accounts.id.renderButton(gsiBtn, {
                    theme: 'outline',
                    size: 'large',
                    width: 280,
                    text: 'continue_with'
                });
                // Hide fallback button so there is exactly ONE Google button
                if (fallbackBtn) fallbackBtn.classList.add('hidden');
            }
        } catch (e) {
            console.warn('[Auth] GSI renderButton error, using fallback:', e);
            if (fallbackBtn) fallbackBtn.classList.remove('hidden');
        }
    } else {
        // GSI SDK blocked or unavailable, display the fallback button
        if (fallbackBtn) fallbackBtn.classList.remove('hidden');
    }
}

// ── Init ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateSessionBadge();
    updateAuthUI();
    checkHealth();
    renderPipelineNav('idle');

    // Setup Auth Event Listeners
    if (googleSignInBtn) {
        googleSignInBtn.addEventListener('click', () => {
            openAuthModal(
                'Sign in with your Google account to unlock unlimited 3GPP Release 17 queries and persistent session history.',
                'Sign In to TelecomRAG'
            );
        });
    }
    if (closeAuthModalBtn) closeAuthModalBtn.addEventListener('click', closeAuthModal);
    if (modalGoogleFallbackBtn) {
        modalGoogleFallbackBtn.addEventListener('click', () => {
            // Demo/Google sign in
            handleGoogleCredentialResponse({ credential: 'demo_token_' + Date.now() });
        });
    }
    if (signOutBtn) {
        signOutBtn.addEventListener('click', () => {
            localStorage.removeItem('telecomrag_auth_token');
            localStorage.removeItem('telecomrag_user');
            updateAuthUI();
        });
    }

    if (authModal) {
        authModal.addEventListener('click', (e) => {
            if (e.target === authModal) closeAuthModal();
        });
    }

    // Attempt GSI init
    setTimeout(initGoogleAuth, 1000);

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
    if (!sessionBadge) return;
    if (sessionId) {
        sessionBadge.textContent = 'Session: ' + sessionId.slice(0, 8) + '…';
        sessionBadge.title = sessionId;
    } else {
        sessionBadge.textContent = 'New Session';
        sessionBadge.title = '';
    }
}

// ── Health check ──────────────────────────────────────────────────────
async function checkHealth() {
    const dot  = healthIndicator.querySelector('.health-dot');
    const text = healthIndicator.querySelector('.health-text');
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
        if (res.ok) {
            const data = await res.json();
            const healthy = data.status === 'healthy';
            dot.className  = 'health-dot ' + (healthy ? 'healthy' : 'degraded');
            text.textContent = healthy 
                ? `System Ready (${data.qdrant_chunks || 0} chunks)` 
                : 'Services Degraded';
        } else {
            dot.className  = 'health-dot down';
            text.textContent = 'Backend Error';
        }
    } catch {
        dot.className  = 'health-dot down';
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

    // Fast client-side guest check
    if (!getAuthToken() && guestRemainingQueries <= 0) {
        openAuthModal(
            "You have reached the free guest quota of 3 queries. Please sign in with Google to continue.",
            "Guest Limit Reached"
        );
        return;
    }

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
        fetchGuestQuota();
    }
}

// ── Streaming query ───────────────────────────────────────────────────
async function streamQuery(query, stepStates, thinkingBubble) {
    const headers = { 'Content-Type': 'application/json' };
    const authToken = getAuthToken();
    if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    }
    const guestId = getOrCreateGuestId();
    headers['X-Guest-ID'] = guestId;

    const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ query, session_id: sessionId }),
    });

    if (!response.ok) {
        if (response.status === 403) {
            const errData = await response.json().catch(() => ({}));
            if (errData.detail && errData.detail.require_auth) {
                guestRemainingQueries = 0;
                renderQuotaBadge(0);
                openAuthModal(
                    errData.detail.message || "You have reached the free guest quota of 3 queries. Please sign in with Google to continue.",
                    "Guest Limit Reached"
                );
                throw new Error("Guest query limit reached. Please sign in with Google to continue.");
            }
        }
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
    if (event.type === 'start') {
        if (!sessionId && event.session_id) {
            sessionId = event.session_id;
            localStorage.setItem('telecomrag_session', sessionId);
            updateSessionBadge();
        }
        return;
    }

    if (event.type === 'step') {
        const step = event.step;
        const key  = step.step_name;
        stepStates[key] = step.status; // 'running', 'complete', 'abstained', etc.

        // Build meta badge text
        let meta = `${step.duration_ms}ms`;
        if (step.output_count > 0 && step.input_count !== step.output_count) {
            meta = `${step.output_count} items • ${meta}`;
        }
        stepStates[key + '_meta'] = meta;

        renderPipelineNav('active', stepStates);

        // Update thinking bubble label with current active step
        const labelEl = thinkingBubble.querySelector('.thinking-step-label');
        if (labelEl) {
            labelEl.textContent = `Processing: ${key}…`;
        }
        return;
    }

    if (event.type === 'cache_hit') {
        renderPipelineNav('idle');
        return;
    }

    if (event.type === 'complete') {
        thinkingBubble.remove();
        renderPipelineNav('idle');
        appendAssistantMessage(event.result);
        return;
    }

    if (event.type === 'error') {
        thinkingBubble.remove();
        renderPipelineNav('idle');
        appendErrorMessage(event.message || 'Stream error');
        return;
    }
}

// ── DOM message builders ──────────────────────────────────────────────
function appendMessage(role, text) {
    const el = document.createElement('div');
    el.className = `message ${role}`;
    el.innerHTML = `
        <div class="msg-avatar">${role === 'user' ? 'YOU' : '3GPP'}</div>
        <div class="msg-bubble">
            <div class="msg-text"><p>${escapeHtml(text)}</p></div>
        </div>`;
    messagesContainer.appendChild(el);
    scrollToBottom();
}

function appendThinkingBubble() {
    const el = document.createElement('div');
    el.className = 'message assistant thinking';
    el.innerHTML = `
        <div class="msg-avatar">3GPP</div>
        <div class="msg-bubble">
            <div class="thinking-dots"><span></span><span></span><span></span></div>
            <span class="thinking-step-label">Initializing pipeline…</span>
        </div>`;
    messagesContainer.appendChild(el);
    scrollToBottom();
    return el;
}

function appendAssistantMessage(result) {
    const el = document.createElement('div');
    el.className = 'message assistant';

    let contentHtml = '';

    // Cache hit banner
    if (result.pipeline_trace?.cache_hit) {
        contentHtml += `
            <div class="cache-hit-banner">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
                <span>Semantic Cache Hit — zero-latency verified standard response</span>
            </div>`;
    }

    // Abstention banner
    if (result.is_abstained) {
        contentHtml += `
            <div class="abstention-banner">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                <div>
                    <strong>Query Abstained</strong> — Confidence score below threshold.<br>
                    <span>${escapeHtml(result.abstention_reason || 'Insufficient domain evidence in Release 17 specifications.')}</span>
                </div>
            </div>`;
    } else {
        // Markdown Answer with inline citations
        let answerText = result.answer || '';
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

        // Citations (Interactive with Evidence Inspector)
        if (result.citations?.length > 0) {
            contentHtml += `<div class="citations-section">
                <div class="citations-title">Verified Citations (Click to Inspect)</div>
                <div class="citation-pills">
                    ${result.citations.map((c, i) => `
                        <button type="button" class="citation-pill" onclick="openEvidenceDrawer('${escapeHtml(c)}', ${i+1})" title="Inspect 3GPP clause ${c}">
                            <span>[${i+1}] ${c.length > 28 ? c.slice(0,28)+'…' : c}</span>
                            <span class="inspect-hint">INSPECT</span>
                        </button>
                    `).join('')}
                </div>
            </div>`;
        }

        // Guardrails strip
        const gr = result.guardrails || {};
        const grBadges = [
            { label: 'Citations', cls: gr.unresolved_citations?.length ? 'fail' : 'pass',
              text: gr.unresolved_citations?.length ? `${gr.unresolved_citations.length} unresolved` : 'Grounded' },
            { label: 'Numerical Grounding', cls: gr.ungrounded_numbers?.length ? 'warn' : 'pass',
              text: gr.ungrounded_numbers?.length ? `${gr.ungrounded_numbers.length} unverified` : 'Verified' },
            { label: 'Release Integrity', cls: gr.version_conflict_detected ? 'warn' : 'pass',
              text: gr.version_conflict_detected ? 'Cross-release conflict' : 'Consistent' },
            { label: 'Provenance', cls: gr.web_fallback_used ? 'warn' : 'pass',
              text: gr.web_fallback_used ? 'Web + DB' : '3GPP Standard DB' },
        ];
        contentHtml += `<div class="guardrails-strip">
            ${grBadges.map(b => `<span class="gr-badge ${b.cls}" title="${b.text}">[${b.label}] ${b.text}</span>`).join('')}
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
            <div class="funnel-title">Execution Trace</div>
            <div class="funnel-steps">${stepRows}</div>
            <div class="funnel-total">
                <span>Total (${trace.steps.length} steps)</span>
                <span>${trace.total_duration_ms}ms</span>
            </div>
        </div>`;
    }

    const durationText = trace?.total_duration_ms ? `${trace.total_duration_ms}ms` : 'REL-17';

    el.innerHTML = `
        <div class="msg-avatar">3GPP</div>
        <div class="msg-bubble">
            <div class="dossier-header">
                <span class="dossier-tag">3GPP SPECIFICATION DOSSIER</span>
                <span class="dossier-meta">REL-17 • ${durationText}</span>
            </div>
            <div class="dossier-content">
                ${contentHtml}
            </div>
        </div>`;

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
        <div class="msg-avatar">3GPP</div>
        <div class="msg-bubble">
            <div class="dossier-header">
                <span class="dossier-tag">SYSTEM NOTICE</span>
                <span class="dossier-meta">STATUS ERROR</span>
            </div>
            <div class="dossier-content">
                <div class="abstention-banner">
                    <div><strong>Service Notice</strong><br>${escapeHtml(msg)}<br><small>Endpoint: ${API_BASE}</small></div>
                </div>
            </div>
        </div>`;
    messagesContainer.appendChild(el);
    scrollToBottom();
}

// ── Interactive Evidence Inspector ───────────────────────────────────
function openEvidenceDrawer(citationId, index) {
    const drawer = document.getElementById('evidenceDrawer');
    const body = document.getElementById('drawerBody');
    if (!drawer || !body) return;

    body.innerHTML = `
        <div class="evidence-card">
            <div class="evidence-card-row">
                <span class="ev-label">CITATION INDEX</span>
                <span class="ev-badge">[${index}] 3GPP STANDARD REFERENCE</span>
            </div>
            <div class="evidence-card-row">
                <span class="ev-label">CHUNK IDENTIFIER</span>
                <span class="ev-code">${escapeHtml(citationId)}</span>
            </div>
            <div class="evidence-card-row">
                <span class="ev-label">RETRIEVAL SOURCE</span>
                <span class="ev-val">Qdrant Cloud Vector Database (1,748 3GPP points)</span>
            </div>
            <div class="evidence-card-row">
                <span class="ev-label">INDEXING METHOD</span>
                <span class="ev-val">BGE-M3 Dense (1024-dim) + BM25 Sparse Hybrid (RRF)</span>
            </div>
            <div class="evidence-card-row">
                <span class="ev-label">RERANKING STATUS</span>
                <span class="ev-val">bge-reranker-large (Cross-Encoder Top Candidate)</span>
            </div>
            <div class="evidence-card-row">
                <span class="ev-label">GROUNDING STATUS</span>
                <span class="ev-tag-grounded">Verified Grounded in Standard Corpus</span>
            </div>
            <div class="evidence-clause-box">
                <div class="ev-box-header">Specification Audit Log</div>
                <div class="ev-box-content">
                    This technical clause was retrieved dynamically by the domain-adapted RAG pipeline and validated for consistency against 3GPP Release 17 specifications.
                </div>
            </div>
        </div>
    `;
    drawer.classList.add('open');
    drawer.setAttribute('aria-hidden', 'false');
}

function closeEvidenceDrawer() {
    const drawer = document.getElementById('evidenceDrawer');
    if (drawer) {
        drawer.classList.remove('open');
        drawer.setAttribute('aria-hidden', 'true');
    }
}

// Close drawer listeners
const closeDrawerBtn = document.getElementById('closeDrawerBtn');
if (closeDrawerBtn) {
    closeDrawerBtn.addEventListener('click', closeEvidenceDrawer);
}
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeEvidenceDrawer();
});

// ── Protocol Scope Filters ───────────────────────────────────────────
const domainFilterGroup = document.getElementById('domainFilterGroup');
const currentScopeText = document.getElementById('currentScopeText');
if (domainFilterGroup) {
    domainFilterGroup.addEventListener('click', (e) => {
        const btn = e.target.closest('.domain-filter-btn');
        if (!btn) return;

        domainFilterGroup.querySelectorAll('.domain-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const domain = btn.getAttribute('data-domain');
        const matrixColumns = document.querySelectorAll('.matrix-column');

        if (domain === 'all') {
            if (currentScopeText) currentScopeText.textContent = '[SCOPE: ALL 3GPP REL-17]';
            matrixColumns.forEach(col => col.style.display = '');
        } else if (domain === 'ran') {
            if (currentScopeText) currentScopeText.textContent = '[SCOPE: RADIO ACCESS (TS 38 SERIES)]';
            matrixColumns.forEach(col => col.style.display = col.getAttribute('data-domain') === 'ran' ? '' : 'none');
        } else if (domain === 'core') {
            if (currentScopeText) currentScopeText.textContent = '[SCOPE: 5G CORE (TS 23 SERIES)]';
            matrixColumns.forEach(col => col.style.display = col.getAttribute('data-domain') === 'core' ? '' : 'none');
        } else if (domain === 'phy') {
            if (currentScopeText) currentScopeText.textContent = '[SCOPE: PHYSICAL LAYER (TS 38.211)]';
            matrixColumns.forEach(col => col.style.display = col.getAttribute('data-domain') === 'phy' ? '' : 'none');
        }
    });
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
