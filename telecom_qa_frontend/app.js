document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchInput');
    
    const loadingState = document.getElementById('loadingState');
    const errorState = document.getElementById('errorState');
    const resultsArea = document.getElementById('resultsArea');
    
    const answerContent = document.getElementById('answerContent');
    const citationsList = document.getElementById('citationsList');
    
    const errorTitle = document.getElementById('errorTitle');
    const errorMessage = document.getElementById('errorMessage');

    // Guardrail DOM elements
    const grAbstain = document.getElementById('gr-abstain');
    const grAbstainDesc = document.getElementById('gr-abstain-desc');
    const grWeb = document.getElementById('gr-web');
    const grWebDesc = document.getElementById('gr-web-desc');
    const grVersion = document.getElementById('gr-version');
    const grVersionDesc = document.getElementById('gr-version-desc');
    const grCitations = document.getElementById('gr-citations');
    const grCitationsDesc = document.getElementById('gr-citations-desc');

    const API_URL = 'http://127.0.0.1:8001/query';

    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = searchInput.value.trim();
        if (!query) return;

        // Reset UI state
        errorState.classList.add('hidden');
        resultsArea.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });

            if (!response.ok) {
                throw new Error(`HTTP Error: ${response.status} - ${response.statusText}`);
            }

            const data = await response.json();
            renderResults(data);
            
        } catch (error) {
            console.error("API Error:", error);
            showError("Connection Failed", "Ensure the FastAPI backend is running on port 8000.");
        } finally {
            loadingState.classList.add('hidden');
        }
    });

    function renderResults(data) {
        // 1. Format Answer and Citations
        let displayAnswer = data.answer;
        citationsList.innerHTML = '';
        
        if (data.citations && data.citations.length > 0) {
            data.citations.forEach((cit, index) => {
                const citeNum = index + 1;
                
                // Replace the ugly UUID wherever it appears with a clean number
                const regex = new RegExp(cit, 'g');
                displayAnswer = displayAnswer.replace(regex, citeNum);
                
                // Render the citation card with the matching number
                const card = document.createElement('div');
                card.className = 'citation-card';
                card.innerHTML = `
                    <span class="citation-id">[${citeNum}] Source: ${cit}</span>
                `;
                citationsList.appendChild(card);
            });
        } else {
            citationsList.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.9rem;">No citations provided for this response.</p>';
        }

        // Render Markdown Answer
        if (data.is_abstained) {
            answerContent.innerHTML = `<div style="color: var(--status-error); font-weight: 500;">
                ⚠️ System Abstained: ${data.abstention_reason || "Insufficient information to answer."}
            </div>`;
        } else if (data.guardrails.answered_from_general_knowledge) {
            // Warn the user inside the answer box too
            answerContent.innerHTML = `<div style="margin-bottom: 1rem; color: var(--status-warning); font-weight: 500;">
                ⚠️ Note: The local database and web search were insufficient. This answer was generated using the AI's general knowledge.
            </div>` + marked.parse(displayAnswer);
        } else {
            // Use marked.js to render the cleaned markdown
            answerContent.innerHTML = marked.parse(displayAnswer);
        }

        // 3. Update Guardrails
        updateGuardrail(
            grAbstain, grAbstainDesc, 
            !data.guardrails.answered_from_general_knowledge, 
            data.guardrails.answered_from_general_knowledge ? "Answer generated from AI memory." : "Answer grounded in documents."
        );

        updateGuardrail(
            grWeb, grWebDesc, 
            !data.guardrails.web_fallback_used, 
            data.guardrails.answered_from_general_knowledge 
                ? "N/A (Web search bypassed for AI memory)" 
                : (data.guardrails.web_fallback_used ? "Answer augmented via Web Search." : "Pure 3GPP data used.")
        );

        updateGuardrail(
            grVersion, grVersionDesc, 
            !data.guardrails.version_conflict_detected, 
            data.guardrails.version_conflict_detected ? "Warning: Conflicting standard versions found." : "No cross-release conflicts detected."
        );

        const hasUnresolvedCitations = data.guardrails.unresolved_citations && data.guardrails.unresolved_citations.length > 0;
        updateGuardrail(
            grCitations, grCitationsDesc, 
            !hasUnresolvedCitations, 
            data.guardrails.answered_from_general_knowledge 
                ? "N/A (No citations required for AI memory)"
                : (hasUnresolvedCitations ? "Warning: LLM hallucinated citations." : "All claims grounded in DB.")
        );

        // Show the results area
        resultsArea.classList.remove('hidden');
    }

    function updateGuardrail(element, descElement, isSuccess, text) {
        const icon = element.querySelector('.gr-status-icon');
        const title = element.querySelector('.gr-title');
        
        if (isSuccess) {
            icon.textContent = '✅';
            title.style.color = 'var(--status-success)';
        } else {
            icon.textContent = '⚠️';
            title.style.color = 'var(--status-warning)';
        }
        
        descElement.textContent = text;
    }

    function showError(title, message) {
        errorTitle.textContent = title;
        errorMessage.textContent = message;
        errorState.classList.remove('hidden');
    }
});
