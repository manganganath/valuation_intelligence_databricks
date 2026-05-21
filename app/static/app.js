/**
 * Valuation Intelligence Agent — Frontend Logic
 * Handles SSE streaming, response rendering, trace panel, and session management.
 */

(function () {
    'use strict';

    let currentSessionId = window.SESSION_ID || '';
    let isLoading = false;

    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const chatArea = document.getElementById('chat-area');
    const messagesDiv = document.getElementById('messages');
    const emptyState = document.getElementById('empty-state');
    const tracePanel = document.getElementById('trace-panel');
    const traceSteps = document.getElementById('trace-steps');
    const newSessionBtn = document.getElementById('new-session-btn');

    // --- Query Submission (SSE Streaming) ---

    async function submitQuery() {
        const message = queryInput.value.trim();
        if (!message || isLoading) return;

        isLoading = true;
        sendBtn.disabled = true;
        queryInput.value = '';
        queryInput.style.height = 'auto';
        emptyState.style.display = 'none';

        appendMessage('user', message);

        // Show loading indicator
        const loadingEl = document.createElement('div');
        loadingEl.className = 'loading';
        loadingEl.innerHTML = '<div class="spinner"></div><span class="loading-text">Thinking...</span>';
        messagesDiv.appendChild(loadingEl);
        scrollToBottom();

        // Prepare the assistant message container (will be filled by streaming tokens)
        const assistantEl = document.createElement('div');
        assistantEl.className = 'message message-assistant';
        assistantEl.style.display = 'none';
        assistantEl.innerHTML = '<div class="label">Agent</div><div class="content"></div><div class="trace-links"></div>';
        messagesDiv.appendChild(assistantEl);
        const contentDiv = assistantEl.querySelector('.content');
        const traceLinksDiv = assistantEl.querySelector('.trace-links');

        // Reset trace panel
        tracePanel.classList.remove('hidden');
        traceSteps.innerHTML = '';

        let fullContent = '';
        let responseId = '';

        try {
            const resp = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message, session_id: currentSessionId }),
            });

            if (!resp.ok) {
                loadingEl.remove();
                const err = await resp.json();
                appendMessage('assistant', 'Error: ' + (err.error || 'Unknown error'));
                assistantEl.remove();
                return;
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const events = buffer.split('\n\n');
                buffer = events.pop(); // Keep incomplete event in buffer

                for (const eventBlock of events) {
                    if (!eventBlock.trim()) continue;

                    const lines = eventBlock.split('\n');
                    let eventType = '';
                    let eventData = '';

                    for (const line of lines) {
                        if (line.startsWith('event: ')) eventType = line.slice(7);
                        else if (line.startsWith('data: ')) eventData = line.slice(6);
                    }

                    if (!eventType || !eventData) continue;

                    let parsed;
                    try { parsed = JSON.parse(eventData); } catch (e) { continue; }

                    if (eventType === 'trace') {
                        addTraceStep(parsed);
                        // Update loading text
                        const loadingText = loadingEl.querySelector('.loading-text');
                        if (loadingText) {
                            if (parsed.type === 'tool_call') {
                                loadingText.textContent = 'Calling ' + parsed.label + '...';
                            } else if (parsed.type === 'tool_result') {
                                loadingText.textContent = 'Processing results...';
                            }
                        }
                    } else if (eventType === 'token') {
                        // First token — hide loading, show assistant message
                        if (!assistantEl.style.display || assistantEl.style.display === 'none') {
                            loadingEl.remove();
                            assistantEl.style.display = '';
                        }
                        fullContent += parsed.text;
                        contentDiv.innerHTML = renderMarkdown(fullContent);
                        scrollToBottom();
                    } else if (eventType === 'done') {
                        loadingEl.remove();
                        fullContent = parsed.content || fullContent;
                        responseId = parsed.response_id || '';
                        if (fullContent && assistantEl.style.display === 'none') {
                            assistantEl.style.display = '';
                        }
                        contentDiv.innerHTML = renderMarkdown(fullContent);
                        // Trace links removed — trace panel is already visible
                        // Add final "message" trace step
                        addTraceStep({ type: 'message', label: 'Agent Response', data: fullContent.substring(0, 200) + (fullContent.length > 200 ? '...' : '') });
                        scrollToBottom();
                    } else if (eventType === 'session') {
                        currentSessionId = parsed.session_id || currentSessionId;
                        refreshSessions();
                    } else if (eventType === 'error') {
                        loadingEl.remove();
                        assistantEl.style.display = '';
                        contentDiv.textContent = 'Error: ' + (parsed.message || 'Unknown error');
                    }
                }
            }
        } catch (err) {
            loadingEl.remove();
            assistantEl.style.display = '';
            contentDiv.textContent = 'Connection error: ' + err.message;
        } finally {
            isLoading = false;
            sendBtn.disabled = false;
            queryInput.focus();
        }
    }

    // --- Message Rendering ---

    function appendMessage(role, content, responseId) {
        const el = document.createElement('div');
        el.className = 'message message-' + role;

        if (role === 'user') {
            el.innerHTML = '<div class="label">You</div><div>' + escapeHtml(content) + '</div>';
        } else {
            let html = '<div class="label">Agent</div>';
            html += '<div class="content">' + renderMarkdown(content) + '</div>';
            if (responseId) {
                html += '<a class="trace-link" onclick="document.getElementById(\'trace-panel\').classList.remove(\'hidden\')">\u{1F50D} View Trace</a> ';
                html += '<a class="trace-link" onclick="document.getElementById(\'trace-panel\').classList.remove(\'hidden\')">\u{1F4C8} View Trace Details</a>';
            }
            el.innerHTML = html;
        }

        messagesDiv.appendChild(el);
        scrollToBottom();
    }

    function renderMarkdown(text) {
        if (!text) return '';
        if (typeof marked !== 'undefined') {
            return marked.parse(text);
        }
        // Fallback: return escaped text with line breaks
        return escapeHtml(text).replace(/\n/g, '<br>');
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // --- Trace Panel ---

    function addTraceStep(step) {
        const el = document.createElement('div');
        el.className = 'trace-step ' + step.type;

        const header = document.createElement('div');
        header.className = 'trace-step-header';
        header.innerHTML = '<span class="trace-step-icon"></span><span>' + escapeHtml(step.label) + '</span>';

        const data = document.createElement('div');
        data.className = 'trace-step-data';
        data.textContent = typeof step.data === 'string' ? step.data : JSON.stringify(step.data, null, 2);

        header.addEventListener('click', function () { el.classList.toggle('expanded'); });

        el.appendChild(header);
        el.appendChild(data);
        traceSteps.appendChild(el);
        traceSteps.scrollTop = traceSteps.scrollHeight;
    }

    function showTrace(steps) {
        tracePanel.classList.remove('hidden');
        traceSteps.innerHTML = '';
        steps.forEach(addTraceStep);
    }

    // --- Session Management ---

    async function refreshSessions() {
        try {
            const resp = await fetch('/sessions');
            const sessions = await resp.json();
            const list = document.getElementById('session-list');
            list.innerHTML = '';
            sessions.forEach(function (s) {
                const item = document.createElement('div');
                item.className = 'session-item' + (s.session_id === currentSessionId ? ' active' : '');
                item.dataset.sessionId = s.session_id;

                var timeSpan = document.createElement('span');
                timeSpan.className = 'session-time';
                timeSpan.textContent = s.created_at.substring(0, 16);

                var previewSpan = document.createElement('span');
                previewSpan.className = 'session-preview';
                previewSpan.textContent = s.preview ? s.preview.substring(0, 60) : 'New session';

                var deleteBtn = document.createElement('button');
                deleteBtn.className = 'session-delete';
                deleteBtn.textContent = '\u00d7';
                deleteBtn.title = 'Delete session';
                deleteBtn.addEventListener('click', function (e) {
                    e.stopPropagation();
                    deleteSessionById(s.session_id);
                });

                item.appendChild(timeSpan);
                item.appendChild(previewSpan);
                item.appendChild(deleteBtn);
                item.addEventListener('click', function () { loadSession(s.session_id); });
                list.appendChild(item);
            });
        } catch (e) {
            console.error('Failed to refresh sessions:', e);
        }
    }

    async function deleteSessionById(sessionId) {
        try {
            await fetch('/sessions/' + sessionId, { method: 'DELETE' });
            if (sessionId === currentSessionId) {
                currentSessionId = '';
                messagesDiv.innerHTML = '';
                emptyState.style.display = '';
            }
            refreshSessions();
        } catch (e) {
            console.error('Failed to delete session:', e);
        }
    }

    async function loadSession(sessionId) {
        currentSessionId = sessionId;
        messagesDiv.innerHTML = '';
        emptyState.style.display = 'none';

        document.querySelectorAll('.session-item').forEach(function (el) {
            el.classList.toggle('active', el.dataset.sessionId === sessionId);
        });

        try {
            const resp = await fetch('/sessions/' + sessionId);
            const history = await resp.json();
            history.forEach(function (msg) { appendMessage(msg.role, msg.content); });
        } catch (e) {
            console.error('Failed to load session:', e);
        }
    }

    // --- Event Listeners ---

    sendBtn.addEventListener('click', submitQuery);

    queryInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            submitQuery();
        }
    });

    queryInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });

    document.querySelectorAll('.chip').forEach(function (chip) {
        chip.addEventListener('click', function () {
            queryInput.value = this.dataset.query;
            queryInput.style.height = 'auto';
            queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
            queryInput.focus();
        });
    });

    document.getElementById('toggle-trace-btn').addEventListener('click', function () { tracePanel.classList.toggle('hidden'); });
    newSessionBtn.addEventListener('click', function () {
        currentSessionId = '';
        messagesDiv.innerHTML = '';
        emptyState.style.display = '';
        tracePanel.classList.add('hidden');
        traceSteps.innerHTML = '';
        document.querySelectorAll('.session-item').forEach(function (el) { el.classList.remove('active'); });
        queryInput.focus();
    });

    // Load sessions on page init, auto-load most recent session with messages
    refreshSessions().then(function () {
        // Find the first session with a preview (has messages)
        var items = document.querySelectorAll('.session-item');
        for (var i = 0; i < items.length; i++) {
            var preview = items[i].querySelector('.session-preview');
            if (preview && preview.textContent !== 'New session') {
                loadSession(items[i].dataset.sessionId);
                return;
            }
        }
    });

})();
