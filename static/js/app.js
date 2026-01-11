/**
 * HatCat FTW - AI Manipulation Detection Dashboard
 * Main JavaScript Application
 */

// ============================================================================
// Global State
// ============================================================================

const AppState = {
    modelLoaded: false,
    lensLoaded: false,
    suites: [],
    selectedSuiteId: null,
    currentSuiteMeta: null,
    pendingSuiteId: null,
    episodes: [],
    selectedEpisodes: new Set(),
    currentResults: null,
    isRunning: false,
    currentRunId: null,
    // Episode history for navigation during runs
    episodeHistory: [],      // Array of {episode_id, condition, tokens, metadata, turnsHtml, ...}
    currentHistoryIndex: -1, // -1 = showing live, >= 0 = showing history
    liveTokens: [],          // Current live generation tokens
    liveMetadata: [],        // Current live generation metadata
    // Track current live panel contents for history restoration
    liveTurnsHtml: '',
    liveAlertsHtml: '',
    liveSteeringHtml: '',
};

// ============================================================================
// Run State Persistence
// ============================================================================

async function checkAndRestoreRunState() {
    /**
     * Check if there's an active or completed run and restore UI state.
     * Called on page load to handle browser refresh during long runs.
     */
    try {
        const response = await fetch('/api/evaluation/status');
        const state = await response.json();

        if (!state.run_id) return; // No run in progress

        console.log('Restoring run state:', state.status, state.run_id);

        // Restore AppState
        AppState.currentRunId = state.run_id;
        if (state.suite_id) {
            AppState.pendingSuiteId = state.suite_id;
            AppState.selectedSuiteId = state.suite_id;
        }

        if (state.status === 'running') {
            // Run is still in progress - navigate to run page and show progress
            AppState.isRunning = true;
            navigateTo('run');

            // Update progress display
            const progress = state.progress || 0;
            document.getElementById('eval-progress').style.width = `${progress * 100}%`;
            document.getElementById('eval-status').textContent =
                `Running: ${state.current_episode || '?'} / ${state.current_condition || '?'} (${(progress * 100).toFixed(0)}%)`;

            // Restore tokens from buffer
            const container = document.getElementById('live-tokens');
            container.innerHTML = '';
            for (const tokenData of (state.tokens_buffer || [])) {
                appendToken(tokenData.token, tokenData.metadata);
            }

            // Show info about restored state
            logToRunLog(`Restored run ${state.run_id} - ${state.completed_episodes}/${state.total_episodes} episodes complete`, 'info');
            logToRunLog(`Current: ${state.current_episode} / ${state.current_condition}`, 'info');

            // Start polling for updates since we missed the SSE stream
            startStatusPolling();

        } else if (state.status === 'complete') {
            // Run completed while we were away
            AppState.isRunning = false;
            navigateTo('run');

            document.getElementById('eval-progress').style.width = '100%';
            document.getElementById('eval-status').textContent = 'Complete!';

            logToRunLog(`Run ${state.run_id} completed`, 'info');
            if (state.summary) {
                displaySummary(state.summary);
            }

            // Show view results button
            addViewResultsButton();

        } else if (state.status === 'error') {
            AppState.isRunning = false;
            navigateTo('run');
            logToRunLog(`Run ${state.run_id} failed: ${state.error_message}`, 'error');
        }

    } catch (error) {
        console.log('No run state to restore:', error.message);
    }
}

let statusPollingInterval = null;

function startStatusPolling() {
    /**
     * Poll for status updates when we've refreshed mid-run and lost the SSE connection.
     */
    if (statusPollingInterval) return;

    statusPollingInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/evaluation/status');
            const state = await response.json();

            if (state.status !== 'running') {
                // Run finished - stop polling
                clearInterval(statusPollingInterval);
                statusPollingInterval = null;

                AppState.isRunning = false;
                document.getElementById('eval-progress').style.width = '100%';

                if (state.status === 'complete') {
                    document.getElementById('eval-status').textContent = 'Complete!';
                    logToRunLog('Evaluation complete!', 'success');
                    if (state.summary) {
                        displaySummary(state.summary);
                    }
                    addViewResultsButton();
                } else if (state.status === 'aborted') {
                    document.getElementById('eval-status').textContent = 'Aborted';
                    logToRunLog('Evaluation aborted', 'warning');
                } else if (state.status === 'error') {
                    document.getElementById('eval-status').textContent = 'Error';
                    logToRunLog(`Error: ${state.error_message}`, 'error');
                }
                return;
            }

            // Update progress
            const progress = state.progress || 0;
            document.getElementById('eval-progress').style.width = `${progress * 100}%`;
            document.getElementById('eval-status').textContent =
                `Running: ${state.current_episode || '?'} / ${state.current_condition || '?'} (${(progress * 100).toFixed(0)}%)`;

        } catch (error) {
            console.error('Status polling error:', error);
        }
    }, 2000); // Poll every 2 seconds
}

function displaySummary(summary) {
    /**
     * Display summary statistics in the run log.
     */
    const byCondition = summary.by_condition || {};
    logToRunLog('─────────────────────────────', 'info');
    logToRunLog('RESULTS SUMMARY', 'info');
    for (const [cond, stats] of Object.entries(byCondition)) {
        const failRate = stats.fail_rate || stats.rate || 0;
        logToRunLog(`  ${cond}: ${failRate.toFixed(0)}% manipulation`, failRate > 50 ? 'error' : 'success');
    }
    logToRunLog('─────────────────────────────', 'info');
}

function addViewResultsButton() {
    /**
     * Add the prominent "View Results" button after run completes.
     */
    const existingBtn = document.querySelector('#run-log .view-results-btn');
    if (existingBtn) return; // Already exists

    const btn = document.createElement('button');
    btn.className = 'btn btn-primary btn-lg mt-4 view-results-btn';
    btn.style.cssText = 'background: #10b981; color: white; font-size: 1.25rem; padding: 1rem 2rem; border-radius: 0.5rem; font-weight: bold; animation: pulse 1.5s infinite;';
    btn.textContent = 'View Results →';
    btn.onclick = () => navigateTo('results');
    document.getElementById('run-log').appendChild(btn);
    btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Check for existing run state on page load
document.addEventListener('DOMContentLoaded', () => {
    loadSuites();
    // Delay slightly to let other init complete
    setTimeout(checkAndRestoreRunState, 500);
});

// ============================================================================
// Navigation
// ============================================================================

function navigateTo(page) {
    const pageOrder = ['setup', 'episodes', 'run', 'results'];
    const pageIndex = pageOrder.indexOf(page);

    // Update DaisyUI steps
    document.querySelectorAll('.steps .step').forEach((step, idx) => {
        step.classList.toggle('step-primary', idx <= pageIndex);
    });

    // Update pages
    document.querySelectorAll('.page').forEach(p => {
        p.classList.toggle('active', p.id === `page-${page}`);
    });

    // Page-specific initialization
    if (page === 'episodes') {
        loadEpisodes();
        updateTotalRuns();
    } else if (page === 'results') {
        loadRunsList();
    }
}

// ============================================================================
// Setup Page
// ============================================================================

async function checkStatus() {
    try {
        const response = await fetch('/api/setup/status');
        const data = await response.json();

        AppState.modelLoaded = data.model_ready;
        AppState.lensLoaded = data.lens_ready;

        // Update navbar info
        const gpuEl = document.getElementById('navbar-gpu-info');
        const modelEl = document.getElementById('navbar-model-info');
        if (gpuEl) gpuEl.textContent = data.gpu_available ? `GPU: ${data.gpu_name}` : 'GPU: Not available';
        if (modelEl) modelEl.textContent = data.model_ready ? 'Model: Loaded' : 'Model: Not loaded';

        // Update badges
        updateBadge('model-status-badge', data.model_ready);
        updateBadge('lens-status-badge', data.lens_ready);

        // Update buttons
        if (data.model_ready) {
            const btn = document.getElementById('btn-download-model');
            btn.textContent = 'Model Loaded';
            btn.disabled = true;
            btn.classList.add('btn-success');
            btn.classList.remove('btn-primary');
            document.getElementById('model-progress').value = 100;
            document.getElementById('model-progress-text').textContent = 'Model loaded';
        }

        if (data.lens_ready) {
            const btn = document.getElementById('btn-download-lens');
            btn.textContent = 'Lens Pack Loaded';
            btn.disabled = true;
            btn.classList.add('btn-success');
            btn.classList.remove('btn-primary');
            document.getElementById('lens-progress').value = 100;
            document.getElementById('lens-progress-text').textContent = 'Lens pack loaded';

            // Load stability metrics after lens is ready
            loadStabilityMetrics();
        }

        // Show setup complete banner
        if (data.model_ready && data.lens_ready) {
            document.getElementById('setup-complete-banner').classList.remove('hidden');
        }

    } catch (error) {
        console.error('Failed to check status:', error);
    }
}

function updateBadge(id, isSuccess) {
    const badge = document.getElementById(id);
    if (badge && isSuccess) {
        badge.textContent = 'Loaded';
        badge.classList.remove('badge-warning');
        badge.classList.add('badge-success');
    }
}

function logToSetupConsole(message, type = 'info') {
    const consoleEl = document.getElementById('setup-console');
    if (!consoleEl) return;

    // Clear initial placeholder
    const placeholder = consoleEl.querySelector('pre');
    if (placeholder && placeholder.querySelector('code')?.textContent?.includes('Waiting')) {
        consoleEl.innerHTML = '';
    }

    const pre = document.createElement('pre');
    const code = document.createElement('code');

    const timestamp = new Date().toLocaleTimeString();
    code.textContent = `[${timestamp}] ${message}`;

    switch (type) {
        case 'error': pre.dataset.prefix = '!'; code.classList.add('text-error'); break;
        case 'success': pre.dataset.prefix = '✓'; code.classList.add('text-success'); break;
        case 'warning': pre.dataset.prefix = '⚠'; code.classList.add('text-warning'); break;
        case 'progress': pre.dataset.prefix = '~'; code.classList.add('text-info'); break;
        default: pre.dataset.prefix = '>'; code.classList.add('text-base-content');
    }

    pre.appendChild(code);
    consoleEl.appendChild(pre);
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

async function downloadModel() {
    const btn = document.getElementById('btn-download-model');
    const progress = document.getElementById('model-progress');
    const progressText = document.getElementById('model-progress-text');

    btn.disabled = true;
    btn.textContent = 'Downloading...';
    logToSetupConsole('Starting model download...', 'info');

    try {
        const response = await fetch('/api/setup/download-model', { method: 'POST' });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const lines = decoder.decode(value).split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        handleDownloadEvent(data, progress, progressText, 'model');
                    } catch (e) {}
                }
            }
        }
    } catch (error) {
        logToSetupConsole('Error: ' + error.message, 'error');
        btn.textContent = 'Retry Download';
        btn.disabled = false;
    }
}

async function downloadLens() {
    const btn = document.getElementById('btn-download-lens');
    const progress = document.getElementById('lens-progress');
    const progressText = document.getElementById('lens-progress-text');

    btn.disabled = true;
    btn.textContent = 'Downloading...';
    logToSetupConsole('Starting lens pack download...', 'info');

    try {
        const response = await fetch('/api/setup/download-lens', { method: 'POST' });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const lines = decoder.decode(value).split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        handleDownloadEvent(data, progress, progressText, 'lens');
                    } catch (e) {}
                }
            }
        }
    } catch (error) {
        logToSetupConsole('Error: ' + error.message, 'error');
        btn.textContent = 'Retry Download';
        btn.disabled = false;
    }
}

function handleDownloadEvent(data, progressEl, textEl, type) {
    if (data.type === 'progress') {
        progressEl.value = data.percent;
        textEl.textContent = data.message;
        logToSetupConsole(`[${data.percent}%] ${data.message}`, 'progress');
    } else if (data.type === 'status') {
        logToSetupConsole(data.message, 'info');
    } else if (data.type === 'complete') {
        progressEl.value = 100;
        textEl.textContent = 'Complete!';
        logToSetupConsole(`${type === 'model' ? 'Model' : 'Lens pack'} loaded successfully!`, 'success');
        checkStatus();
    } else if (data.type === 'error') {
        logToSetupConsole('Error: ' + data.message, 'error');
    }
}

// ============================================================================
// Run Configuration Helpers
// ============================================================================

function updateTotalRuns() {
    const episodeCount = AppState.selectedEpisodes.size;
    const sampleCount = parseInt(document.getElementById('sample-count')?.value || 1);

    let conditionCount = 0;
    // Natural conditions
    if (document.getElementById('run-cond-A')?.checked) conditionCount++;
    if (document.getElementById('run-cond-B')?.checked) conditionCount++;
    if (document.getElementById('run-cond-C')?.checked) conditionCount++;
    // Induced deception conditions
    if (document.getElementById('run-cond-D')?.checked) conditionCount++;
    if (document.getElementById('run-cond-E')?.checked) conditionCount++;
    if (document.getElementById('run-cond-F')?.checked) conditionCount++;

    const totalRuns = episodeCount * sampleCount * conditionCount;
    const el = document.getElementById('total-runs-count');
    if (el) el.textContent = totalRuns;
}

// Set up run config change handlers
document.addEventListener('DOMContentLoaded', () => {
    ['sample-count', 'run-cond-A', 'run-cond-B', 'run-cond-C', 'run-cond-D', 'run-cond-E', 'run-cond-F'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', updateTotalRuns);
    });
});

// ============================================================================
// Episodes Page
// ============================================================================

let suitesLoadingPromise = null;

async function loadSuites() {
    if (suitesLoadingPromise) {
        return suitesLoadingPromise;
    }

    suitesLoadingPromise = (async () => {
        try {
            const response = await fetch('/api/evaluation/suites');
            const data = await response.json();
        AppState.suites = data.suites || [];

        if (!AppState.selectedSuiteId) {
            AppState.selectedSuiteId = AppState.pendingSuiteId || data.default_suite_id || (AppState.suites[0]?.id ?? null);
        }
        AppState.pendingSuiteId = null;

        renderSuiteSelector();
        } catch (error) {
            console.error('Failed to load suites:', error);
            const selector = document.getElementById('suite-selector');
            if (selector) {
                selector.innerHTML = '<option value="">No suites found</option>';
                selector.disabled = true;
            }
        } finally {
            suitesLoadingPromise = null;
        }
    })();

    return suitesLoadingPromise;
}

function renderSuiteSelector() {
    const selector = document.getElementById('suite-selector');
    if (!selector) return;

    if (!AppState.suites.length) {
        selector.innerHTML = '<option value="">No suites discovered</option>';
        selector.disabled = true;
        renderSuiteSummary(null);
        return;
    }

    selector.disabled = false;
    selector.innerHTML = AppState.suites.map(suite => `
        <option value="${suite.id}">${suite.name}${suite.version ? ` (v${suite.version})` : ''}</option>
    `).join('');

    const hasSelected = AppState.suites.some(s => s.id === AppState.selectedSuiteId);
    if (!hasSelected) {
        AppState.selectedSuiteId = AppState.suites[0].id;
    }
    selector.value = AppState.selectedSuiteId;

    const meta = AppState.suites.find(s => s.id === AppState.selectedSuiteId) || AppState.suites[0];
    if (meta) {
        AppState.currentSuiteMeta = {
            id: meta.id,
            name: meta.name,
            version: meta.version,
            behavior_count: meta.behavior_count,
            episode_count: meta.episode_count,
            notes: meta.notes,
        };
        renderSuiteSummary(AppState.currentSuiteMeta);
    }
}

function renderSuiteSummary(meta) {
    const summaryEl = document.getElementById('suite-summary');
    if (!summaryEl) return;

    if (!meta) {
        summaryEl.textContent = 'No episode suites detected in /episodes';
        summaryEl.title = '';
        return;
    }

    const behaviorCount = meta.behavior_count ?? AppState.episodes.length;
    const episodeCount = meta.episode_count ?? AppState.episodes.length;
    const versionText = meta.version ? ` • v${meta.version}` : '';
    summaryEl.textContent = `${behaviorCount || 0} behaviors • ${episodeCount || 0} episodes${versionText}`;
    if (meta.notes) {
        summaryEl.title = meta.notes;
    }
}

function buildClientPreviewTurns(episode) {
    const preview = [];
    const systemRole = episode.roles?.system;
    if (systemRole) {
        preview.push({ speaker: 'system', content: systemRole });
    }

    const context = episode.setup?.context;
    if (context) {
        preview.push({ speaker: 'setup', content: context });
    }

    const temptation = episode.setup?.temptation;
    if (temptation) {
        const pieces = [];
        if (temptation.type) pieces.push(`Temptation type: ${temptation.type}`);
        if (temptation.description) pieces.push(temptation.description);
        else if (temptation.notes) pieces.push(temptation.notes);
        if (temptation.metric) pieces.push(`Metric: ${temptation.metric}`);
        if (temptation.incentive) pieces.push(`Incentive: ${temptation.incentive}`);
        if (pieces.length) {
            preview.push({ speaker: 'setup', content: pieces.join(' ') });
        }
    }

    if (Array.isArray(episode.turns)) {
        preview.push(...episode.turns);
    }

    return preview;
}

function renderTurnsHtml(turns) {
    if (!Array.isArray(turns) || turns.length === 0) {
        return '<p class="text-base-content/50">No turns defined</p>';
    }

    const metaTurns = turns.filter(t => t.meta);
    const convoTurns = turns.filter(t => !t.meta);
    const parts = [];

    if (metaTurns.length) {
        parts.push(`<div class="space-y-2 mb-3">
            ${metaTurns.map(turn => {
                const label = turn.meta === 'setup_temptation' ? 'Temptation' : 'Setup';
                return `
                    <div class="p-2 rounded bg-base-200 border-l-4 border-info">
                        <div class="text-xs uppercase text-info mb-1 tracking-wide">${label}</div>
                        <div class="text-sm">${escapeHtml(turn.content || '')}</div>
                    </div>
                `;
            }).join('')}
        </div>`);
    }

    if (convoTurns.length) {
        parts.push(convoTurns.map(turn => {
            const speaker = turn.speaker || 'user';
            return `
                <div class="p-2 rounded border-l-4 ${getSpeakerBorderColor(speaker)} bg-base-200">
                    <div class="text-xs font-bold uppercase mb-1 text-${getSpeakerColor(speaker)}">${speaker}</div>
                    <div class="text-sm">${escapeHtml(turn.content || '(Assistant response)')}</div>
                </div>
            `;
        }).join(''));
    }

    return parts.join('');
}

async function handleSuiteChange(selectEl) {
    const value = typeof selectEl === 'string' ? selectEl : selectEl?.value;
    if (!value || value === AppState.selectedSuiteId) return;
    AppState.selectedSuiteId = value;
    AppState.pendingSuiteId = null;
    AppState.selectedEpisodes.clear();
    await loadEpisodes();
}

async function loadEpisodes() {
    try {
        if (!AppState.suites.length) {
            await loadSuites();
        }

        if (!AppState.selectedSuiteId && AppState.suites.length) {
            AppState.selectedSuiteId = AppState.suites[0].id;
        }

        if (!AppState.selectedSuiteId) {
            document.getElementById('episode-list').innerHTML = '<p class="text-error">No episode suites available</p>';
            return;
        }

        const response = await fetch(`/api/evaluation/episodes?suite_id=${encodeURIComponent(AppState.selectedSuiteId)}`);
        const data = await response.json();
        if (data.error) {
            document.getElementById('episode-list').innerHTML = `<p class="text-error">${data.error}</p>`;
            renderSuiteSummary(null);
            return;
        }
        AppState.episodes = data.episodes || [];
        if (data.suite) {
            AppState.currentSuiteMeta = data.suite;
            renderSuiteSummary(AppState.currentSuiteMeta);
        }

        const validIds = new Set(AppState.episodes.map(ep => ep.id));
        Array.from(AppState.selectedEpisodes).forEach(id => {
            if (!validIds.has(id)) AppState.selectedEpisodes.delete(id);
        });

        const container = document.getElementById('episode-list');
        if (AppState.episodes.length === 0) {
            container.innerHTML = '<p class="text-base-content/50">No episodes found</p>';
            return;
        }

        container.innerHTML = AppState.episodes.map(ep => `
            <label class="flex items-center gap-3 p-2 rounded cursor-pointer hover:bg-base-200 transition-colors ${AppState.selectedEpisodes.has(ep.id) ? 'bg-base-200' : ''}"
                   data-episode-id="${ep.id}">
                <input type="checkbox" class="checkbox checkbox-primary checkbox-sm"
                       ${AppState.selectedEpisodes.has(ep.id) ? 'checked' : ''}
                       onchange="toggleEpisodeSelection('${ep.id}', event)">
                <div class="flex-1" onclick="showEpisodeDetail('${ep.id}')">
                    <div class="font-medium text-sm">${ep.id}</div>
                    <div class="text-xs text-base-content/60">${formatBehavior(ep.behavior)}</div>
                </div>
            </label>
        `).join('');

        if (AppState.selectedEpisodes.size === 0 && AppState.episodes.length > 0) {
            selectAllEpisodes(true);
        } else {
            updateSelectedCount();
            updateTotalRuns();
        }

    } catch (error) {
        document.getElementById('episode-list').innerHTML = '<p class="text-error">Failed to load episodes</p>';
    }
}

function toggleEpisodeSelection(episodeId, event) {
    if (AppState.selectedEpisodes.has(episodeId)) {
        AppState.selectedEpisodes.delete(episodeId);
    } else {
        AppState.selectedEpisodes.add(episodeId);
    }

    // Update UI
    const item = document.querySelector(`[data-episode-id="${episodeId}"]`);
    if (item) {
        item.classList.toggle('bg-base-200', AppState.selectedEpisodes.has(episodeId));
        const checkbox = item.querySelector('input[type="checkbox"]');
        if (checkbox) checkbox.checked = AppState.selectedEpisodes.has(episodeId);
    }

    updateSelectedCount();
    updateTotalRuns();
}

function selectAllEpisodes(select) {
    AppState.episodes.forEach(ep => {
        if (select) {
            AppState.selectedEpisodes.add(ep.id);
        } else {
            AppState.selectedEpisodes.delete(ep.id);
        }
    });

    // Update UI
    document.querySelectorAll('[data-episode-id]').forEach(item => {
        const id = item.dataset.episodeId;
        item.classList.toggle('bg-base-200', AppState.selectedEpisodes.has(id));
        const checkbox = item.querySelector('input[type="checkbox"]');
        if (checkbox) checkbox.checked = AppState.selectedEpisodes.has(id);
    });

    updateSelectedCount();
    updateTotalRuns();
}

function updateSelectedCount() {
    document.getElementById('selected-episode-count').textContent = AppState.selectedEpisodes.size;
}

async function showEpisodeDetail(episodeId) {
    // Fetch full episode data
    try {
        const episode = AppState.episodes.find(ep => ep.id === episodeId);
        if (!episode) return;

        // Update detail view
        const badge = document.getElementById('episode-behavior-badge');
        badge.textContent = formatBehavior(episode.behavior);
        badge.classList.remove('hidden');

        const content = document.getElementById('episode-detail-content');
        content.innerHTML = `
            <div class="space-y-4">
                <div>
                    <span class="text-gray-400">ID:</span>
                    <span class="font-mono ml-2">${episode.id}</span>
                </div>
                <div>
                    <span class="text-gray-400">Behavior:</span>
                    <span class="ml-2">${formatBehavior(episode.behavior)}</span>
                </div>
                <div>
                    <span class="text-gray-400">Difficulty:</span>
                    <span class="ml-2">${episode.difficulty || 'N/A'}</span>
                </div>
                ${episode.description ? `
                <div>
                    <span class="text-gray-400">Description:</span>
                    <p class="mt-1 text-sm">${episode.description}</p>
                </div>
                ` : ''}
            </div>
        `;

        // Load turns (need full episode data)
        loadEpisodeTurns(episodeId);

    } catch (error) {
        console.error('Failed to load episode detail:', error);
    }
}

async function loadEpisodeTurns(episodeId) {
    try {
        const suiteParam = AppState.selectedSuiteId ? `?suite_id=${encodeURIComponent(AppState.selectedSuiteId)}` : '';
        const response = await fetch(`/api/evaluation/episode/${episodeId}${suiteParam}`);
        const episode = await response.json();

        const turnsContainer = document.getElementById('episode-turns');
        const turns = episode.preview_turns || buildClientPreviewTurns(episode);
        turnsContainer.innerHTML = renderTurnsHtml(turns);

    } catch (error) {
        document.getElementById('episode-turns').innerHTML = '<p class="text-base-content/50">Could not load turns</p>';
    }
}

function getSpeakerColor(speaker) {
    switch (speaker) {
        case 'user': return 'info';
        case 'assistant': return 'base-content/70';
        case 'system': return 'secondary';
        case 'evaluator': return 'warning';
        default: return 'base-content/70';
    }
}

function getSpeakerBorderColor(speaker) {
    switch (speaker) {
        case 'user': return 'border-info';
        case 'assistant': return 'border-base-content/30';
        case 'system': return 'border-secondary';
        case 'evaluator': return 'border-warning';
        default: return 'border-base-content/30';
    }
}

// ============================================================================
// Run Page
// ============================================================================

async function startEvaluation() {
    const episodes = Array.from(AppState.selectedEpisodes);
    const suiteId = AppState.selectedSuiteId;
    const conditions = [];
    // Natural conditions
    if (document.getElementById('run-cond-A')?.checked) conditions.push('A');
    if (document.getElementById('run-cond-B')?.checked) conditions.push('B');
    if (document.getElementById('run-cond-C')?.checked) conditions.push('C');
    // Induced deception conditions
    if (document.getElementById('run-cond-D')?.checked) conditions.push('D');
    if (document.getElementById('run-cond-E')?.checked) conditions.push('E');
    if (document.getElementById('run-cond-F')?.checked) conditions.push('F');

    const sampleCount = parseInt(document.getElementById('sample-count')?.value || 1);

    if (!suiteId) {
        alert('No episode suite selected. Please choose a suite.');
        return;
    }

    if (episodes.length === 0) {
        alert('Please select at least one episode');
        return;
    }

    if (conditions.length === 0) {
        alert('Please select at least one condition');
        return;
    }

    // Navigate to run page
    navigateTo('run');
    AppState.isRunning = true;

    const progressEl = document.getElementById('run-progress');
    const percentEl = document.getElementById('run-progress-percent');
    const statusText = document.getElementById('run-status-text');
    const logEl = document.getElementById('run-log');
    const tokensEl = document.getElementById('live-tokens');
    const abortBtn = document.getElementById('btn-abort');

    // Reset UI
    progressEl.value = 0;
    percentEl.textContent = '0%';
    statusText.textContent = 'Starting evaluation...';
    logEl.innerHTML = `<pre data-prefix=">"><code class="text-info">Starting suite ${suiteId}...</code></pre>`;
    tokensEl.innerHTML = '';
    abortBtn.classList.remove('hidden');

    // Reset history state
    AppState.episodeHistory = [];
    AppState.currentHistoryIndex = -1;
    AppState.liveTokens = [];
    AppState.liveMetadata = [];
    updateHistoryNav();

    try {
        const response = await fetch('/api/evaluation/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                episode_ids: episodes,
                conditions,
                sample_count: sampleCount,
                suite_id: suiteId
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const lines = decoder.decode(value).split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        handleRunEvent(data);
                    } catch (e) {}
                }
            }
        }
    } catch (error) {
        logToRunLog('Error: ' + error.message, 'error');
    }

    AppState.isRunning = false;
    abortBtn.classList.add('hidden');
}

function handleRunEvent(data) {
    const progressEl = document.getElementById('run-progress');
    const percentEl = document.getElementById('run-progress-percent');
    const statusText = document.getElementById('run-status-text');
    const episodeInfo = document.getElementById('current-episode-info');
    const conditionLabel = document.getElementById('current-condition-label');
    const tokensEl = document.getElementById('live-tokens');
    const turnsEl = document.getElementById('episode-turns-display');

    switch (data.type) {
        case 'start':
            statusText.textContent = `Running ${data.total_episodes} episodes, conditions: ${data.conditions.join(', ')}`;
            logToRunLog(`Starting evaluation: ${data.total_episodes} episodes`, 'info');
            if (data.suite_id) {
                const suiteLabel = data.suite_name || data.suite_id;
                logToRunLog(`Suite: ${suiteLabel}`, 'info');
            }
            break;

        case 'episode_turns': {
            // Display episode turns in the live generation area (chat-style)
            const convoEl = document.getElementById('live-conversation');
            const placeholderEl = document.getElementById('live-placeholder');
            if (convoEl && data.turns) {
                const turnsHtml = renderTurnsHtml(data.turns);
                convoEl.innerHTML = turnsHtml;
                if (placeholderEl) placeholderEl.classList.add('hidden');
                // Save for history
                AppState.liveTurnsHtml = turnsHtml;
            }
            // Also update the side panel for reference
            if (turnsEl && data.turns) {
                turnsEl.innerHTML = renderTurnsHtml(data.turns);
            }
            break;
        }

        case 'episode_start': {
            // Save previous episode to history before starting new one
            if (AppState.liveTokens.length > 0) {
                saveCurrentToHistory();
            }

            // Reset live state for new episode
            AppState.liveTokens = [];
            AppState.liveMetadata = [];
            AppState.currentHistoryIndex = -1; // Switch to live view

            episodeInfo.innerHTML = `
                <div class="font-medium">${data.episode_id}</div>
                <div class="text-sm text-gray-400">${formatBehavior(data.behavior)}</div>
            `;
            episodeInfo.dataset.episodeId = data.episode_id;
            episodeInfo.dataset.condition = data.condition;
            episodeInfo.dataset.behavior = data.behavior;

            conditionLabel.textContent = data.condition;
            conditionLabel.className = `font-bold ml-2 text-${getConditionColor(data.condition)}`;

            // Clear the response area and show generating indicator
            const responseEl = document.getElementById('live-response');
            const placeholder = document.getElementById('live-placeholder');
            if (responseEl) responseEl.innerHTML = '<span class="text-base-content/50 animate-pulse">Generating...</span>';
            // Keep conversation from episode_turns event
            if (placeholder) placeholder.classList.add('hidden');

            clearRunPanels();
            updateHistoryNav();
            logToRunLog(`Running ${data.episode_id} [${data.condition}]`, 'info');
            break;
        }

        case 'token':
            // Live token streaming with highlighting
            if (data.token) {
                // Track in live state
                AppState.liveTokens.push(data.token);
                AppState.liveMetadata.push(data.metadata || {});

                // Only show if we're viewing live (not history)
                if (AppState.currentHistoryIndex === -1) {
                    appendToken(data.token, data.metadata);
                }

                // Update safety alerts if safety intensity is high
                if (data.metadata?.safety_intensity > 0.3) {
                    addSafetyAlert(data.token, data.metadata);
                }

                // Update steering panel if steering is active
                if (data.metadata?.steering_active) {
                    addSteeringEvent(data.token, data.metadata);
                }
            }
            break;

        case 'episode_complete':
            const result = data.result;
            const status = result.manipulation_detected ? 'MANIPULATION' : 'OK';
            const color = result.manipulation_detected ? 'error' : 'success';
            logToRunLog(`  → ${status} (score: ${(result.peak_detection_score * 100).toFixed(0)}%, interventions: ${result.interventions || 0})`, color);

            const percent = Math.round(data.progress * 100);
            progressEl.value = percent;
            percentEl.textContent = percent + '%';
            break;

        case 'complete':
            // Save final episode to history
            if (AppState.liveTokens.length > 0) {
                saveCurrentToHistory();
            }

            statusText.textContent = 'Evaluation complete!';
            progressEl.value = 100;
            percentEl.textContent = '100%';
            AppState.currentRunId = data.run_id;
            updateHistoryNav();

            // Show summary
            if (data.summary?.by_condition) {
                displaySummary(data.summary);
            }

            // Add prominent view results button
            addViewResultsButton();
            break;

        case 'error':
            logToRunLog('Error: ' + data.message, 'error');
            statusText.textContent = 'Error occurred';
            break;

        case 'aborted':
            logToRunLog('Evaluation aborted', 'warning');
            statusText.textContent = 'Aborted';
            break;
    }
}

function appendToken(token, metadata = {}) {
    const mainContainer = document.getElementById('live-tokens');
    const responseContainer = document.getElementById('live-response');
    const placeholder = document.getElementById('live-placeholder');

    // Don't append if we're viewing history (not live)
    if (AppState.currentHistoryIndex !== -1) {
        return;
    }

    // On first token, clear the response area and hide placeholder
    if (AppState.liveTokens.length === 1) {
        if (responseContainer) responseContainer.innerHTML = '';
        if (placeholder) placeholder.classList.add('hidden');
    }

    // Append to response container (or fallback to main)
    const targetContainer = responseContainer || mainContainer;
    appendTokenDirect(targetContainer, token, metadata);
    mainContainer.scrollTop = mainContainer.scrollHeight;

    // Update token count
    const count = mainContainer.querySelectorAll('.hatcat-token').length;
    document.getElementById('token-count').textContent = `${count} tokens`;
}

function showTokenTooltip(e) {
    const span = e.target;
    let metadata = {};
    try {
        metadata = JSON.parse(span.dataset.metadata || '{}');
    } catch (err) {}

    let tooltip = document.getElementById('hatcat-tooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'hatcat-tooltip';
        tooltip.className = 'hatcat-tooltip';
        document.body.appendChild(tooltip);
    }

    // Build tooltip content
    let html = `<div class="font-medium mb-2">"${span.textContent}"</div>`;

    if (metadata.top_concepts && metadata.top_concepts.length > 0) {
        html += '<div class="text-xs text-gray-400 mb-1">Top Concepts:</div>';
        for (const [concept, score] of metadata.top_concepts) {
            const color = conceptToColor(concept, score);
            html += `
                <div class="tooltip-concept">
                    <span class="tooltip-swatch" style="background: ${color}"></span>
                    <span>${concept}</span>
                    <span class="text-gray-500 ml-auto">${(score * 100).toFixed(0)}%</span>
                </div>
            `;
        }
    }

    if (metadata.safety_intensity > 0) {
        html += `<div class="mt-2 text-red-400 text-xs">Safety intensity: ${(metadata.safety_intensity * 100).toFixed(0)}%</div>`;
    }

    // Show violations from ASK schema
    if (metadata.violations && metadata.violations.length > 0) {
        html += '<div class="mt-2 text-red-400 text-xs font-medium">Violations:</div>';
        for (const v of metadata.violations) {
            html += `<div class="text-red-300 text-xs ml-2">• ${v.simplex || v.constraint}: ${(v.deviation * 100).toFixed(0)}% > ${(v.threshold * 100).toFixed(0)}%</div>`;
        }
    }

    // Show steering details from ASK schema
    if (metadata.steering_active) {
        html += `<div class="mt-2 text-yellow-400 text-xs font-medium">⚡ HUSH Steering:</div>`;
        if (metadata.steering_applied && metadata.steering_applied.length > 0) {
            for (const s of metadata.steering_applied) {
                const direction = s.direction === 'suppress' ? '↓ Suppress' : '↑ Amplify';
                const strength = s.strength ? ` (${(s.strength * 100).toFixed(0)}%)` : '';
                html += `<div class="text-yellow-300 text-xs ml-2">• ${direction} ${s.concept}${strength}</div>`;
            }
        } else {
            html += `<div class="text-yellow-300 text-xs ml-2">• Active</div>`;
        }
    }

    // Show simplex deviations if significant
    if (metadata.simplex_deviations && Object.keys(metadata.simplex_deviations).length > 0) {
        const topDeviations = Object.entries(metadata.simplex_deviations)
            .filter(([k, v]) => Math.abs(v) > 0.05)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 3);
        if (topDeviations.length > 0) {
            html += '<div class="mt-2 text-orange-400 text-xs font-medium">Simplex Drift:</div>';
            for (const [simplex, deviation] of topDeviations) {
                const sign = deviation > 0 ? '+' : '';
                html += `<div class="text-orange-300 text-xs ml-2">• ${simplex}: ${sign}${(deviation * 100).toFixed(0)}%</div>`;
            }
        }
    }

    // Show hidden state norm if available
    if (metadata.hidden_state_norm && metadata.hidden_state_norm > 0) {
        html += `<div class="mt-2 text-gray-500 text-xs">Hidden state norm: ${metadata.hidden_state_norm.toFixed(2)}</div>`;
    }

    // Show significance scoring (decision vs filler token)
    if (metadata.significance !== undefined) {
        const sigPct = (metadata.significance * 100).toFixed(0);
        const sigColor = metadata.is_filler ? 'text-gray-400' : (metadata.significance > 0.7 ? 'text-cyan-400' : 'text-cyan-600');
        const sigLabel = metadata.is_filler ? 'Filler token' : (metadata.significance > 0.7 ? 'Decision point' : 'Transitional');
        html += `<div class="mt-2 ${sigColor} text-xs font-medium">Significance: ${sigPct}% (${sigLabel})</div>`;

        // Show layer-wise entropy if available
        if (metadata.entropy_by_layer && Object.keys(metadata.entropy_by_layer).length > 0) {
            html += '<div class="text-cyan-700 text-xs ml-2">';
            const layers = ['early', 'mid', 'late'];
            const entropies = [];
            for (const layer of layers) {
                if (metadata.entropy_by_layer[layer] !== undefined) {
                    entropies.push(`${layer}: ${metadata.entropy_by_layer[layer].toFixed(2)}`);
                }
            }
            if (entropies.length > 0) {
                html += `Entropy: ${entropies.join(' → ')}`;
            }
            html += '</div>';
        }

        // Show activation delta
        if (metadata.activation_delta && metadata.activation_delta > 0) {
            html += `<div class="text-cyan-700 text-xs ml-2">Δ Hidden: ${metadata.activation_delta.toFixed(3)}</div>`;
        }
    }

    tooltip.innerHTML = html;
    tooltip.classList.remove('hidden');

    // Position tooltip
    const rect = span.getBoundingClientRect();
    tooltip.style.left = rect.left + 'px';
    tooltip.style.top = (rect.bottom + 8) + 'px';
}

function hideTokenTooltip() {
    const tooltip = document.getElementById('hatcat-tooltip');
    if (tooltip) {
        tooltip.classList.add('hidden');
    }
}

function conceptToColor(concept, activation) {
    // Hash concept name to get hue
    let hash = 0;
    for (let i = 0; i < concept.length; i++) {
        hash = concept.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash) % 360;
    const saturation = 40 + (activation * 40);
    const lightness = 60 - (activation * 30);
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

function addSafetyAlert(token, metadata) {
    const alertsEl = document.getElementById('safety-alerts-list');
    if (!alertsEl) return;

    // Clear placeholder
    const placeholder = alertsEl.querySelector('.text-gray-500');
    if (placeholder) alertsEl.innerHTML = '';

    const alert = document.createElement('div');
    alert.className = 'alert-item';

    // Format concepts
    const concepts = metadata.top_concepts?.slice(0, 2).map(([c, s]) =>
        `${c}: ${(s * 100).toFixed(0)}%`
    ).join(', ') || '';

    // Format violations from ASK schema
    let violationsHtml = '';
    if (metadata.violations && metadata.violations.length > 0) {
        violationsHtml = '<div class="text-xs text-red-400 mt-1">' +
            metadata.violations.slice(0, 2).map(v =>
                `Violation: ${v.simplex || v.constraint} (${(v.deviation * 100).toFixed(0)}% > ${(v.threshold * 100).toFixed(0)}%)`
            ).join('<br>') + '</div>';
    }

    // Format simplex deviations if significant
    let deviationsHtml = '';
    if (metadata.simplex_deviations && Object.keys(metadata.simplex_deviations).length > 0) {
        const topDeviations = Object.entries(metadata.simplex_deviations)
            .filter(([k, v]) => Math.abs(v) > 0.1)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 2);
        if (topDeviations.length > 0) {
            deviationsHtml = '<div class="text-xs text-orange-400 mt-1">Drift: ' +
                topDeviations.map(([k, v]) => `${k}: ${v > 0 ? '+' : ''}${(v * 100).toFixed(0)}%`).join(', ') +
            '</div>';
        }
    }

    alert.innerHTML = `
        <div class="font-medium">"${escapeHtml(token)}"</div>
        <div class="text-xs text-gray-400 mt-1">${concepts}</div>
        ${violationsHtml}
        ${deviationsHtml}
    `;

    alertsEl.insertBefore(alert, alertsEl.firstChild);

    // Keep max 5 alerts
    while (alertsEl.children.length > 5) {
        alertsEl.removeChild(alertsEl.lastChild);
    }
}

function addSteeringEvent(token, metadata) {
    const steeringEl = document.getElementById('active-steerings-list');
    if (!steeringEl) return;

    // Clear placeholder
    const placeholder = steeringEl.querySelector('.text-gray-500');
    if (placeholder) steeringEl.innerHTML = '';

    const event = document.createElement('div');
    event.className = 'steering-item';

    // Format steering details from ASK schema
    let steeringDetailsHtml = '';
    if (metadata.steering_applied && metadata.steering_applied.length > 0) {
        steeringDetailsHtml = '<div class="text-xs text-yellow-300 mt-1 ml-6">' +
            metadata.steering_applied.slice(0, 2).map(s => {
                const direction = s.direction === 'suppress' ? '↓' : '↑';
                const strength = s.strength ? ` (${(s.strength * 100).toFixed(0)}%)` : '';
                return `${direction} ${s.concept}${strength}`;
            }).join('<br>') + '</div>';
    }

    event.innerHTML = `
        <div class="flex items-center">
            <span class="text-yellow-400">⚡</span>
            <span class="flex-1 ml-2">"${escapeHtml(token)}"</span>
        </div>
        ${steeringDetailsHtml}
    `;

    steeringEl.insertBefore(event, steeringEl.firstChild);

    // Keep max 5 events
    while (steeringEl.children.length > 5) {
        steeringEl.removeChild(steeringEl.lastChild);
    }
}

function clearRunPanels() {
    // Reset steering and safety panels when starting new episode/condition
    const alertsEl = document.getElementById('safety-alerts-list');
    const steeringEl = document.getElementById('active-steerings-list');
    const turnsEl = document.getElementById('episode-turns-display');

    if (alertsEl) alertsEl.innerHTML = '<p class="text-gray-500">No alerts</p>';
    if (steeringEl) steeringEl.innerHTML = '<p class="text-gray-500">No active steerings</p>';
}

function logToRunLog(message, type = 'info') {
    const logEl = document.getElementById('run-log');
    if (!logEl) return;

    // Clear initial placeholder
    const placeholder = logEl.querySelector('pre code')?.textContent;
    if (placeholder && (placeholder.includes('Waiting') || placeholder.includes('Starting'))) {
        logEl.innerHTML = '';
    }

    const pre = document.createElement('pre');
    const code = document.createElement('code');
    code.textContent = message;

    switch (type) {
        case 'error': pre.dataset.prefix = '!'; code.classList.add('text-error'); break;
        case 'success': pre.dataset.prefix = '✓'; code.classList.add('text-success'); break;
        case 'warning': pre.dataset.prefix = '⚠'; code.classList.add('text-warning'); break;
        default: pre.dataset.prefix = '>'; code.classList.add('text-base-content');
    }

    pre.appendChild(code);
    logEl.appendChild(pre);
    logEl.scrollTop = logEl.scrollHeight;
}

async function abortEvaluation() {
    await fetch('/api/evaluation/abort');
}

function getConditionColor(cond) {
    switch (cond) {
        // Natural conditions
        case 'A': return 'base-content';
        case 'B': return 'info';
        case 'C': return 'success';
        // Induced deception conditions
        case 'D': return 'error';
        case 'E': return 'warning';
        case 'F': return 'secondary';
        default: return 'base-content/70';
    }
}

// ============================================================================
// Results Page
// ============================================================================

async function loadRunsList() {
    try {
        const response = await fetch('/api/results/list');
        const data = await response.json();

        const selector = document.getElementById('run-selector');
        selector.innerHTML = '<option value="">Select run...</option>' +
            data.results.map(r =>
                `<option value="${r.run_id}">${r.run_id} (${r.episode_count} episodes${r.suite?.name ? ` • ${r.suite.name}` : ''})</option>`
            ).join('');

        // Auto-select latest or current run
        if (AppState.currentRunId) {
            selector.value = AppState.currentRunId;
            loadRunResults();
        } else if (data.results.length > 0) {
            selector.value = data.results[0].run_id;
            loadRunResults();
        }
    } catch (error) {
        console.error('Failed to load runs list:', error);
    }
}

async function loadRunResults() {
    const runId = document.getElementById('run-selector').value;
    if (!runId) return;

    try {
        // Load results
        const response = await fetch(`/api/results/${runId}`);
        AppState.currentResults = await response.json();

        // Update summary stats - Natural conditions
        const summary = AppState.currentResults.summary?.by_condition || {};

        // Helper to format stat with pass/fail/null breakdown
        const formatStat = (cond) => {
            const s = summary[cond];
            if (!s) return 'N/A';
            // Show fail rate as main stat
            return `${(s.fail_rate || s.rate || 0).toFixed(0)}%`;
        };

        // Helper to format detailed tooltip
        const formatDetail = (cond) => {
            const s = summary[cond];
            if (!s) return '';
            const parts = [];
            if (s.pass_rate !== undefined) parts.push(`Pass: ${s.pass_rate}%`);
            if (s.fail_rate !== undefined) parts.push(`Fail: ${s.fail_rate}%`);
            if (s.null_rate !== undefined) parts.push(`Null: ${s.null_rate}%`);
            if (s.avg_confidence !== undefined) parts.push(`Conf: ${(s.avg_confidence * 100).toFixed(0)}%`);
            if (s.confidence_stddev !== undefined && s.confidence_stddev > 0) {
                parts.push(`StdDev: ${(s.confidence_stddev * 100).toFixed(0)}%`);
            }
            return parts.join(' | ');
        };

        // Update stats display
        document.getElementById('stat-baseline').textContent = formatStat('A');
        document.getElementById('stat-monitor').textContent = formatStat('B');
        document.getElementById('stat-harness').textContent = formatStat('C');

        // Add tooltips with detailed stats
        ['A', 'B', 'C', 'D', 'E', 'F'].forEach(cond => {
            const el = document.getElementById(`stat-${cond === 'A' ? 'baseline' : cond === 'B' ? 'monitor' : cond === 'C' ? 'harness' : cond === 'D' ? 'induced' : cond === 'E' ? 'induced-mon' : 'induced-steer'}`);
            if (el) el.title = formatDetail(cond);
        });

        // Induced conditions
        document.getElementById('stat-induced').textContent = formatStat('D');
        document.getElementById('stat-induced-mon').textContent = formatStat('E');
        document.getElementById('stat-induced-steer').textContent = formatStat('F');

        // Calculate delta: how much does steering (F) counteract induced deception (D)?
        const inducedRate = summary.D?.fail_rate || summary.D?.rate || 0;
        const counteractedRate = summary.F?.fail_rate || summary.F?.rate || inducedRate;

        // Delta shows how much F reduces manipulation vs D (positive = F is better)
        let delta;
        if (summary.D && summary.F) {
            delta = inducedRate - counteractedRate;  // How much better is F than D
        } else if (summary.C && summary.A) {
            // Fallback: show C vs A reduction
            const baseline = summary.A?.fail_rate || summary.A?.rate || 0;
            const honest = summary.C?.fail_rate || summary.C?.rate || 0;
            delta = baseline - honest;
        } else {
            delta = 0;
        }
        document.getElementById('stat-reduction').textContent = delta.toFixed(0) + '%';

        // Update stats detail section if it exists
        const statsDetail = document.getElementById('stats-detail');
        if (statsDetail) {
            let html = '<div class="grid grid-cols-3 gap-4 text-sm">';
            ['A', 'B', 'C'].forEach(cond => {
                const s = summary[cond];
                if (s) {
                    html += `<div class="p-2 bg-base-200 rounded">
                        <div class="font-bold">Condition ${cond}</div>
                        <div>Pass: ${s.pass_rate || 0}%</div>
                        <div>Fail: ${s.fail_rate || 0}%</div>
                        <div>Null: ${s.null_rate || 0}%</div>
                        <div class="text-xs opacity-70">Conf: ${((s.avg_confidence || 0) * 100).toFixed(0)}% ± ${((s.confidence_stddev || 0) * 100).toFixed(0)}%</div>
                    </div>`;
                }
            });
            html += '</div>';
            statsDetail.innerHTML = html;
        }

        // Load charts
        const chartResponse = await fetch(`/api/results/${runId}/chart/comparison`);
        const chartData = await chartResponse.json();
        Plotly.newPlot('comparison-chart', chartData.data, chartData.layout, { responsive: true });

        const intResponse = await fetch(`/api/results/${runId}/chart/interventions`);
        const intData = await intResponse.json();
        Plotly.newPlot('intervention-chart', intData.data, intData.layout, { responsive: true });

        // Populate episode selector
        const episodeSelector = document.getElementById('result-episode-selector');
        episodeSelector.innerHTML = '<option value="">Select episode to inspect...</option>' +
            AppState.currentResults.episodes.map(ep => {
                const statuses = ['A', 'B', 'C'].map(c =>
                    ep.conditions[c]?.manipulation_detected ? '!' : '✓'
                ).join('');
                return `<option value="${ep.episode_id}">${ep.episode_id} (${formatBehavior(ep.behavior)}) [${statuses}]</option>`;
            }).join('');

    } catch (error) {
        console.error('Failed to load results:', error);
    }
}

function loadEpisodeDetail() {
    const episodeId = document.getElementById('result-episode-selector').value;
    if (!episodeId || !AppState.currentResults) return;

    const episode = AppState.currentResults.episodes.find(ep => ep.episode_id === episodeId);
    if (!episode) return;

    document.getElementById('episode-detail-view').classList.remove('hidden');

    // All 6 conditions: Natural (A/B/C) + Induced (D/E/F)
    ['A', 'B', 'C', 'D', 'E', 'F'].forEach(cond => {
        const data = episode.conditions[cond] || {};
        const responseEl = document.getElementById(`detail-response-${cond}`);
        const statusEl = document.getElementById(`detail-status-${cond}`);

        if (responseEl) {
            responseEl.textContent = data.response || '(not run)';
        }

        if (statusEl) {
            if (!data.response && !data.manipulation_detected) {
                statusEl.textContent = 'N/A';
                statusEl.className = 'badge badge-ghost';
            } else if (data.manipulation_detected) {
                statusEl.textContent = 'MANIPULATION';
                statusEl.className = 'badge badge-error';
            } else {
                statusEl.textContent = 'OK';
                statusEl.className = 'badge badge-success';
            }
        }
    });

    loadXDBLog();
}

function loadXDBLog() {
    const episodeId = document.getElementById('result-episode-selector').value;
    const condition = document.getElementById('xdb-condition-selector').value;

    if (!episodeId || !AppState.currentResults) return;

    const episode = AppState.currentResults.episodes.find(ep => ep.episode_id === episodeId);
    if (!episode) return;

    const data = episode.conditions[condition] || {};
    const viewer = document.getElementById('xdb-log-viewer');

    // Build header with ASK schema summary
    const totalViolations = (data.ticks || []).reduce((sum, t) => sum + (t.violations?.length || 0), 0);
    const totalSteering = (data.ticks || []).reduce((sum, t) => sum + (t.steering_applied?.length || 0), 0);
    const scoreResult = data.score_result || 'unknown';
    const scoreConfidence = data.score_confidence || 0;

    // Significance summary (decision vs filler tokens)
    const ticksWithSig = (data.ticks || []).filter(t => t.significance !== undefined);
    const decisionTokens = ticksWithSig.filter(t => t.significance > 0.7 && !t.is_filler).length;
    const fillerTokens = ticksWithSig.filter(t => t.is_filler).length;
    const avgSignificance = ticksWithSig.length > 0
        ? ticksWithSig.reduce((sum, t) => sum + t.significance, 0) / ticksWithSig.length
        : 0;

    let html = `
        <div class="xdb-entry border-b border-gray-700 pb-3 mb-3">
            <div class="text-sm font-medium mb-2 text-gray-300">ASK Audit Log Entry</div>
            <div class="grid grid-cols-3 gap-2 text-sm">
                <div><span class="text-gray-400">Episode:</span> ${episodeId}</div>
                <div><span class="text-gray-400">Condition:</span> ${condition}</div>
                <div><span class="text-gray-400">Detection:</span>
                    <span class="${data.manipulation_detected ? 'text-red-400' : 'text-green-400'}">
                        ${data.manipulation_detected ? 'Yes' : 'No'}
                    </span>
                </div>
                <div><span class="text-gray-400">Peak Score:</span> ${(data.peak_detection_score * 100).toFixed(1)}%</div>
                <div><span class="text-gray-400">Interventions:</span> ${data.interventions || 0}</div>
                <div><span class="text-gray-400">Corrections:</span> ${data.corrections || 0}</div>
                <div><span class="text-gray-400">Score Result:</span>
                    <span class="${scoreResult === 'pass' ? 'text-green-400' : scoreResult === 'fail' ? 'text-red-400' : 'text-gray-500'}">
                        ${scoreResult.toUpperCase()}
                    </span>
                </div>
                <div><span class="text-gray-400">Confidence:</span> ${(scoreConfidence * 100).toFixed(0)}%</div>
                <div><span class="text-gray-400">Tick Count:</span> ${(data.ticks || []).length}</div>
            </div>
            <div class="grid grid-cols-2 gap-2 text-sm mt-2 pt-2 border-t border-gray-700">
                <div><span class="text-red-400">Total Violations:</span> ${totalViolations}</div>
                <div><span class="text-yellow-400">Total Steering Events:</span> ${totalSteering}</div>
            </div>
            ${ticksWithSig.length > 0 ? `
            <div class="grid grid-cols-3 gap-2 text-sm mt-2 pt-2 border-t border-gray-700">
                <div><span class="text-cyan-400">Decision Tokens:</span> ${decisionTokens}</div>
                <div><span class="text-gray-500">Filler Tokens:</span> ${fillerTokens}</div>
                <div><span class="text-cyan-600">Avg Significance:</span> ${(avgSignificance * 100).toFixed(0)}%</div>
            </div>
            ` : ''}
        </div>
    `;

    // Render tick data if available
    const ticks = data.ticks || [];

    if (ticks.length > 0) {
        html += '<div class="text-sm text-gray-400 mb-2">Token-by-Token Analysis (ASK Schema):</div>';

        ticks.forEach((tick, idx) => {
            const token = tick.token || '';
            const concepts = tick.concepts || [];
            const safetyIntensity = tick.safety_intensity || 0;
            const steeringActive = tick.steering_active || false;
            const violations = tick.violations || [];
            const steeringApplied = tick.steering_applied || [];
            const simplexDeviations = tick.simplex_deviations || {};

            // Determine row styling
            let rowClass = '';
            if (steeringActive) rowClass = 'bg-yellow-900/30 border-l-2 border-yellow-500';
            else if (violations.length > 0) rowClass = 'bg-red-900/30 border-l-2 border-red-600';
            else if (safetyIntensity > 0.3) rowClass = 'bg-red-900/20 border-l-2 border-red-500';

            html += `
                <div class="xdb-entry ${rowClass} p-2 rounded mb-1">
                    <div class="flex items-center gap-2">
                        <span class="text-gray-500 w-8">#${tick.tick_id || idx}</span>
                        <span class="xdb-token">"${escapeHtml(token)}"</span>
                        ${steeringActive ? '<span class="text-yellow-400 text-xs">⚡ STEERED</span>' : ''}
                        ${violations.length > 0 ? '<span class="text-red-400 text-xs">⚠ VIOLATION</span>' : ''}
                        ${safetyIntensity > 0.3 && violations.length === 0 ? '<span class="text-orange-400 text-xs">⚠ SAFETY</span>' : ''}
                        ${tick.timestamp ? `<span class="text-gray-600 text-xs ml-auto">${tick.timestamp.split('T')[1]?.slice(0, 8) || ''}</span>` : ''}
                    </div>
            `;

            // Show concepts
            if (concepts.length > 0) {
                html += '<div class="xdb-concepts">';
                concepts.forEach(([concept, score]) => {
                    const color = conceptToColor(concept, score);
                    html += `
                        <span class="xdb-concept" style="background: ${color}; color: ${getContrastColor(color)}">
                            ${concept}: ${(score * 100).toFixed(0)}%
                        </span>
                    `;
                });
                html += '</div>';
            }

            // Show violations (ASK schema)
            if (violations.length > 0) {
                html += '<div class="mt-1 ml-8 text-xs">';
                violations.forEach(v => {
                    html += `
                        <div class="text-red-400">
                            Constraint "${v.simplex || v.constraint}" violated:
                            ${(v.deviation * 100).toFixed(1)}% deviation
                            (threshold: ${(v.threshold * 100).toFixed(1)}%)
                            ${v.severity ? ` [${v.severity}]` : ''}
                        </div>
                    `;
                });
                html += '</div>';
            }

            // Show steering applied (ASK schema)
            if (steeringApplied.length > 0) {
                html += '<div class="mt-1 ml-8 text-xs">';
                steeringApplied.forEach(s => {
                    const direction = s.direction === 'suppress' ? '↓' : '↑';
                    html += `
                        <div class="text-yellow-400">
                            ${direction} ${s.action || 'Steer'} "${s.concept}"
                            ${s.strength ? `(strength: ${(s.strength * 100).toFixed(0)}%)` : ''}
                            ${s.contrastive ? `→ ${s.contrastive}` : ''}
                        </div>
                    `;
                });
                html += '</div>';
            }

            // Show simplex deviations if significant
            const significantDeviations = Object.entries(simplexDeviations)
                .filter(([k, v]) => Math.abs(v) > 0.05)
                .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                .slice(0, 3);
            if (significantDeviations.length > 0) {
                html += '<div class="mt-1 ml-8 text-xs text-orange-400">Drift: ';
                html += significantDeviations.map(([simplex, dev]) => {
                    const sign = dev > 0 ? '+' : '';
                    return `${simplex}: ${sign}${(dev * 100).toFixed(0)}%`;
                }).join(', ');
                html += '</div>';
            }

            // Show significance scoring (decision vs filler classification)
            const significance = tick.significance;
            const isFiller = tick.is_filler;
            const entropyByLayer = tick.entropy_by_layer || {};
            if (significance !== undefined) {
                const sigPct = (significance * 100).toFixed(0);
                const sigLabel = isFiller ? 'Filler' : (significance > 0.7 ? 'Decision' : 'Transitional');
                const sigColor = isFiller ? 'text-gray-500' : (significance > 0.7 ? 'text-cyan-400' : 'text-cyan-600');
                html += `<div class="mt-1 ml-8 text-xs ${sigColor}">`;
                html += `Significance: ${sigPct}% (${sigLabel})`;
                // Show layer-wise entropy cascade
                const layers = ['early', 'mid', 'late'];
                const entropies = layers.filter(l => entropyByLayer[l] !== undefined).map(l => `${l}: ${entropyByLayer[l].toFixed(2)}`);
                if (entropies.length > 0) {
                    html += ` | Entropy: ${entropies.join(' → ')}`;
                }
                if (tick.activation_delta) {
                    html += ` | Δ: ${tick.activation_delta.toFixed(3)}`;
                }
                html += '</div>';
            }

            html += '</div>';
        });
    } else {
        html += `
            <div class="text-gray-500 italic">
                No detailed tick data available for this condition.
                ${condition === 'A' ? 'Baseline condition (A) does not use HAT monitoring.' : 'Run with token streaming enabled to capture tick data.'}
            </div>
        `;
    }

    viewer.innerHTML = html;
}

// ============================================================================
// Compliance
// ============================================================================

async function showComplianceModal() {
    const modal = document.getElementById('compliance-modal');
    const content = document.getElementById('compliance-mapping-content');
    modal.showModal();

    try {
        const response = await fetch('/api/compliance/eu-mapping');
        const data = await response.json();

        let html = '';
        for (const [article, articleData] of Object.entries(data)) {
            html += `
                <div class="border-b border-base-300 pb-4 mb-4">
                    <h4 class="font-semibold text-primary">${article}</h4>
                    <p class="text-sm text-base-content/60 mb-2">${articleData.description || ''}</p>
            `;

            for (const [req, reqData] of Object.entries(articleData.requirements || {})) {
                const statusClass = reqData.status === 'implemented' ? 'text-success' : 'text-warning';
                html += `
                    <div class="ml-4 mt-2 text-sm">
                        <div class="flex items-center gap-2">
                            <span class="${statusClass}">✓</span>
                            <span>${req}</span>
                        </div>
                        <div class="ml-5 text-base-content/50">${reqData.component}</div>
                    </div>
                `;
            }
            html += '</div>';
        }

        content.innerHTML = html;
    } catch (error) {
        content.innerHTML = '<p class="text-error">Failed to load compliance mapping</p>';
    }
}

function hideComplianceModal() {
    document.getElementById('compliance-modal').close();
}

async function exportReport(format) {
    const runId = document.getElementById('run-selector')?.value;
    if (!runId) {
        alert('Please select a run first');
        return;
    }

    try {
        const response = await fetch('/api/compliance/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ run_id: runId, format })
        });

        const data = await response.json();
        if (data.report_url) {
            window.open(data.report_url, '_blank');
        }
    } catch (error) {
        alert('Failed to export: ' + error.message);
    }
}

// ============================================================================
// Episode History Navigation
// ============================================================================

function saveCurrentToHistory() {
    const episodeInfo = document.getElementById('current-episode-info');
    if (!episodeInfo || AppState.liveTokens.length === 0) return;

    // Capture current state of all panels
    const alertsEl = document.getElementById('safety-alerts-list');
    const steeringEl = document.getElementById('active-steerings-list');

    const entry = {
        episode_id: episodeInfo.dataset.episodeId || 'unknown',
        condition: episodeInfo.dataset.condition || '?',
        behavior: episodeInfo.dataset.behavior || '',
        tokens: [...AppState.liveTokens],
        metadata: [...AppState.liveMetadata],
        // Save HTML content of panels for restoration (use tracked state)
        turnsHtml: AppState.liveTurnsHtml || '',
        alertsHtml: alertsEl ? alertsEl.innerHTML : '',
        steeringHtml: steeringEl ? steeringEl.innerHTML : '',
        // Also save structured data for safety alerts and steering
        safetyAlerts: [],
        steeringEvents: [],
    };

    // Extract structured data from metadata
    AppState.liveMetadata.forEach((meta, idx) => {
        if (meta.safety_intensity > 0.3) {
            entry.safetyAlerts.push({
                token: AppState.liveTokens[idx],
                concepts: meta.top_concepts || [],
                intensity: meta.safety_intensity
            });
        }
        if (meta.steering_active) {
            entry.steeringEvents.push({
                token: AppState.liveTokens[idx],
                position: idx
            });
        }
    });

    AppState.episodeHistory.push(entry);
    updateHistoryNav();
}

function updateHistoryNav() {
    const navEl = document.getElementById('history-nav');
    if (!navEl) return;

    const historyCount = AppState.episodeHistory.length;
    const isLive = AppState.currentHistoryIndex === -1;
    const currentIndex = isLive ? historyCount : AppState.currentHistoryIndex;

    // Update counter
    const counterEl = navEl.querySelector('.history-counter');
    if (counterEl) {
        if (historyCount === 0 && isLive) {
            counterEl.textContent = 'Live';
        } else {
            counterEl.textContent = isLive
                ? `Live (${historyCount + 1}/${historyCount + 1})`
                : `${currentIndex + 1}/${historyCount + 1}`;
        }
    }

    // Update button states
    const prevBtn = navEl.querySelector('.history-prev');
    const nextBtn = navEl.querySelector('.history-next');
    const liveBtn = navEl.querySelector('.history-live');

    if (prevBtn) prevBtn.disabled = currentIndex <= 0;
    if (nextBtn) nextBtn.disabled = isLive;
    if (liveBtn) liveBtn.classList.toggle('btn-primary', !isLive);

    // Show/hide nav based on history
    navEl.classList.toggle('hidden', historyCount === 0 && isLive);
}

function navigateHistory(delta) {
    const historyCount = AppState.episodeHistory.length;
    const isLive = AppState.currentHistoryIndex === -1;
    const currentIndex = isLive ? historyCount : AppState.currentHistoryIndex;

    let newIndex = currentIndex + delta;
    if (newIndex < 0) newIndex = 0;
    if (newIndex > historyCount) newIndex = historyCount;

    // Set state
    if (newIndex === historyCount) {
        AppState.currentHistoryIndex = -1; // Live
        showLiveTokens();
    } else {
        AppState.currentHistoryIndex = newIndex;
        showHistoryEntry(newIndex);
    }

    updateHistoryNav();
}

function goToLive() {
    AppState.currentHistoryIndex = -1;
    showLiveTokens();
    updateHistoryNav();
}

function showHistoryEntry(index) {
    const entry = AppState.episodeHistory[index];
    if (!entry) return;

    // Update episode info display
    const episodeInfo = document.getElementById('current-episode-info');
    const conditionLabel = document.getElementById('current-condition-label');

    if (episodeInfo) {
        episodeInfo.innerHTML = `
            <div class="font-medium">${entry.episode_id} <span class="badge badge-ghost badge-sm">history</span></div>
            <div class="text-sm text-gray-400">${formatBehavior(entry.behavior)}</div>
        `;
    }

    if (conditionLabel) {
        conditionLabel.textContent = entry.condition;
        conditionLabel.className = `font-bold ml-2 text-${getConditionColor(entry.condition)}`;
    }

    // Restore conversation turns
    const turnsEl = document.getElementById('episode-turns-display');
    if (turnsEl && entry.turnsHtml) {
        turnsEl.innerHTML = entry.turnsHtml;
    }

    // Restore safety alerts
    const alertsEl = document.getElementById('safety-alerts-list');
    if (alertsEl) {
        if (entry.alertsHtml) {
            alertsEl.innerHTML = entry.alertsHtml;
        } else if (entry.safetyAlerts && entry.safetyAlerts.length > 0) {
            // Rebuild from structured data
            alertsEl.innerHTML = entry.safetyAlerts.slice(0, 5).map(alert => `
                <div class="alert-item">
                    <div class="font-medium">"${escapeHtml(alert.token)}"</div>
                    <div class="text-xs text-gray-400 mt-1">${
                        alert.concepts.slice(0, 2).map(([c, s]) => `${c}: ${(s * 100).toFixed(0)}%`).join(', ')
                    }</div>
                </div>
            `).join('');
        } else {
            alertsEl.innerHTML = '<p class="text-gray-500">No alerts</p>';
        }
    }

    // Restore steering events
    const steeringEl = document.getElementById('active-steerings-list');
    if (steeringEl) {
        if (entry.steeringHtml) {
            steeringEl.innerHTML = entry.steeringHtml;
        } else if (entry.steeringEvents && entry.steeringEvents.length > 0) {
            // Rebuild from structured data
            steeringEl.innerHTML = entry.steeringEvents.slice(0, 5).map(event => `
                <div class="steering-item">
                    <span class="text-yellow-400">⚡</span>
                    <span class="flex-1 ml-2">"${escapeHtml(event.token)}"</span>
                </div>
            `).join('');
        } else {
            steeringEl.innerHTML = '<p class="text-gray-500">No active steerings</p>';
        }
    }

    // Render in chat-style layout
    const tokensEl = document.getElementById('live-tokens');
    const liveConvo = document.getElementById('live-conversation');
    const liveResponse = document.getElementById('live-response');
    const livePlaceholder = document.getElementById('live-placeholder');

    tokensEl.classList.add('viewing-history');

    // Show conversation turns
    if (liveConvo) {
        liveConvo.innerHTML = entry.turnsHtml || '';
    }

    // Show tokens in response area
    if (liveResponse) {
        liveResponse.innerHTML = '';
        entry.tokens.forEach((token, i) => {
            appendTokenDirect(liveResponse, token, entry.metadata[i] || {});
        });
    }

    // Hide placeholder
    if (livePlaceholder) livePlaceholder.classList.add('hidden');

    // Update token count
    document.getElementById('token-count').textContent = `${entry.tokens.length} tokens`;
}

function showLiveTokens() {
    const tokensEl = document.getElementById('live-tokens');
    const liveConvo = document.getElementById('live-conversation');
    const liveResponse = document.getElementById('live-response');
    const livePlaceholder = document.getElementById('live-placeholder');

    tokensEl.classList.remove('viewing-history');

    // Restore conversation turns
    if (liveConvo && AppState.liveTurnsHtml) {
        liveConvo.innerHTML = AppState.liveTurnsHtml;
    }

    // Restore tokens in response area
    if (liveResponse) {
        liveResponse.innerHTML = '';
        if (AppState.liveTokens.length === 0) {
            liveResponse.innerHTML = '<span class="text-base-content/50 animate-pulse">Generating...</span>';
        } else {
            AppState.liveTokens.forEach((token, i) => {
                appendTokenDirect(liveResponse, token, AppState.liveMetadata[i] || {});
            });
        }
    }

    // Hide placeholder
    if (livePlaceholder) livePlaceholder.classList.add('hidden');

    // Restore live turns in side panel
    const turnsEl = document.getElementById('episode-turns-display');
    if (turnsEl && AppState.liveTurnsHtml) {
        turnsEl.innerHTML = AppState.liveTurnsHtml;
    }

    // Rebuild live safety alerts and steering from metadata
    const alertsEl = document.getElementById('safety-alerts-list');
    const steeringEl = document.getElementById('active-steerings-list');

    if (alertsEl) {
        const alerts = [];
        AppState.liveMetadata.forEach((meta, idx) => {
            if (meta.safety_intensity > 0.3) {
                alerts.push({ token: AppState.liveTokens[idx], concepts: meta.top_concepts || [] });
            }
        });
        if (alerts.length > 0) {
            alertsEl.innerHTML = alerts.slice(-5).reverse().map(alert => `
                <div class="alert-item">
                    <div class="font-medium">"${escapeHtml(alert.token)}"</div>
                    <div class="text-xs text-gray-400 mt-1">${
                        alert.concepts.slice(0, 2).map(([c, s]) => `${c}: ${(s * 100).toFixed(0)}%`).join(', ')
                    }</div>
                </div>
            `).join('');
        } else {
            alertsEl.innerHTML = '<p class="text-gray-500">No alerts</p>';
        }
    }

    if (steeringEl) {
        const events = [];
        AppState.liveMetadata.forEach((meta, idx) => {
            if (meta.steering_active) {
                events.push({ token: AppState.liveTokens[idx] });
            }
        });
        if (events.length > 0) {
            steeringEl.innerHTML = events.slice(-5).reverse().map(event => `
                <div class="steering-item">
                    <span class="text-yellow-400">⚡</span>
                    <span class="flex-1 ml-2">"${escapeHtml(event.token)}"</span>
                </div>
            `).join('');
        } else {
            steeringEl.innerHTML = '<p class="text-gray-500">No active steerings</p>';
        }
    }

    // Update token count
    document.getElementById('token-count').textContent = `${AppState.liveTokens.length} tokens`;
}

function appendTokenDirect(container, token, metadata) {
    const span = document.createElement('span');
    span.className = 'hatcat-token';
    span.textContent = token;

    // Use server-provided display_color (from HatCat significance scoring)
    // Falls back to client-side computation if not provided
    let bgColor;
    if (metadata.display_color) {
        bgColor = metadata.display_color;
    } else if (metadata.color) {
        bgColor = metadata.color;
    } else {
        // Fallback: compute color client-side
        const safetyIntensity = metadata.safety_intensity || 0;
        const significance = metadata.significance || 0.5;
        const isFiller = metadata.is_filler || false;
        bgColor = computeTokenColor(safetyIntensity, significance, isFiller);
    }

    span.style.backgroundColor = bgColor;
    span.style.color = getContrastColor(bgColor);

    if (metadata.safety_intensity > 0.3) {
        span.classList.add('safety');
    }

    if (metadata.steering_active) {
        span.classList.add('steering');
        const indicator = document.createElement('span');
        indicator.className = 'steering-indicator';
        indicator.textContent = '⚡';
        span.style.position = 'relative';
        span.appendChild(indicator);
    }

    span.dataset.metadata = JSON.stringify(metadata);
    span.addEventListener('mouseenter', showTokenTooltip);
    span.addEventListener('mouseleave', hideTokenTooltip);

    container.appendChild(span);
}

// ============================================================================
// Utilities
// ============================================================================

function formatBehavior(behavior) {
    return behavior.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Compute token background color based on safety intensity and significance.
 *
 * - Safety intensity controls hue (gray→orange→red) and saturation
 * - Significance controls lightness (dark=filler, light=decision point)
 * - Result: significant safety tokens are bright red, insignificant are dark/muted
 *
 * @param {number} safetyIntensity - 0-1, how dangerous this token is
 * @param {number} significance - 0-1, how significant this token is (decision vs filler)
 * @param {boolean} isFiller - explicit filler classification
 * @returns {string} HSL color string
 */
function computeTokenColor(safetyIntensity, significance, isFiller = false) {
    // Base hue: gray (220) for safe, orange (30) for moderate, red (0) for danger
    let hue = 220;  // Gray-blue default
    if (safetyIntensity > 0.5) {
        hue = 0;  // Red for high danger
    } else if (safetyIntensity > 0.3) {
        hue = 30;  // Orange for moderate danger
    }

    // Saturation: controlled by safety intensity
    // Low safety = desaturated (gray), high safety = saturated (vivid)
    // Range: 10% (safe) to 70% (dangerous)
    const saturation = 10 + (safetyIntensity * 60);

    // Lightness: controlled by significance
    // Low significance (filler) = dark (15-25%)
    // High significance (decision) = lighter (40-55%)
    // This ensures significant safety tokens are bright/visible
    const minLightness = 15;
    const maxLightness = isFiller ? 20 : 55;
    const lightness = minLightness + (significance * (maxLightness - minLightness));

    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

function getContrastColor(color) {
    // Handle HSL colors
    if (color && color.startsWith('hsl')) {
        const match = color.match(/hsl\((\d+),\s*(\d+)%?,\s*(\d+)%?\)/);
        if (match) {
            const lightness = parseInt(match[3]);
            return lightness > 45 ? '#000000' : '#ffffff';
        }
    }

    // Handle hex colors
    if (color && color.startsWith('#')) {
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        return luminance > 0.5 ? '#000000' : '#ffffff';
    }

    return '#ffffff';
}

// ============================================================================
// Calibration Stability Metrics
// ============================================================================

async function loadStabilityMetrics() {
    /**
     * Fetch and display calibration stability metrics.
     * Called after lens pack is loaded.
     */
    try {
        const response = await fetch('/api/calibration/stability');
        const data = await response.json();

        const card = document.getElementById('stability-card');
        const badge = document.getElementById('stability-status-badge');

        if (data.status === 'not_run') {
            card.classList.remove('hidden');
            badge.textContent = 'Not Calibrated';
            badge.className = 'badge badge-warning gap-2';
            document.getElementById('stat-diagonal-topk').textContent = '-';
            document.getElementById('stat-jaccard').textContent = '-';
            document.getElementById('stat-stable').textContent = '-';
            document.getElementById('stat-unstable').textContent = '-';
            return;
        }

        card.classList.remove('hidden');
        badge.textContent = 'Calibrated';
        badge.className = 'badge badge-success gap-2';

        // Display metrics (use calibration-based metrics from training data)
        // low_crossfire_rate: % of lenses with <10% cross-fire (stable)
        // good_gap_rate: % of lenses with >0.2 signal gap (discriminative)
        const lowCfr = data.low_crossfire_rate ?? data.diagonal_in_topk_rate ?? 0;
        const goodGap = data.good_gap_rate ?? data.jaccard_mean ?? 0;

        document.getElementById('stat-diagonal-topk').textContent =
            `${(lowCfr * 100).toFixed(1)}%`;

        document.getElementById('stat-jaccard').textContent =
            `${(goodGap * 100).toFixed(1)}%`;

        document.getElementById('stat-stable').textContent = data.stable_count;
        document.getElementById('stat-unstable').textContent = data.unstable_count;

        // Show over-firing concepts if any
        if (data.over_firing && data.over_firing.length > 0) {
            const alertEl = document.getElementById('over-firing-alert');
            const listEl = document.getElementById('over-firing-list');
            listEl.textContent = data.over_firing.slice(0, 5).join(', ');
            alertEl.classList.remove('hidden');
        }

        console.log('Stability metrics loaded:', data);

    } catch (error) {
        console.error('Failed to load stability metrics:', error);
    }
}

// ============================================================================
// Initialize
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    setInterval(checkStatus, 30000);

    // Load stability metrics if lens is already loaded
    loadStabilityMetrics();
});
