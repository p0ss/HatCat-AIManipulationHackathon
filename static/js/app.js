/**
 * AI Manipulation Detection Dashboard
 * Main JavaScript for the hackathon demo
 */

// ============================================================================
// Tab Navigation
// ============================================================================

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Update button states
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update content visibility
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');

        // Load tab-specific data
        if (btn.dataset.tab === 'results') {
            loadResultsList();
        } else if (btn.dataset.tab === 'compliance') {
            loadEUMapping();
            loadAuditLogsList();
        }
    });
});

// ============================================================================
// Setup Tab
// ============================================================================

async function checkStatus() {
    try {
        const response = await fetch('/api/setup/status');
        const data = await response.json();

        updateStatusIndicator('hatcat', data.hatcat_linked);
        updateStatusIndicator('model', data.model_ready);
        updateStatusIndicator('lens', data.lens_ready);
        updateStatusIndicator('gpu', data.gpu_available);

        // HatCat status with install option
        const hatcatInstallRow = document.getElementById('hatcat-install-row');
        const hatcatPathText = document.getElementById('hatcat-path-text');

        if (data.hatcat_linked) {
            document.getElementById('status-hatcat-text').textContent = 'Installed';
            hatcatInstallRow.classList.add('hidden');
        } else if (data.hatcat_path_exists) {
            document.getElementById('status-hatcat-text').textContent = 'Not installed';
            hatcatInstallRow.classList.remove('hidden');
            hatcatPathText.textContent = `Found at: ${data.hatcat_path}`;
        } else {
            document.getElementById('status-hatcat-text').textContent = 'Not found';
            hatcatInstallRow.classList.remove('hidden');
            hatcatPathText.textContent = `Expected at: ${data.hatcat_path}`;
        }

        document.getElementById('status-model-text').textContent = data.model_ready ? 'Loaded' : 'Not loaded';
        document.getElementById('status-lens-text').textContent = data.lens_ready ? 'Ready' : 'Not loaded';
        document.getElementById('status-gpu-text').textContent = data.gpu_available ? 'Available' : 'Not available';

        // GPU info
        if (data.gpu_available) {
            document.getElementById('gpu-info').innerHTML = `
                <div class="text-white font-medium">${data.gpu_name || 'GPU'}</div>
                <div class="text-gray-400">${data.vram_gb || 0} GB VRAM</div>
            `;
        } else {
            document.getElementById('gpu-info').innerHTML = `
                <div class="text-yellow-400">No GPU detected</div>
                <div class="text-gray-400">Running on CPU will be slow</div>
            `;
        }

        // Update model button state
        const modelBtn = document.getElementById('btn-download-model');
        const modelBtnText = document.getElementById('model-btn-text');
        const modelSpinner = document.getElementById('model-spinner');
        const modelProgress = document.getElementById('model-progress-fill');
        const modelProgressText = document.getElementById('model-progress-text');

        if (data.model_ready) {
            modelBtn.disabled = true;
            modelBtnText.textContent = 'Model Loaded';
            modelSpinner.classList.add('hidden');
            modelBtn.classList.remove('btn-primary');
            modelBtn.classList.add('bg-green-600', 'hover:bg-green-700');
            modelProgress.style.width = '100%';
            modelProgressText.textContent = 'Model loaded';
        }

        // Update lens button state
        const lensBtn = document.getElementById('btn-download-lens');
        const lensBtnText = document.getElementById('lens-btn-text');
        const lensSpinner = document.getElementById('lens-spinner');
        const lensProgress = document.getElementById('lens-progress-fill');
        const lensProgressText = document.getElementById('lens-progress-text');

        if (data.lens_ready) {
            lensBtn.disabled = true;
            lensBtnText.textContent = 'Lens Pack Ready';
            lensSpinner.classList.add('hidden');
            lensBtn.classList.remove('btn-primary');
            lensBtn.classList.add('bg-green-600', 'hover:bg-green-700');
            lensProgress.style.width = '100%';
            lensProgressText.textContent = 'Lens pack ready';
        }

        // Show setup complete banner if both ready
        checkSetupComplete(data.model_ready, data.lens_ready);

    } catch (error) {
        console.error('Failed to check status:', error);
    }
}

async function installHatCat() {
    const btn = document.getElementById('btn-install-hatcat');
    btn.disabled = true;
    btn.textContent = 'Installing...';

    logToConsole('Installing HatCat package...', 'info');

    try {
        const response = await fetch('/api/setup/install-hatcat', { method: 'POST' });
        const data = await response.json();

        if (response.ok) {
            logToConsole('HatCat installed successfully!', 'success');
            btn.textContent = 'Installed';
            checkStatus();
        } else {
            logToConsole('Failed to install HatCat: ' + data.detail, 'error');
            btn.textContent = 'Retry Install';
            btn.disabled = false;
        }
    } catch (error) {
        logToConsole('Error installing HatCat: ' + error.message, 'error');
        btn.textContent = 'Retry Install';
        btn.disabled = false;
    }
}

function updateStatusIndicator(id, isOk) {
    const indicator = document.getElementById('status-' + id);
    indicator.classList.remove('status-ok', 'status-pending', 'status-error');
    indicator.classList.add(isOk ? 'status-ok' : 'status-pending');
}

function checkSetupComplete(modelReady, lensReady) {
    const banner = document.getElementById('setup-complete-banner');
    if (modelReady && lensReady) {
        banner.classList.remove('hidden');
    } else {
        banner.classList.add('hidden');
    }
}

function goToEvaluation() {
    // Click the Evaluation tab
    document.querySelector('[data-tab="evaluation"]').click();
}

function logToConsole(message, type = 'info') {
    const console = document.getElementById('download-console');
    const line = document.createElement('p');

    const timestamp = new Date().toLocaleTimeString();
    line.innerHTML = `<span class="text-gray-500">[${timestamp}]</span> ${message}`;

    switch (type) {
        case 'error':
            line.classList.add('text-red-400');
            break;
        case 'success':
            line.classList.add('text-green-400');
            break;
        case 'warning':
            line.classList.add('text-yellow-400');
            break;
        case 'progress':
            line.classList.add('text-blue-400');
            break;
        default:
            line.classList.add('text-gray-300');
    }

    // Clear the "waiting" message on first log
    if (console.querySelector('.text-gray-500:only-child')) {
        console.innerHTML = '';
    }

    console.appendChild(line);
    console.scrollTop = console.scrollHeight;
}

async function downloadModel() {
    const btn = document.getElementById('btn-download-model');
    const spinner = document.getElementById('model-spinner');
    const btnText = document.getElementById('model-btn-text');
    const progressFill = document.getElementById('model-progress-fill');
    const progressText = document.getElementById('model-progress-text');

    btn.disabled = true;
    spinner.classList.remove('hidden');
    btnText.textContent = 'Downloading...';
    progressFill.style.width = '0%';

    logToConsole('Starting model download...', 'info');

    try {
        const response = await fetch('/api/setup/download-model', { method: 'POST' });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.type === 'progress') {
                            progressFill.style.width = data.percent + '%';
                            progressText.textContent = data.message;
                            logToConsole(`[${data.percent}%] ${data.message}`, 'progress');
                        } else if (data.type === 'status') {
                            logToConsole(data.message, 'info');
                        } else if (data.type === 'complete') {
                            progressFill.style.width = '100%';
                            progressText.textContent = 'Model loaded successfully!';
                            btnText.textContent = 'Model Loaded';
                            spinner.classList.add('hidden');
                            btn.classList.remove('btn-primary');
                            btn.classList.add('bg-green-600', 'hover:bg-green-700');
                            logToConsole('Model loaded successfully!', 'success');
                            checkStatus();
                        } else if (data.type === 'error') {
                            progressText.textContent = 'Error: ' + data.message;
                            btnText.textContent = 'Retry Download';
                            spinner.classList.add('hidden');
                            btn.disabled = false;
                            logToConsole('Error: ' + data.message, 'error');
                        }
                    } catch (e) {}
                }
            }
        }
    } catch (error) {
        progressText.textContent = 'Error: ' + error.message;
        btnText.textContent = 'Retry Download';
        spinner.classList.add('hidden');
        btn.disabled = false;
        logToConsole('Error: ' + error.message, 'error');
    }
}

async function downloadLens() {
    const btn = document.getElementById('btn-download-lens');
    const spinner = document.getElementById('lens-spinner');
    const btnText = document.getElementById('lens-btn-text');
    const progressFill = document.getElementById('lens-progress-fill');
    const progressText = document.getElementById('lens-progress-text');

    btn.disabled = true;
    spinner.classList.remove('hidden');
    btnText.textContent = 'Downloading...';
    progressFill.style.width = '0%';

    logToConsole('Starting lens pack download...', 'info');

    try {
        const response = await fetch('/api/setup/download-lens', { method: 'POST' });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.type === 'progress') {
                            progressFill.style.width = data.percent + '%';
                            progressText.textContent = data.message;
                            logToConsole(`[${data.percent}%] ${data.message}`, 'progress');
                        } else if (data.type === 'status') {
                            logToConsole(data.message, 'info');
                        } else if (data.type === 'complete') {
                            progressFill.style.width = '100%';
                            progressText.textContent = 'Lens pack ready!';
                            btnText.textContent = 'Lens Pack Ready';
                            spinner.classList.add('hidden');
                            btn.classList.remove('btn-primary');
                            btn.classList.add('bg-green-600', 'hover:bg-green-700');
                            logToConsole('Lens pack ready!', 'success');
                            checkStatus();
                        } else if (data.type === 'error') {
                            progressText.textContent = 'Error: ' + data.message;
                            btnText.textContent = 'Retry Download';
                            spinner.classList.add('hidden');
                            btn.disabled = false;
                            logToConsole('Error: ' + data.message, 'error');
                        }
                    } catch (e) {}
                }
            }
        }
    } catch (error) {
        progressText.textContent = 'Error: ' + error.message;
        btnText.textContent = 'Retry Download';
        spinner.classList.add('hidden');
        btn.disabled = false;
        logToConsole('Error: ' + error.message, 'error');
    }
}

// ============================================================================
// Evaluation Tab
// ============================================================================

async function loadEpisodes() {
    try {
        const response = await fetch('/api/evaluation/episodes');
        const data = await response.json();

        const container = document.getElementById('episode-list');

        if (!data.episodes || data.episodes.length === 0) {
            container.innerHTML = '<p class="text-gray-500">No episodes found</p>';
            return;
        }

        container.innerHTML = data.episodes.map(ep => `
            <label class="flex items-start gap-3 cursor-pointer p-2 rounded hover:bg-gray-700">
                <input type="checkbox" class="episode-checkbox mt-1" value="${ep.id}" checked>
                <div>
                    <div class="font-medium">${ep.id}</div>
                    <div class="text-sm text-gray-400">${ep.behavior.replace(/_/g, ' ')}</div>
                </div>
            </label>
        `).join('');
    } catch (error) {
        document.getElementById('episode-list').innerHTML = '<p class="text-red-400">Failed to load episodes</p>';
    }
}

function selectAllEpisodes() {
    document.querySelectorAll('.episode-checkbox').forEach(cb => cb.checked = true);
}

async function runEvaluation() {
    const episodes = Array.from(document.querySelectorAll('.episode-checkbox:checked')).map(cb => cb.value);
    const conditions = [];
    if (document.getElementById('cond-A').checked) conditions.push('A');
    if (document.getElementById('cond-B').checked) conditions.push('B');
    if (document.getElementById('cond-C').checked) conditions.push('C');

    if (episodes.length === 0) {
        alert('Please select at least one episode');
        return;
    }

    if (conditions.length === 0) {
        alert('Please select at least one condition');
        return;
    }

    const btnRun = document.getElementById('btn-run');
    const btnAbort = document.getElementById('btn-abort');
    const progressFill = document.getElementById('eval-progress-fill');
    const progressText = document.getElementById('eval-progress-text');
    const liveProgress = document.getElementById('live-progress');

    btnRun.classList.add('hidden');
    btnAbort.classList.remove('hidden');
    progressFill.style.width = '0%';
    liveProgress.innerHTML = '';

    try {
        const response = await fetch('/api/evaluation/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ episode_ids: episodes, conditions: conditions })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        handleEvalEvent(data, progressFill, progressText, liveProgress);
                    } catch (e) {}
                }
            }
        }
    } catch (error) {
        addLogLine(liveProgress, 'Error: ' + error.message, 'text-red-400');
    }

    btnRun.classList.remove('hidden');
    btnAbort.classList.add('hidden');
}

function handleEvalEvent(data, progressFill, progressText, liveProgress) {
    switch (data.type) {
        case 'start':
            addLogLine(liveProgress, `Starting evaluation: ${data.total_episodes} episodes, conditions: ${data.conditions.join(', ')}`, 'text-blue-400');
            break;

        case 'episode_start':
            addLogLine(liveProgress, `Running ${data.episode_id} [${data.condition}] - ${data.behavior.replace(/_/g, ' ')}`, 'text-gray-300');
            break;

        case 'episode_complete':
            const result = data.result;
            const color = result.manipulation_detected ? 'text-red-400' : 'text-green-400';
            const status = result.manipulation_detected ? 'MANIPULATION' : 'OK';
            addLogLine(liveProgress, `  -> ${status} (score: ${(result.peak_detection_score * 100).toFixed(0)}%, interventions: ${result.interventions})`, color);

            progressFill.style.width = (data.progress * 100) + '%';
            progressText.textContent = `${Math.round(data.progress * 100)}% complete`;
            break;

        case 'complete':
            addLogLine(liveProgress, `Evaluation complete! Run ID: ${data.run_id}`, 'text-green-400');
            progressFill.style.width = '100%';
            progressText.textContent = 'Complete!';
            loadResultsList();
            break;

        case 'error':
            addLogLine(liveProgress, `Error: ${data.message}`, 'text-red-400');
            break;

        case 'aborted':
            addLogLine(liveProgress, 'Evaluation aborted', 'text-yellow-400');
            break;
    }
}

function addLogLine(container, text, colorClass = '') {
    const line = document.createElement('p');
    line.textContent = text;
    if (colorClass) line.className = colorClass;
    container.appendChild(line);
    container.scrollTop = container.scrollHeight;
}

async function abortEvaluation() {
    await fetch('/api/evaluation/abort');
}

// ============================================================================
// Results Tab
// ============================================================================

async function loadResultsList() {
    try {
        const response = await fetch('/api/results/list');
        const data = await response.json();

        const selector = document.getElementById('run-selector');
        selector.innerHTML = '<option value="">Select run...</option>' +
            data.results.map(r => `<option value="${r.run_id}">${r.run_id} (${r.episode_count} episodes)</option>`).join('');

        if (data.results.length > 0) {
            selector.value = data.results[0].run_id;
            loadResults();
        }
    } catch (error) {
        console.error('Failed to load results list:', error);
    }
}

async function loadResults() {
    const runId = document.getElementById('run-selector').value;
    if (!runId) return;

    try {
        // Load comparison chart
        const chartResponse = await fetch(`/api/results/${runId}/chart/comparison`);
        const chartData = await chartResponse.json();
        Plotly.newPlot('comparison-chart', chartData.data, chartData.layout, { responsive: true });

        // Load intervention chart
        const intResponse = await fetch(`/api/results/${runId}/chart/interventions`);
        const intData = await intResponse.json();
        Plotly.newPlot('intervention-chart', intData.data, intData.layout, { responsive: true });

        // Load summary stats
        const resultsResponse = await fetch(`/api/results/${runId}`);
        const results = await resultsResponse.json();

        const summary = results.summary?.by_condition || {};
        document.getElementById('stat-baseline').textContent = (summary.A?.rate || 0).toFixed(0) + '%';
        document.getElementById('stat-monitor').textContent = (summary.B?.rate || 0).toFixed(0) + '%';
        document.getElementById('stat-harness').textContent = (summary.C?.rate || 0).toFixed(0) + '%';

        const baseline = summary.A?.rate || 0;
        const harness = summary.C?.rate || 0;
        const reduction = baseline > 0 ? ((baseline - harness) / baseline * 100) : 0;
        document.getElementById('stat-reduction').textContent = reduction.toFixed(0) + '%';

    } catch (error) {
        console.error('Failed to load results:', error);
    }
}

// ============================================================================
// Compliance Tab
// ============================================================================

async function loadEUMapping() {
    try {
        const response = await fetch('/api/compliance/eu-mapping');
        const data = await response.json();

        const container = document.getElementById('eu-mapping');
        let html = '';

        for (const [article, articleData] of Object.entries(data)) {
            html += `<div class="border-b border-gray-700 pb-4">
                <h3 class="font-semibold text-blue-400">${article}</h3>
                <p class="text-sm text-gray-400 mb-2">${articleData.description || ''}</p>`;

            for (const [req, reqData] of Object.entries(articleData.requirements || {})) {
                const statusClass = reqData.status === 'implemented' ? 'text-green-400' : 'text-yellow-400';
                html += `<div class="ml-4 mt-2 text-sm">
                    <div class="flex items-center gap-2">
                        <span class="${statusClass}">&#10003;</span>
                        <span class="text-gray-300">${req}</span>
                    </div>
                    <div class="ml-5 text-gray-500">${reqData.component}</div>
                </div>`;
            }

            html += '</div>';
        }

        container.innerHTML = html;
    } catch (error) {
        document.getElementById('eu-mapping').innerHTML = '<p class="text-red-400">Failed to load EU mapping</p>';
    }
}

async function loadAuditLogsList() {
    try {
        const response = await fetch('/api/compliance/audit-logs');
        const data = await response.json();

        const selector = document.getElementById('audit-log-selector');
        selector.innerHTML = '<option value="">Select audit log...</option>' +
            data.logs.map(log => `<option value="${log.filename}">${log.filename}</option>`).join('');
    } catch (error) {
        console.error('Failed to load audit logs list:', error);
    }
}

async function loadAuditLog() {
    const filename = document.getElementById('audit-log-selector').value;
    if (!filename) return;

    try {
        const response = await fetch(`/api/compliance/audit-logs/${filename}`);
        const data = await response.json();
        document.getElementById('audit-log-content').textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        document.getElementById('audit-log-content').textContent = 'Failed to load audit log';
    }
}

async function exportReport(format) {
    const runId = document.getElementById('run-selector')?.value;
    if (!runId) {
        alert('Please select a run from the Results tab first');
        return;
    }

    try {
        const response = await fetch('/api/compliance/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ run_id: runId, format: format })
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
// Initialize
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    loadEpisodes();

    // Refresh status periodically
    setInterval(checkStatus, 30000);
});
