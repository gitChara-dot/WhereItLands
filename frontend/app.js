/**
 * WhereItLands - Match Prediction Frontend Client
 * Handles asynchronous API communication, form validation, and DOM rendering.
 */

(function () {
    'use strict';

    // Base API URL configuration
    const API_BASE_URL = (window.location.protocol.startsWith('http') && !window.location.port.includes('5500'))
        ? window.location.origin
        : 'http://127.0.0.1:8000';

    // DOM Element References
    const elements = {
        // Status
        statusDot: document.getElementById('statusDot'),
        statusLabel: document.getElementById('statusLabel'),
        
        // Form & Inputs
        form: document.getElementById('predictionForm'),
        homeSelect: document.getElementById('homeTeamSelect'),
        awaySelect: document.getElementById('awayTeamSelect'),
        swapBtn: document.getElementById('swapTeamsBtn'),
        neutralToggle: document.getElementById('neutralToggle'),
        dateInput: document.getElementById('dateInput'),
        iterationsInput: document.getElementById('iterationsInput'),
        predictBtn: document.getElementById('predictBtn'),
        btnText: document.querySelector('.btn-text'),
        btnSpinner: document.querySelector('.btn-spinner'),
        errorAlert: document.getElementById('errorAlert'),
        errorMessage: document.getElementById('errorMessage'),

        // Results Container
        emptyState: document.getElementById('emptyState'),
        resultsContent: document.getElementById('resultsContent'),
        resHomeName: document.getElementById('resHomeName'),
        resAwayName: document.getElementById('resAwayName'),

        // Probability Metrics
        homeWinVal: document.getElementById('homeWinVal'),
        drawVal: document.getElementById('drawVal'),
        awayWinVal: document.getElementById('awayWinVal'),
        homeWinBarFill: document.getElementById('homeWinBarFill'),
        drawBarFill: document.getElementById('drawBarFill'),
        awayWinBarFill: document.getElementById('awayWinBarFill'),

        // Stacked Bar
        stackedHome: document.getElementById('stackedHomeSegment'),
        stackedDraw: document.getElementById('stackedDrawSegment'),
        stackedAway: document.getElementById('stackedAwaySegment'),

        // Scorelines Table
        scorelinesTableBody: document.getElementById('scorelinesTableBody')
    };

    /**
     * Display or hide error alert banner with a custom message.
     * @param {string|null} message - Error description or null to dismiss.
     */
    function setError(message) {
        if (message) {
            elements.errorMessage.textContent = message;
            elements.errorAlert.classList.remove('hidden');
        } else {
            elements.errorAlert.classList.add('hidden');
            elements.errorMessage.textContent = '';
        }
    }

    /**
     * Toggle button loading state.
     * @param {boolean} isLoading - Whether API request is in-flight.
     */
    function setLoading(isLoading) {
        elements.predictBtn.disabled = isLoading;
        if (isLoading) {
            elements.btnText.textContent = 'Calculating Odds...';
            elements.btnSpinner.classList.remove('hidden');
        } else {
            elements.btnText.textContent = 'Run Match Simulation';
            elements.btnSpinner.classList.add('hidden');
        }
    }

    /**
     * Check backend readiness status via GET /health endpoint.
     */
    async function checkHealth() {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'ready') {
                    elements.statusDot.className = 'status-dot active';
                    elements.statusLabel.textContent = 'API Connected';
                    return;
                }
            }
            elements.statusDot.className = 'status-dot error';
            elements.statusLabel.textContent = 'Models Missing';
        } catch (error) {
            elements.statusDot.className = 'status-dot error';
            elements.statusLabel.textContent = 'API Offline';
        }
    }

    /**
     * Populate team dropdown selectors from GET /teams endpoint.
     */
    async function fetchTeams() {
        try {
            const response = await fetch(`${API_BASE_URL}/teams`);
            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const data = await response.json();
            const teams = data.teams || [];

            if (teams.length === 0) {
                throw new Error('No team records available in dataset.');
            }

            // Clear loading placeholder
            elements.homeSelect.innerHTML = '<option value="" disabled selected>Select Home Team</option>';
            elements.awaySelect.innerHTML = '<option value="" disabled selected>Select Away Team</option>';

            teams.forEach(team => {
                const optHome = document.createElement('option');
                optHome.value = team;
                optHome.textContent = team;
                elements.homeSelect.appendChild(optHome);

                const optAway = document.createElement('option');
                optAway.value = team;
                optAway.textContent = team;
                elements.awaySelect.appendChild(optAway);
            });

            // Set default selections if available
            if (teams.includes('Argentina')) elements.homeSelect.value = 'Argentina';
            if (teams.includes('France')) elements.awaySelect.value = 'France';

        } catch (error) {
            setError(`Initialization error: Failed to connect to API at ${API_BASE_URL}. Ensure Uvicorn server is running.`);
        }
    }

    /**
     * Swap the currently selected home and away teams.
     */
    function swapTeams() {
        const temp = elements.homeSelect.value;
        elements.homeSelect.value = elements.awaySelect.value;
        elements.awaySelect.value = temp;
    }

    /**
     * Format a decimal probability to a rounded percentage string.
     * @param {number} decimalProb - Probability value between 0 and 1.
     * @returns {string} Formatted percentage (e.g., "54.2%").
     */
    function formatPercent(decimalProb) {
        return (decimalProb * 100).toFixed(1) + '%';
    }

    /**
     * Render calculated predictions into DOM components.
     * @param {string} homeTeam - Name of the home team.
     * @param {string} awayTeam - Name of the away team.
     * @param {Object} prediction - Prediction response object from API.
     */
    function renderResults(homeTeam, awayTeam, prediction) {
        const homeWinPct = prediction.home_win_chance;
        const drawPct = prediction.draw_chance;
        const awayWinPct = prediction.away_win_chance;

        // Set banner titles
        elements.resHomeName.textContent = homeTeam;
        elements.resAwayName.textContent = awayTeam;

        // Update cards text
        elements.homeWinVal.textContent = formatPercent(homeWinPct);
        elements.drawVal.textContent = formatPercent(drawPct);
        elements.awayWinVal.textContent = formatPercent(awayWinPct);

        // Update card tracks
        elements.homeWinBarFill.style.width = formatPercent(homeWinPct);
        elements.drawBarFill.style.width = formatPercent(drawPct);
        elements.awayWinBarFill.style.width = formatPercent(awayWinPct);

        // Update stacked distribution bar
        elements.stackedHome.style.width = formatPercent(homeWinPct);
        elements.stackedDraw.style.width = formatPercent(drawPct);
        elements.stackedAway.style.width = formatPercent(awayWinPct);

        // Render Top Scorelines Table
        elements.scorelinesTableBody.innerHTML = '';
        const topResults = prediction.top_results || [];

        topResults.forEach((item, index) => {
            const homeGoals = item[0];
            const awayGoals = item[1];
            const scoreProb = item[2]; // Percentage (0-100)

            const row = document.createElement('tr');

            row.innerHTML = `
                <td class="col-rank">
                    <span class="rank-badge">#${index + 1}</span>
                </td>
                <td class="col-score">
                    <span class="score-badge">${homeGoals} - ${awayGoals}</span>
                </td>
                <td class="col-prob">
                    <div class="prob-cell-wrapper">
                        <span class="prob-percentage-text">${scoreProb.toFixed(1)}%</span>
                        <div class="prob-inline-bar-track">
                            <div class="prob-inline-bar-fill" style="width: ${Math.min(scoreProb * 3, 100)}%;"></div>
                        </div>
                    </div>
                </td>
            `;

            elements.scorelinesTableBody.appendChild(row);
        });

        // Switch visible view
        elements.emptyState.classList.add('hidden');
        elements.resultsContent.classList.remove('hidden');
    }

    /**
     * Handle match prediction form submission.
     * @param {Event} event - Submit event.
     */
    async function handleFormSubmit(event) {
        event.preventDefault();
        setError(null);

        const homeTeam = elements.homeSelect.value;
        const awayTeam = elements.awaySelect.value;
        const isNeutral = elements.neutralToggle.checked;
        const iterations = parseInt(elements.iterationsInput.value, 10);
        const dateValue = elements.dateInput ? elements.dateInput.value.trim() : '';

        if (!homeTeam || !awayTeam) {
            setError('Please select both a home team and an away team.');
            return;
        }

        if (homeTeam === awayTeam) {
            setError('Home and away selections must be different national teams.');
            return;
        }

        if (isNaN(iterations) || iterations < 1 || iterations > 6) {
            setError('Please specify a valid scoreline count between 1 and 6.');
            return;
        }

        setLoading(true);

        const payload = {
            home_team: homeTeam,
            away_team: awayTeam,
            neutral: isNeutral,
            iterations: iterations
        };

        if (dateValue) {
            payload.date = dateValue;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/prediction`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Server returned status ${response.status}`);
            }

            const predictionData = await response.json();
            renderResults(homeTeam, awayTeam, predictionData);

        } catch (error) {
            setError(`Prediction failed: ${error.message}`);
        } finally {
            setLoading(false);
        }
    }

    // Attach Event Listeners
    elements.form.addEventListener('submit', handleFormSubmit);
    elements.swapBtn.addEventListener('click', swapTeams);

    // Initial Execution
    checkHealth();
    fetchTeams();

})();
