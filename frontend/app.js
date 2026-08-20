/**
 * WhereItLands - Match Prediction Frontend Client
 * Flat Design System implementation with 1-Click Recruiter Presets,
 * dynamic outcome calculation, and asynchronous REST API integration.
 */

(function () {
    'use strict';

    // Base API URL configuration
    const API_BASE_URL = (window.location.protocol.startsWith('http') && !window.location.port.includes('5500'))
        ? window.location.origin
        : 'http://127.0.0.1:8000';

    // DOM Element References
    const elements = {
        // Status Indicator
        statusDot: document.getElementById('statusDot'),
        statusLabel: document.getElementById('statusLabel'),
        
        // Form & Interactive Controls
        form: document.getElementById('predictionForm'),
        homeSelect: document.getElementById('homeTeamSelect'),
        awaySelect: document.getElementById('awayTeamSelect'),
        swapBtn: document.getElementById('swapTeamsBtn'),
        neutralToggle: document.getElementById('neutralToggle'),
        dateInput: document.getElementById('dateInput'),
        iterationsInput: document.getElementById('iterationsInput'),
        iterationsValue: document.getElementById('iterationsValue'),
        predictBtn: document.getElementById('predictBtn'),
        btnText: document.querySelector('.btn-text'),
        btnSpinner: document.querySelector('.btn-spinner'),
        errorAlert: document.getElementById('errorAlert'),
        errorMessage: document.getElementById('errorMessage'),

        // Preset Chips
        presetChips: document.querySelectorAll('.preset-chip'),

        // Results View & Matchup Banner
        emptyState: document.getElementById('emptyState'),
        resultsContent: document.getElementById('resultsContent'),
        resHomeName: document.getElementById('resHomeName'),
        resAwayName: document.getElementById('resAwayName'),

        // 3-Way Win Probability Elements
        homeWinVal: document.getElementById('homeWinVal'),
        drawVal: document.getElementById('drawVal'),
        awayWinVal: document.getElementById('awayWinVal'),
        homeWinBarFill: document.getElementById('homeWinBarFill'),
        drawBarFill: document.getElementById('drawBarFill'),
        awayWinBarFill: document.getElementById('awayWinBarFill'),
        homeWinnerBadge: document.getElementById('homeWinnerBadge'),
        drawWinnerBadge: document.getElementById('drawWinnerBadge'),
        awayWinnerBadge: document.getElementById('awayWinnerBadge'),

        // Stacked Distribution Bar
        stackedHome: document.getElementById('stackedHomeSegment'),
        stackedDraw: document.getElementById('stackedDrawSegment'),
        stackedAway: document.getElementById('stackedAwaySegment'),

        // Scorelines Table Body
        scorelinesTableBody: document.getElementById('scorelinesTableBody')
    };

    /**
     * Set dynamic date constraints on the date picker.
     */
    function initDateConstraints() {
        if (elements.dateInput) {
            const today = new Date().toISOString().split('T')[0];
            elements.dateInput.max = today;
        }
    }

    /**
     * Sync iterations range slider value with display badge.
     */
    function initIterationsSlider() {
        if (elements.iterationsInput && elements.iterationsValue) {
            const updateSliderBadge = () => {
                const count = elements.iterationsInput.value;
                elements.iterationsValue.textContent = `${count} ${count === '1' ? 'Outcome' : 'Outcomes'}`;
            };
            elements.iterationsInput.addEventListener('input', updateSliderBadge);
            updateSliderBadge();
        }
    }

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
     * Toggle button loading state during API calls.
     * @param {boolean} isLoading - Whether API request is in-flight.
     */
    function setLoading(isLoading) {
        elements.predictBtn.disabled = isLoading;
        if (isLoading) {
            elements.btnText.textContent = 'Simulating Outcome Matrix...';
            elements.btnSpinner.classList.remove('hidden');
        } else {
            elements.btnText.textContent = 'Compute Match Probabilities';
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
                    elements.statusLabel.textContent = 'API Ready (RAM Cached)';
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

            // Clear loading placeholders
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

            // Set default selections if available in dataset
            if (teams.includes('Argentina')) elements.homeSelect.value = 'Argentina';
            if (teams.includes('France')) elements.awaySelect.value = 'France';

        } catch (error) {
            setError(`Initialization error: Failed to connect to API at ${API_BASE_URL}. Ensure FastAPI server is active.`);
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

        // Update card fill tracks
        elements.homeWinBarFill.style.width = formatPercent(homeWinPct);
        elements.drawBarFill.style.width = formatPercent(drawPct);
        elements.awayWinBarFill.style.width = formatPercent(awayWinPct);

        // Determine Favorite (Highest Chance)
        elements.homeWinnerBadge.classList.add('hidden');
        elements.drawWinnerBadge.classList.add('hidden');
        elements.awayWinnerBadge.classList.add('hidden');

        const maxPct = Math.max(homeWinPct, drawPct, awayWinPct);
        if (maxPct === homeWinPct) {
            elements.homeWinnerBadge.classList.remove('hidden');
        } else if (maxPct === drawPct) {
            elements.drawWinnerBadge.classList.remove('hidden');
        } else {
            elements.awayWinnerBadge.classList.remove('hidden');
        }

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
                <td class="th-rank">
                    <span class="rank-badge-flat">#${index + 1}</span>
                </td>
                <td class="th-score">
                    <span class="score-badge-flat">${homeGoals} - ${awayGoals}</span>
                </td>
                <td class="th-prob">
                    <div class="prob-cell">
                        <span class="prob-number">${scoreProb.toFixed(1)}%</span>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill" style="width: ${Math.min(scoreProb * 3.5, 100)}%;"></div>
                        </div>
                    </div>
                </td>
            `;

            elements.scorelinesTableBody.appendChild(row);
        });

        // Switch visible view from empty placeholder to populated results
        elements.emptyState.classList.add('hidden');
        elements.resultsContent.classList.remove('hidden');
    }

    /**
     * Execute prediction request against the backend.
     */
    async function executePrediction(homeTeam, awayTeam, isNeutral, iterations, dateValue) {
        if (!homeTeam || !awayTeam) {
            setError('Please select both a home team and an away team.');
            return;
        }

        if (homeTeam === awayTeam) {
            setError('Home and away selections must be different national teams.');
            return;
        }

        if (isNaN(iterations) || iterations < 1 || iterations > 6) {
            setError('Please specify a scoreline count between 1 and 6.');
            return;
        }

        if (dateValue) {
            const today = new Date().toISOString().split('T')[0];
            if (dateValue > today) {
                setError('Simulation date cannot be in the future.');
                return;
            }
            if (dateValue < '2000-01-01') {
                setError('Simulation date cannot be earlier than 2000-01-01.');
                return;
            }
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

        await executePrediction(homeTeam, awayTeam, isNeutral, iterations, dateValue);
    }

    /**
     * Initialize 1-Click Recruiter Demo Preset Chips.
     */
    function initPresets() {
        elements.presetChips.forEach(chip => {
            chip.addEventListener('click', async () => {
                const home = chip.getAttribute('data-home');
                const away = chip.getAttribute('data-away');
                const neutral = chip.getAttribute('data-neutral') === 'true';
                const date = chip.getAttribute('data-date') || '';
                const iterations = parseInt(chip.getAttribute('data-iterations') || '3', 10);

                if (elements.homeSelect) elements.homeSelect.value = home;
                if (elements.awaySelect) elements.awaySelect.value = away;
                if (elements.neutralToggle) elements.neutralToggle.checked = neutral;
                if (elements.dateInput) elements.dateInput.value = date;
                if (elements.iterationsInput) {
                    elements.iterationsInput.value = iterations;
                    if (elements.iterationsValue) {
                        elements.iterationsValue.textContent = `${iterations} Outcomes`;
                    }
                }

                setError(null);
                await executePrediction(home, away, neutral, iterations, date);
            });
        });
    }

    // Attach Event Listeners
    elements.form.addEventListener('submit', handleFormSubmit);
    elements.swapBtn.addEventListener('click', swapTeams);

    // Initial Execution
    initDateConstraints();
    initIterationsSlider();
    initPresets();
    checkHealth();
    fetchTeams();

})();
