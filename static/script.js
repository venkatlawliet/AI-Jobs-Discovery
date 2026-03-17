document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchInput');
    const chatHistory = document.getElementById('chatHistory');
    const intentContainer = document.getElementById('intentContainer');
    const intentTags = document.getElementById('intentTags');
    const resultsContainer = document.getElementById('resultsContainer');
    const loader = document.getElementById('loader');
    const metricsDisplay = document.getElementById('metricsDisplay');
    const newChatBtn = document.getElementById('newChatBtn');

    let currentHistory = [];

    // Initialize Feather Icons
    feather.replace();

    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = searchInput.value.trim();
        if (!query) return;

        // Clear input and show loading state
        searchInput.value = '';
        addChatBubble(query, 'user');

        loader.classList.remove('hidden');
        resultsContainer.innerHTML = '';
        intentContainer.classList.add('hidden');

        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    history: currentHistory
                })
            });

            const data = await response.json();

            if (data.error) {
                alert(data.error);
                return;
            }

            // Update State
            currentHistory = data.history;
            if (data.metrics) {
                tokenCount.textContent = data.metrics.total_tokens.toLocaleString();
                costCount.textContent = `$${data.metrics.estimated_cost.toFixed(4)}`;
            }

            // Render Intent Tags
            renderIntentTags(data.intent);

            // Render Results
            renderResults(data.results);

            // Add AI response to chatter
            addChatBubble(`Found ${data.results.length} relevant roles.`, 'ai');

        } catch (err) {
            console.error(err);
            addChatBubble("Sorry, an error occurred while searching.", 'ai');
        } finally {
            loader.classList.add('hidden');
        }
    });

    newChatBtn.addEventListener('click', () => {
        currentHistory = [];
        chatHistory.innerHTML = '<div class="empty-state-text">Start your semantic search below!</div>';
        resultsContainer.innerHTML = '';
        intentContainer.classList.add('hidden');
        tokenCount.textContent = '0';
        costCount.textContent = '$0.0000';
    });

    function addChatBubble(text, sender) {
        // Remove empty state text if present
        const emptyText = chatHistory.querySelector('.empty-state-text');
        if (emptyText) emptyText.remove();

        const bubble = document.createElement('div');
        bubble.classList.add('chat-bubble', `chat-${sender}`);
        bubble.textContent = text;
        chatHistory.appendChild(bubble);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function renderIntentTags(intent) {
        intentTags.innerHTML = '';

        if (intent.explicit_query) {
            intentTags.innerHTML += `<span class="tag tag-explicit"><i data-feather="target"></i> ${intent.explicit_query}</span>`;
        }
        if (intent.inferred_query) {
            const inferred = intent.inferred_query.length > 60 ? intent.inferred_query.substring(0, 60) + '...' : intent.inferred_query;
            intentTags.innerHTML += `<span class="tag tag-inferred"><i data-feather="cpu"></i> ${inferred}</span>`;
        }
        if (intent.company_query) {
            intentTags.innerHTML += `<span class="tag tag-company"><i data-feather="briefcase"></i> ${intent.company_query}</span>`;
        }

        // Handle structured filters
        if (intent.filters) {
            const f = intent.filters;
            if (f.location && f.location !== 'null') {
                intentTags.innerHTML += `<span class="tag tag-filter"><i data-feather="map-pin"></i> ${f.location}</span>`;
            }
            if (f.workplace_type && f.workplace_type !== 'null') {
                intentTags.innerHTML += `<span class="tag tag-filter"><i data-feather="wifi"></i> ${f.workplace_type}</span>`;
            }
            if (f.company_names && f.company_names.length > 0) {
                intentTags.innerHTML += `<span class="tag tag-filter"><i data-feather="filter"></i> Companies: ${f.company_names.join(', ')}</span>`;
            }
            if (f.seniority && f.seniority !== 'null') {
                intentTags.innerHTML += `<span class="tag tag-filter"><i data-feather="award"></i> ${f.seniority}</span>`;
            }
            if (f.is_non_profit === true) {
                intentTags.innerHTML += `<span class="tag tag-filter"><i data-feather="heart"></i> Non-Profit</span>`;
            }
        }

        if (intentTags.innerHTML !== '') {
            intentContainer.classList.remove('hidden');
            feather.replace();
        }
    }

    function formatSalary(min, max) {
        if (min && max) return `$${Number(min).toLocaleString()} - $${Number(max).toLocaleString()}`;
        if (min) return `$${Number(min).toLocaleString()}+`;
        if (max) return `Up to $${Number(max).toLocaleString()}`;
        return 'Not listed';
    }

    function renderResults(results) {
        resultsContainer.innerHTML = '';

        if (!results || results.length === 0) {
            resultsContainer.innerHTML = `
                <div style="text-align:center; padding: 40px; color: var(--text-muted);">
                    <i data-feather="frown" style="width: 48px; height: 48px; margin-bottom:16px; opacity:0.5;"></i>
                    <h2>No matching jobs found</h2>
                    <p>Try broadening your search or starting a new chat.</p>
                </div>
            `;
            feather.replace();
            return;
        }

        results.forEach((job, index) => {
            const card = document.createElement('div');
            card.classList.add('job-card');
            card.style.animationDelay = `${index * 0.1}s`;

            const score = job.score ? job.score.toFixed(3) : "N/A";
            const salary = formatSalary(job.salary_min, job.salary_max);
            const seniority = job.seniority && job.seniority !== 'Unknown' ? job.seniority : '';

            card.innerHTML = `
                <div class="job-header">
                    <h3 class="job-title">${job.title}</h3>
                    <span class="relevance-badge">Rel: ${score}</span>
                </div>
                <div class="company-meta">
                    <div class="meta-item"><i data-feather="briefcase"></i> ${job.company}</div>
                    <div class="meta-item"><i data-feather="map-pin"></i> ${job.location}</div>
                    <div class="meta-item"><i data-feather="monitor"></i> ${job.workplace_type}</div>
                </div>
                <div class="company-meta" style="margin-bottom: 16px;">
                    ${seniority ? `<div class="meta-item"><i data-feather="award"></i> ${seniority}</div>` : ''}
                    <div class="meta-item"><i data-feather="dollar-sign"></i> ${salary}</div>
                </div>
                <a href="${job.apply_url || '#'}" target="_blank" class="apply-btn">
                    Apply Here <i data-feather="external-link" style="width: 16px; height: 16px;"></i>
                </a>
            `;
            resultsContainer.appendChild(card);
        });

        feather.replace();
    }

    // Theme Toggle
    const themeToggleBtn = document.getElementById('themeToggle');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const isLight = document.body.classList.toggle('light-theme');
            themeToggleBtn.innerHTML = isLight ? '<i data-feather="moon"></i>' : '<i data-feather="sun"></i>';
            feather.replace();
        });
    }
});
