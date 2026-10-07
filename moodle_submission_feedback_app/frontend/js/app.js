// SkillCat Submission Review Studio
// Master-Detail Linear-Style Reactive Controller

const state = {
  mentorName: 'Om Aryan',
  activities: [],
  activitiesOverview: [],
  globalSummary: {},
  currentView: 'dashboard', // 'dashboard' | 'studio'
  dashboardSearchQuery: '',
  dashboardFilter: 'all', // 'all' | 'pending' | 'completed'
  activeActivity: '',
  submissions: [],
  filteredSubmissions: [],
  selectedIndex: 0,
  activeFilter: 'pass', // 'pass', 'fail', 'unsure', 'approved'
  searchQuery: '',
  theme: localStorage.getItem('studio_theme') || 'dark',
  isSidebarCollapsed: localStorage.getItem('studio_sidebar_collapsed') === 'true',
  isEvalCollapsed: localStorage.getItem('studio_eval_collapsed') === 'true',
  currentMediaImages: [],
  currentMediaIndex: 0,
  isLoadingMedia: false,
  cardStates: {}, // `${name}_${attempt}` -> { grade, feedback, isDirty, originalGrade, originalFeedback }
};

// ============================================================================
// Initialization & Theme
// ============================================================================
function applySidebarState(collapsed) {
  state.isSidebarCollapsed = !!collapsed;
  localStorage.setItem('studio_sidebar_collapsed', state.isSidebarCollapsed ? 'true' : 'false');
  const ws = document.getElementById('studio-workspace');
  if (ws) {
    ws.classList.toggle('is-sidebar-collapsed', state.isSidebarCollapsed);
  }
  const bannerBadge = document.getElementById('banner-count-badge');
  if (bannerBadge) {
    bannerBadge.textContent = state.filteredSubmissions ? state.filteredSubmissions.length : 0;
  }
}

function applyEvalSidebarState(collapsed) {
  state.isEvalCollapsed = !!collapsed;
  localStorage.setItem('studio_eval_collapsed', state.isEvalCollapsed ? 'true' : 'false');
  const ws = document.getElementById('studio-workspace');
  if (ws) {
    ws.classList.toggle('is-eval-collapsed', state.isEvalCollapsed);
  }
}

function applyTheme(theme) {
  state.theme = theme;
  localStorage.setItem('studio_theme', theme);
  document.documentElement.setAttribute('data-theme', theme);
  const sunIcon = document.getElementById('theme-icon-sun');
  const moonIcon = document.getElementById('theme-icon-moon');
  if (sunIcon && moonIcon) {
    if (theme === 'dark') {
      sunIcon.style.display = 'block';
      moonIcon.style.display = 'none';
    } else {
      sunIcon.style.display = 'none';
      moonIcon.style.display = 'block';
    }
  }

  const dashSun = document.querySelector('.dash-theme-icon-sun');
  const dashMoon = document.querySelector('.dash-theme-icon-moon');
  if (dashSun && dashMoon) {
    if (theme === 'dark') {
      dashSun.style.display = 'block';
      dashMoon.style.display = 'none';
    } else {
      dashSun.style.display = 'none';
      dashMoon.style.display = 'block';
    }
  }

  const statsSun = document.querySelector('.stats-theme-icon-sun');
  const statsMoon = document.querySelector('.stats-theme-icon-moon');
  if (statsSun && statsMoon) {
    if (theme === 'dark') {
      statsSun.style.display = 'block';
      statsMoon.style.display = 'none';
    } else {
      statsSun.style.display = 'none';
      statsMoon.style.display = 'block';
    }
  }
}

function showToast(message, type = 'success') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = 'toast-msg';

  let iconSvg = '';
  if (type === 'success') {
    iconSvg = `<svg class="ic" style="color: var(--sc-success);" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>`;
  } else if (type === 'error') {
    iconSvg = `<svg class="ic" style="color: var(--sc-danger);" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
  } else {
    iconSvg = `<svg class="ic" style="color: var(--sc-primary);" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
  }

  toast.innerHTML = `${iconSvg}<span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.25s ease';
    setTimeout(() => toast.remove(), 250);
  }, 3200);
}

function getInitials(name) {
  if (!name) return '??';
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function formatCleanDate(str) {
  if (!str) return 'N/A';
  // Standardize formats like "Monday, September 28, 2026, 8:23 AM" or "Monday, 28 September 2026, 8:23 AM"
  const match = str.match(/([A-Za-z]+)[,\s]+(\d{1,2})(?:st|nd|rd|th)?[,\s]+(\d{4})/i) ||
                str.match(/(\d{1,2})[,\s]+([A-Za-z]+)[,\s]+(\d{4})/i);
  if (match) {
    if (isNaN(match[1])) {
      const month = match[1].slice(0, 3);
      const day = match[2];
      const year = match[3];
      return `${month} ${day}, ${year}`;
    } else {
      const day = match[1];
      const month = match[2].slice(0, 3);
      const year = match[3];
      return `${month} ${day}, ${year}`;
    }
  }
  const d = new Date(str);
  if (!isNaN(d.getTime())) {
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }
  return str.replace(/^[A-Za-z]+,\s*/, '').slice(0, 16).trim();
}

// ============================================================================
// Activities Dashboard Engine & View Routing
// ============================================================================
async function refreshActivitiesData(showToastFeedback = false) {
  try {
    const data = await API.getActivities();
    state.activitiesOverview = data.activities || [];
    state.globalSummary = data.summary || {};
    state.activities = (data.activities || []).map((a) => (typeof a === 'string' ? a : a.name));

    // Populate activity dropdown in studio header
    const selectEl = document.getElementById('activity-select');
    if (selectEl) {
      selectEl.innerHTML = '';
      state.activities.forEach((act) => {
        const opt = document.createElement('option');
        opt.value = act;
        opt.textContent = act;
        selectEl.appendChild(opt);
      });
    }

    if (state.currentView === 'dashboard') {
      renderDashboard();
    }

    if (showToastFeedback) {
      showToast('Activities updated from Google Sheet', 'success');
    }
  } catch (err) {
    console.error('Failed to refresh activities:', err);
    if (showToastFeedback) {
      showToast('Failed to refresh activities: ' + err.message, 'error');
    }
  }
}

function showDashboardView() {
  state.currentView = 'dashboard';
  window.history.replaceState(null, '', window.location.pathname);

  const dashView = document.getElementById('activities-dashboard-view');
  const studioView = document.getElementById('reviewer-studio-view');
  const statsView = document.getElementById('moodle-sync-stats-view');
  if (dashView) dashView.style.display = 'flex';
  if (studioView) studioView.style.display = 'none';
  if (statsView) statsView.style.display = 'none';

  renderDashboard();
}

function renderDashboard() {
  const sum = state.globalSummary || {};
  const totalActs = sum.total_activities || (state.activitiesOverview ? state.activitiesOverview.length : 0);
  const totalPending = sum.pending_count || 0;
  const totalSubs = sum.total_submissions || 0;

  // Update Hero Summary Pill (Single clean pulse, no redundant KPI boxes)
  const heroSummaryText = document.getElementById('hero-summary-text');
  const heroSummaryDot = document.getElementById('hero-summary-dot');
  if (heroSummaryText) {
    if (totalPending > 0) {
      heroSummaryText.textContent = `${totalActs} ${totalActs === 1 ? 'Activity' : 'Activities'} · ${totalPending} Submissions Pending Grading`;
      if (heroSummaryDot) heroSummaryDot.className = 'summary-dot';
    } else if (totalActs > 0) {
      heroSummaryText.textContent = `${totalActs} ${totalActs === 1 ? 'Activity' : 'Activities'} · All Submissions Graded ✓`;
      if (heroSummaryDot) heroSummaryDot.className = 'summary-dot is-all-clear';
    } else {
      heroSummaryText.textContent = 'No activities loaded';
      if (heroSummaryDot) heroSummaryDot.className = 'summary-dot is-all-clear';
    }
  }

  // Filter activities
  const q = (state.dashboardSearchQuery || '').toLowerCase().trim();
  const filter = state.dashboardFilter || 'all';

  const filtered = (state.activitiesOverview || []).filter((act) => {
    const nameMatch = !q || (act.name || '').toLowerCase().includes(q);
    if (!nameMatch) return false;

    if (filter === 'pending') {
      return (act.pending_count || 0) > 0;
    }
    if (filter === 'completed') {
      return (act.pending_count || 0) === 0;
    }
    return true;
  });

  const grid = document.getElementById('activities-cards-grid');
  if (!grid) return;

  if (filtered.length === 0) {
    grid.innerHTML = `
      <div style="grid-column: 1 / -1; padding: 48px; text-align: center; color: var(--sc-text-muted); background: var(--sc-surface-2); border-radius: var(--sc-radius-md); border: 1px dashed var(--sc-border-subtle);">
        <p style="font-size: 15px; font-weight: 500;">No activities matching "${escapeHtml(state.dashboardSearchQuery || filter)}" found.</p>
        <button class="btn btn-secondary btn-sm" onclick="state.dashboardSearchQuery=''; state.dashboardFilter='all'; renderDashboard();" style="margin-top: 12px;">Clear Filters</button>
      </div>
    `;
    return;
  }

  grid.innerHTML = filtered.map((act) => {
    const name = act.name || 'Activity';
    const total = act.total_submissions || 0;
    const pending = act.pending_grading !== undefined ? act.pending_grading : (act.pending_count || 0);
    const approved = act.approved_count || 0;
    const gradedPercent = total > 0 ? Math.round((approved / total) * 100) : 0;

    return `
      <div class="activity-card" data-activity-name="${escapeHtml(name)}">
        <div class="activity-card-header">
          <h3 class="activity-card-title">${escapeHtml(name)}</h3>
          <span class="activity-status-badge ${pending > 0 ? 'is-pending' : 'is-completed'}">
            ${pending > 0 ? '<span class="status-dot"></span>Pending Grading' : '✓ Graded'}
          </span>
        </div>

        <div class="activity-metrics-row">
          <div class="metric-block">
            <span class="metric-label">Submissions Count</span>
            <span class="metric-value">${total}</span>
          </div>
          <div class="metric-block is-pending-metric">
            <span class="metric-label">Pending Grading</span>
            <span class="metric-value ${pending > 0 ? 'text-orange' : 'text-muted'}">${pending}</span>
          </div>
          <div class="metric-block">
            <span class="metric-label">Approved</span>
            <span class="metric-value ${approved > 0 ? 'text-green' : 'text-muted'}">${approved}</span>
          </div>
        </div>

        <div class="activity-progress-wrap">
          <div class="activity-progress-info">
            <span class="prog-text">${approved} of ${total} submissions graded</span>
            <span class="prog-percent">${gradedPercent}%</span>
          </div>
          <div class="activity-progress-track">
            <div class="activity-progress-bar" style="width: ${gradedPercent}%;"></div>
          </div>
        </div>

        <div class="activity-card-footer">
          <button class="btn ${pending > 0 ? 'btn-primary' : 'btn-secondary'} btn-md open-studio-btn" onclick="openReviewStudioForActivity('${escapeHtml(name)}')">
            <span>${pending > 0 ? 'Review Submissions' : 'View Submissions'}</span>
            <svg class="ic" viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"/></svg>
          </button>
          <a href="${API.getExportUrl(name)}" class="btn btn-ghost btn-sm" title="Export CSV" download>
            <svg class="ic" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            <span>CSV</span>
          </a>
        </div>
      </div>
    `;
  }).join('');
}

async function openReviewStudioForActivity(activityName) {
  state.currentView = 'studio';
  window.location.hash = `activity=${encodeURIComponent(activityName)}`;

  const dashView = document.getElementById('activities-dashboard-view');
  const studioView = document.getElementById('reviewer-studio-view');
  const statsView = document.getElementById('moodle-sync-stats-view');
  if (dashView) dashView.style.display = 'none';
  if (studioView) studioView.style.display = 'flex';
  if (statsView) statsView.style.display = 'none';

  await selectActivity(activityName);
}

function showSyncStatsView() {
  state.currentView = 'stats';
  window.location.hash = 'stats';

  const dashView = document.getElementById('activities-dashboard-view');
  const studioView = document.getElementById('reviewer-studio-view');
  const statsView = document.getElementById('moodle-sync-stats-view');
  if (dashView) dashView.style.display = 'none';
  if (studioView) studioView.style.display = 'none';
  if (statsView) statsView.style.display = 'flex';

  renderSyncStats();
}

function renderSyncStats() {
  const sum = state.globalSummary || {};
  const totalSubs = sum.total_submissions || 0;
  const totalActs = sum.total_activities || (state.activitiesOverview ? state.activitiesOverview.length : 0);
  const approved = sum.approved_count || 0;
  const pending = sum.pending_count || 0;
  const overridden = sum.overridden_count || 0;
  const pushed = sum.pushed_count || 0;
  const passCount = sum.pass_count || 0;
  const failCount = sum.fail_count || 0;
  const passRate = sum.overall_pass_rate !== undefined ? sum.overall_pass_rate : (totalSubs > 0 ? ((passCount / totalSubs) * 100).toFixed(1) : 0.0);

  // Ready count for Gradebook Push
  const readyCountEl = document.getElementById('stats-push-ready-count');
  if (readyCountEl) {
    readyCountEl.textContent = `${approved} Approved Ready`;
  }

  // 6 KPI cards
  const totalSubsEl = document.getElementById('stats-total-subs');
  if (totalSubsEl) totalSubsEl.textContent = totalSubs.toLocaleString();

  const totalActsEl = document.getElementById('stats-total-acts');
  if (totalActsEl) totalActsEl.textContent = `Across ${totalActs} ${totalActs === 1 ? 'activity' : 'activities'}`;

  const approvedSubsEl = document.getElementById('stats-approved-subs');
  if (approvedSubsEl) approvedSubsEl.textContent = approved.toLocaleString();

  const approvedPctEl = document.getElementById('stats-approved-pct');
  if (approvedPctEl) {
    const pct = totalSubs > 0 ? Math.round((approved / totalSubs) * 100) : 0;
    approvedPctEl.textContent = `${pct}% completion`;
  }

  const pendingSubsEl = document.getElementById('stats-pending-subs');
  if (pendingSubsEl) pendingSubsEl.textContent = pending.toLocaleString();

  const pendingPctEl = document.getElementById('stats-pending-pct');
  if (pendingPctEl) {
    pendingPctEl.textContent = pending > 0 ? 'Waiting on mentor' : 'All caught up';
  }

  const passRateEl = document.getElementById('stats-pass-rate');
  if (passRateEl) passRateEl.textContent = `${passRate}%`;

  const passRatioEl = document.getElementById('stats-pass-ratio');
  if (passRatioEl) passRatioEl.textContent = `${passCount} pass / ${failCount} fail`;

  const overriddenSubsEl = document.getElementById('stats-overridden-subs');
  if (overriddenSubsEl) overriddenSubsEl.textContent = overridden.toLocaleString();

  const agreementRateEl = document.getElementById('stats-agreement-rate');
  if (agreementRateEl) {
    const totalReviewed = approved + overridden;
    const agreePct = totalReviewed > 0 ? (100 - Math.round((overridden / totalReviewed) * 100)) : 100;
    agreementRateEl.textContent = `AI Agreement: ${agreePct}%`;
  }

  const pushedSubsEl = document.getElementById('stats-pushed-subs');
  if (pushedSubsEl) pushedSubsEl.textContent = pushed.toLocaleString();

  // Table header count badge
  const countBadge = document.getElementById('stats-table-count-badge');
  if (countBadge) {
    countBadge.textContent = `${totalActs} ${totalActs === 1 ? 'Activity' : 'Activities'}`;
  }

  // Breakdown table body
  const tbody = document.getElementById('stats-activities-tbody');
  if (!tbody) return;

  const acts = state.activitiesOverview || [];
  if (acts.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 32px; color: var(--sc-text-muted);">No activity records found.</td></tr>`;
    return;
  }

  tbody.innerHTML = acts.map(act => {
    const actName = act.name || 'Activity';
    const reviewed = (act.approved_count || 0) + (act.overridden_count || 0);
    const passRt = act.pass_rate !== undefined ? `${act.pass_rate}%` : '0.0%';
    const lastMod = act.last_modified ? escapeHtml(act.last_modified) : '—';

    return `
      <tr>
        <td class="stats-act-name-cell">
          <div>${escapeHtml(actName)}</div>
        </td>
        <td class="stats-num-mono">${act.total_submissions || 0}</td>
        <td class="stats-num-mono" style="color: var(--sc-success); font-weight: 700;">${passRt}</td>
        <td class="stats-num-mono">${reviewed}</td>
        <td class="stats-num-mono" style="${(act.pending_count || 0) > 0 ? 'color: var(--sc-primary); font-weight: 700;' : 'color: var(--sc-text-muted);'}">
          ${act.pending_count || 0}
        </td>
        <td>
          <div class="stats-pills-breakdown">
            <span class="pill-count is-pass" title="Pass count">${act.pass_count || 0} Pass</span>
            <span class="pill-count is-fail" title="Fail count">${act.fail_count || 0} Fail</span>
            <span class="pill-count is-unsure" title="Unsure count">${act.unsure_count || 0} Unsure</span>
          </div>
        </td>
        <td style="font-size: 11.5px; color: var(--sc-text-muted); font-family: var(--sc-font-mono);">${lastMod}</td>
        <td style="text-align: right;">
          <button class="btn-stats-review" onclick="openReviewStudioForActivity('${escapeHtml(actName)}')" title="Open Activity in Review Studio">
            <span>Review</span>
            <svg class="ic" viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"/></svg>
          </button>
        </td>
      </tr>
    `;
  }).join('');
}

// Expose routing helpers globally for HTML onclick handlers
window.showDashboardView = showDashboardView;
window.openReviewStudioForActivity = openReviewStudioForActivity;
window.showSyncStatsView = showSyncStatsView;

async function initApp() {
  applyTheme(state.theme);
  applySidebarState(state.isSidebarCollapsed);
  applyEvalSidebarState(state.isEvalCollapsed);

  try {
    const mentorInfo = await API.getMentorInfo();
    if (mentorInfo.mentor_name) {
      state.mentorName = mentorInfo.mentor_name;

      // Studio header profile
      const nameDisp = document.getElementById('mentor-name-display');
      const avatarDisp = document.getElementById('user-avatar');
      const sheetBtn = document.getElementById('sheet-link-btn');
      if (nameDisp) nameDisp.textContent = mentorInfo.mentor_name;
      if (avatarDisp) avatarDisp.textContent = getInitials(mentorInfo.mentor_name);
      if (sheetBtn) sheetBtn.href = mentorInfo.sheet_url;

      // Dashboard header profile
      const dashNameDisp = document.getElementById('dash-mentor-name-display');
      const dashAvatarDisp = document.getElementById('dash-user-avatar');
      const dashSheetBtn = document.getElementById('dash-sheet-link-btn');
      if (dashNameDisp) dashNameDisp.textContent = mentorInfo.mentor_name;
      if (dashAvatarDisp) dashAvatarDisp.textContent = getInitials(mentorInfo.mentor_name);
      if (dashSheetBtn) dashSheetBtn.href = mentorInfo.sheet_url;

      // Stats header profile
      const statsNameDisp = document.getElementById('stats-mentor-name-display');
      const statsAvatarDisp = document.getElementById('stats-user-avatar');
      const statsSheetBtn = document.getElementById('stats-sheet-link-btn');
      if (statsNameDisp) statsNameDisp.textContent = mentorInfo.mentor_name;
      if (statsAvatarDisp) statsAvatarDisp.textContent = getInitials(mentorInfo.mentor_name);
      if (statsSheetBtn) statsSheetBtn.href = mentorInfo.sheet_url;
    }

    await refreshActivitiesData(false);

    // View routing: Check if a specific activity or stats page is targeted in hash or query param
    const hash = window.location.hash || '';
    const params = new URLSearchParams(window.location.search);
    const targetActivity = params.get('activity') || (hash.startsWith('#activity=') ? decodeURIComponent(hash.replace('#activity=', '')) : '');

    if (hash === '#stats' || hash === '#moodle-sync') {
      showSyncStatsView();
    } else if (targetActivity && state.activities.includes(targetActivity)) {
      await openReviewStudioForActivity(targetActivity);
    } else {
      // Default: Land reviewer on Activities Dashboard
      showDashboardView();
    }
  } catch (err) {
    console.error('Init error:', err);
    showToast('Failed to load studio data: ' + err.message, 'error');
  }
}

// ============================================================================
// Activity Loading & State Sync
// ============================================================================
async function selectActivity(activityName) {
  state.activeActivity = activityName;
  const selectEl = document.getElementById('activity-select');
  if (selectEl && selectEl.value !== activityName) selectEl.value = activityName;

  try {
    const data = await API.getActivityDetails(activityName);
    state.submissions = data.submissions || [];
    state.summary = data.summary || null;
    state.activityInstructions = data.instructions || '';
    state.activityEdgeCases = data.edge_cases || '';
    state.activityGuidelines = data.guidelines || '';

    // Preserve dirty states
    const newCardStates = {};
    state.submissions.forEach((sub) => {
      const key = `${sub.name}_${sub.attempt_number}`;
      const existing = state.cardStates[key];
      const defaultGrade = sub.grade.toLowerCase().includes('pass') ? 'Pass' : 'Fail';
      const defaultFeedback = sub.feedback_comment || '';

      if (existing && existing.isDirty) {
        newCardStates[key] = existing;
      } else {
        newCardStates[key] = {
          grade: defaultGrade,
          feedback: defaultFeedback,
          isDirty: false,
          originalGrade: defaultGrade,
          originalFeedback: defaultFeedback,
        };
      }
    });
    state.cardStates = newCardStates;

    // Sync default active filter if current has no items
    const counts = {
      pass: state.submissions.filter(isSubmissionPass).length,
      fail: state.submissions.filter(isSubmissionFail).length,
      unsure: state.submissions.filter((s) => !isSubmissionApproved(s) && isSubmissionUnsure(s)).length,
      approved: state.submissions.filter(isSubmissionApproved).length,
    };
    if ((counts[state.activeFilter] || 0) === 0) {
      if (counts.pass > 0) state.activeFilter = 'pass';
      else if (counts.unsure > 0) state.activeFilter = 'unsure';
      else if (counts.fail > 0) state.activeFilter = 'fail';
      else if (counts.approved > 0) state.activeFilter = 'approved';
    }

    document.querySelectorAll('.filter-pill').forEach((btn) => {
      if (btn.dataset.filter === state.activeFilter) btn.classList.add('is-active');
      else btn.classList.remove('is-active');
    });

    updateHeaderMetrics();
    applyFilterAndRender();
  } catch (err) {
    console.error('Failed to load activity:', err);
    showToast('Error loading activity: ' + err.message, 'error');
  }
}

// ============================================================================
// Submission Classifiers (Pass, Fail, Unsure, Approved)
// ============================================================================
function isSubmissionApproved(sub) {
  const r = (sub.review_status || '').toLowerCase().trim();
  return r.includes('approved') || r.includes('overridden');
}

function isSubmissionUnsure(sub) {
  const g = (sub.grade || '').toLowerCase();
  const ag = (sub.agent_grade || '').toLowerCase();
  return g.includes('unsure') || ag.includes('unsure');
}

function isSubmissionPass(sub) {
  if (isSubmissionApproved(sub)) return false;
  if (isSubmissionUnsure(sub)) return false;
  const g = (sub.grade || '').toLowerCase();
  const ag = (sub.agent_grade || '').toLowerCase();
  return g.includes('pass') || ag.includes('pass');
}

function isSubmissionFail(sub) {
  if (isSubmissionApproved(sub)) return false;
  if (isSubmissionUnsure(sub)) return false;
  const g = (sub.grade || '').toLowerCase();
  const ag = (sub.agent_grade || '').toLowerCase();
  return g.includes('fail') || ag.includes('fail') || (!g.includes('pass') && !ag.includes('pass'));
}

function updateHeaderMetrics() {
  const total = state.submissions.length;
  const passCount = state.submissions.filter(isSubmissionPass).length;
  const failCount = state.submissions.filter(isSubmissionFail).length;
  const unsureCount = state.submissions.filter((s) => !isSubmissionApproved(s) && isSubmissionUnsure(s)).length;
  const approvedCount = state.submissions.filter(isSubmissionApproved).length;

  const pct = total > 0 ? Math.round((approvedCount / total) * 100) : 0;
  const allPassingCount = state.submissions.filter((s) => (s.grade || '').toLowerCase().includes('pass')).length;
  const passRate = total > 0 ? Math.round((allPassingCount / total) * 100) : 0;

  const fillEl = document.getElementById('header-progress-fill');
  if (fillEl) fillEl.style.width = `${pct}%`;
  const textEl = document.getElementById('header-progress-text');
  if (textEl) textEl.textContent = `${approvedCount} / ${total} Approved`;

  // Update 4 Filter Pill badges
  const badgePass = document.getElementById('badge-pass');
  if (badgePass) badgePass.textContent = passCount;
  const badgeFail = document.getElementById('badge-fail');
  if (badgeFail) badgeFail.textContent = failCount;
  const badgeUnsure = document.getElementById('badge-unsure');
  if (badgeUnsure) badgeUnsure.textContent = unsureCount;
  const badgeApproved = document.getElementById('badge-approved');
  if (badgeApproved) badgeApproved.textContent = approvedCount;

  const passRateEl = document.getElementById('summary-pass-rate');
  if (passRateEl) passRateEl.textContent = `Pass: ${passRate}%`;
}

// ============================================================================
// Filtering & Roster List
// ============================================================================
function applyFilterAndRender() {
  let list = [...state.submissions];

  // Tab filter: 'pass', 'fail', 'unsure', 'approved'
  if (state.activeFilter === 'pass') {
    list = list.filter(isSubmissionPass);
  } else if (state.activeFilter === 'fail') {
    list = list.filter(isSubmissionFail);
  } else if (state.activeFilter === 'unsure') {
    list = list.filter((s) => !isSubmissionApproved(s) && isSubmissionUnsure(s));
  } else if (state.activeFilter === 'approved') {
    list = list.filter(isSubmissionApproved);
  }

  // Search filter
  if (state.searchQuery.trim()) {
    const q = state.searchQuery.trim().toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    list = list.filter((s) => (s.name || '').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').includes(q));
  }

  state.filteredSubmissions = list;

  // Clear roster counter text
  const totalCount = state.submissions.length;
  const filteredCount = list.length;
  const countTextEl = document.getElementById('roster-count-text');
  if (countTextEl) {
    countTextEl.textContent = `${filteredCount} student${filteredCount === 1 ? '' : 's'}`;
  }

  // Keep selected index within bounds
  if (state.selectedIndex >= list.length) {
    state.selectedIndex = Math.max(0, list.length - 1);
  }

  renderRosterList();
  renderActiveStage();

  const bannerBadge = document.getElementById('banner-count-badge');
  if (bannerBadge) {
    bannerBadge.textContent = list.length;
  }
}

function renderRosterList() {
  const container = document.getElementById('roster-list');
  container.innerHTML = '';

  const posBadge = document.getElementById('roster-pos-badge');
  const prevMini = document.getElementById('roster-prev-btn');
  const nextMini = document.getElementById('roster-next-btn');

  if (state.filteredSubmissions.length === 0) {
    if (posBadge) posBadge.textContent = '0/0';
    if (prevMini) prevMini.disabled = true;
    if (nextMini) nextMini.disabled = true;

    container.innerHTML = `
      <div style="padding: 32px 16px; text-align: center; color: var(--sc-text-disabled); font-size: 13px;">
        <div style="margin-bottom: 8px; color: var(--sc-text-disabled);">
          <svg class="ic-lg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        </div>
        <div>No students in this view</div>
      </div>
    `;
    return;
  }

  if (posBadge) {
    posBadge.textContent = `${state.selectedIndex + 1}/${state.filteredSubmissions.length}`;
  }
  if (prevMini) prevMini.disabled = state.selectedIndex <= 0;
  if (nextMini) nextMini.disabled = state.selectedIndex >= state.filteredSubmissions.length - 1;

  state.filteredSubmissions.forEach((sub, idx) => {
    const key = `${sub.name}_${sub.attempt_number}`;
    const cardState = state.cardStates[key];
    const isSelected = idx === state.selectedIndex;

    const item = document.createElement('div');
    item.className = `roster-item ${isSelected ? 'is-selected' : ''}`;
    item.onclick = () => {
      state.selectedIndex = idx;
      renderRosterList();
      renderActiveStage();
    };

    // Status icon
    let statusIcon = `<svg class="ic-xs" style="color: var(--sc-text-disabled);" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`;
    const rLower = (sub.review_status || '').toLowerCase();
    if (rLower.includes('approved')) {
      statusIcon = `<svg class="ic-xs" style="color: var(--sc-success);" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>`;
    } else if (rLower.includes('overridden')) {
      statusIcon = `<svg class="ic-xs" style="color: var(--sc-accent);" viewBox="0 0 24 24"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>`;
    }

    // Badge styling
    let gradeBadge = '';
    const gLower = (sub.grade || '').toLowerCase();
    const agLower = (sub.agent_grade || '').toLowerCase();
    if (gLower.includes('unsure') || agLower.includes('unsure')) {
      gradeBadge = `<span class="badge badge-unsure">Unsure</span>`;
    } else if (gLower.includes('pass')) {
      gradeBadge = `<span class="badge badge-pass">Pass</span>`;
    } else {
      gradeBadge = `<span class="badge badge-fail">Fail</span>`;
    }

    item.innerHTML = `
      <div class="roster-item-main">
        <div class="student-avatar-sm">${getInitials(sub.name)}</div>
        <div class="roster-text-wrap">
          <div class="roster-student-name">${escapeHtml(sub.name)}</div>
          <div class="roster-sub-line">
            <span>Att #${sub.attempt_number}</span>
            <span>·</span>
            <span>${escapeHtml(formatCleanDate(sub.last_modified))}</span>
          </div>
        </div>
      </div>
      <div class="roster-item-status">
        ${cardState && cardState.isDirty ? `<span class="badge badge-unsaved" title="Unsaved edits"><svg class="ic-xs" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></span>` : ''}
        ${gradeBadge}
        <span style="display: flex; align-items: center;" title="${escapeHtml(sub.review_status)}">${statusIcon}</span>
      </div>
    `;

    container.appendChild(item);
  });
}

// ============================================================================
// Checklist Parser & Normalizer
// ============================================================================
function parseChecklistItem(itemOrLine) {
  let passed = true;
  let instruction = '';
  let comment = '';

  if (typeof itemOrLine === 'string') {
    const line = itemOrLine.trim();
    if (!line) return null;

    // Detect [STATUS] bracket anywhere in string
    const statusMatch = line.match(/\[(.*?)\]/);
    if (statusMatch) {
      const tag = statusMatch[1].toLowerCase().trim();
      if (/fail|failed|✗|false|no/.test(tag)) {
        passed = false;
      } else if (/pass|passed|✓|true|yes/.test(tag)) {
        passed = true;
      }
    }

    // Strip leading bullets, numbers, and [STATUS] bracket completely
    let clean = line.replace(/^[•\-\*\d\.\)\s]*/, '').trim();
    clean = clean.replace(/\[.*?\]\s*/, '').trim();
    clean = clean.replace(/^[•\-\*\s]+/, '').trim();

    // Extract comment
    const dashSplit = clean.split(/\s+[—–]\s+|\s+-\s+/);
    if (dashSplit.length >= 2) {
      instruction = dashSplit[0].trim();
      comment = dashSplit.slice(1).join(' — ').trim();
    } else {
      const parenMatch = clean.match(/^(.*?)\s*\(([^)]+)\)\s*$/);
      if (parenMatch && parenMatch[1].trim()) {
        instruction = parenMatch[1].trim();
        comment = parenMatch[2].trim();
      } else {
        instruction = clean;
      }
    }
  } else if (itemOrLine && typeof itemOrLine === 'object') {
    passed = Boolean(itemOrLine.passed);
    let rawInst = (itemOrLine.instruction || '').trim();
    comment = (itemOrLine.comment || '').trim();

    // Check if raw instruction still has [STATUS] bracket
    const statusMatch = rawInst.match(/\[(.*?)\]/);
    if (statusMatch) {
      const tag = statusMatch[1].toLowerCase().trim();
      if (/fail|failed|✗|false|no/.test(tag)) {
        passed = false;
      } else if (/pass|passed|✓|true|yes/.test(tag)) {
        passed = true;
      }
      rawInst = rawInst.replace(/^[•\-\*\d\.\)\s]*/, '').replace(/\[.*?\]\s*/, '').trim();
    }
    instruction = rawInst.replace(/^[•\-\*\s]+/, '').trim();

    // If comment was empty, check if instruction ends with (comment)
    if (!comment) {
      const parenMatch = instruction.match(/^(.*?)\s*\(([^)]+)\)\s*$/);
      if (parenMatch && parenMatch[1].trim()) {
        instruction = parenMatch[1].trim();
        comment = parenMatch[2].trim();
      }
    }
  }

  if (!instruction && !comment) return null;
  return { passed, instruction, comment };
}

// ============================================================================
// Right Stage: Focused Review Workspace
// ============================================================================
function computeIsDirty(cardState, sub) {
  if (!cardState || !sub) return false;

  const currentGrade = (cardState.grade || '').trim().toLowerCase();
  const currentFeedback = (cardState.feedback || '').trim();

  const origGrade = (cardState.originalGrade || sub.grade || '').trim().toLowerCase();
  const origFeedback = (cardState.originalFeedback || sub.feedback_comment || '').trim();

  // Normalize AI grade baseline
  const rawAiGrade = (sub.agent_grade || sub.grade || '').trim();
  const aiGradeNorm = rawAiGrade.toLowerCase().includes('pass')
    ? 'pass'
    : (rawAiGrade.toLowerCase().includes('fail') ? 'fail' : '');

  // Grade is clean if it matches the loaded original grade OR matches AI's baseline grade
  const gradeIsClean = (currentGrade === origGrade) || (aiGradeNorm && currentGrade === aiGradeNorm);

  // Feedback is clean if it matches original / baseline comment
  const feedbackIsClean = (currentFeedback === origFeedback);

  return !gradeIsClean || !feedbackIsClean;
}

function renderActiveStage() {
  const placeholder = document.getElementById('stage-placeholder');
  const stageContent = document.getElementById('stage-content');

  const sub = state.filteredSubmissions[state.selectedIndex];
  if (!sub) {
    placeholder.style.display = 'flex';
    stageContent.style.display = 'none';
    return;
  }

  placeholder.style.display = 'none';
  stageContent.style.display = 'flex';

  const key = `${sub.name}_${sub.attempt_number}`;
  if (!state.cardStates[key]) {
    const defaultGrade = sub.grade.toLowerCase().includes('pass') ? 'Pass' : 'Fail';
    state.cardStates[key] = {
      grade: defaultGrade,
      feedback: sub.feedback_comment || '',
      originalGrade: defaultGrade,
      originalFeedback: sub.feedback_comment || '',
      isDirty: false,
    };
  } else {
    // Keep isDirty dynamically synchronized with baseline
    state.cardStates[key].isDirty = computeIsDirty(state.cardStates[key], sub);
  }
  const cardState = state.cardStates[key];

  // 1. Stage Header
  const avatarEl = document.getElementById('stage-avatar');
  if (avatarEl) avatarEl.textContent = getInitials(sub.name);
  const nameEl = document.getElementById('stage-student-name');
  if (nameEl) nameEl.textContent = sub.name;
  const attemptBadge = document.getElementById('stage-attempt-badge');
  if (attemptBadge) attemptBadge.textContent = `Attempt #${sub.attempt_number}`;
  const timeEl = document.getElementById('stage-submission-time');
  if (timeEl) timeEl.textContent = `Submitted: ${sub.last_modified || 'N/A'}`;
  const statusEl = document.getElementById('stage-review-status-text');
  if (statusEl) {
    statusEl.textContent = sub.mentor_reviewer
      ? `${sub.review_status} (by ${sub.mentor_reviewer})`
      : sub.review_status;
  }

  // Unsaved badge
  const unsavedBadge = document.getElementById('stage-unsaved-badge');
  if (unsavedBadge) unsavedBadge.style.display = cardState.isDirty ? 'inline-flex' : 'none';

  // Override Active Badge
  const overrideBadge = document.getElementById('stage-override-badge');
  const isOverridden = (sub.review_status || '').toLowerCase().includes('overridden');
  const isApproved = (sub.review_status || '').toLowerCase().includes('approved');
  if (overrideBadge) {
    overrideBadge.style.display = isOverridden ? 'inline-flex' : 'none';
  }

  // Button text: "Update Override" if already overridden, else "Save Override"
  const overrideBtn = document.getElementById('stage-override-btn');
  if (overrideBtn) {
    const btnSpan = overrideBtn.querySelector('span') || overrideBtn;
    btnSpan.textContent = isOverridden ? 'Update Override' : 'Save Override';
  }

  // Agent Grade Badge
  const agentBadge = document.getElementById('stage-agent-grade-badge');
  const agLower = (sub.agent_grade || '').toLowerCase();
  if (agentBadge) {
    agentBadge.className = 'badge';
    if (agLower.includes('unsure')) {
      agentBadge.classList.add('badge-unsure');
      agentBadge.textContent = `AI: ${sub.agent_grade}`;
    } else if (agLower.includes('pass')) {
      agentBadge.classList.add('badge-pass');
      agentBadge.textContent = 'AI: Pass';
    } else {
      agentBadge.classList.add('badge-fail');
      agentBadge.textContent = 'AI: Fail';
    }
  }

  // Drive folder link
  const driveBtn = document.getElementById('stage-drive-btn');
  if (driveBtn) {
    if (sub.media_folder && sub.media_folder.startsWith('http')) {
      driveBtn.href = sub.media_folder;
      driveBtn.style.display = 'inline-flex';
    } else {
      driveBtn.href = '#';
      driveBtn.style.display = 'none';
    }
  }

  // Pagination indicator
  const paginationPos = document.getElementById('pagination-pos');
  if (paginationPos) {
    paginationPos.textContent = `${state.selectedIndex + 1} of ${state.filteredSubmissions.length}`;
  }

  // 2. Left Panel: Online text
  const textEl = document.getElementById('stage-online-text');
  if (textEl) {
    textEl.textContent = sub.online_text || '(No written online text submitted by student)';
  }

  // 3. Middle Stage Card: Load submission images from Drive via OAuth
  loadSubmissionMedia(sub);

  // 4. Right Collapsible Panel: Checklist breakdown
  const checklistPill = document.getElementById('stage-agent-verdict-pill');
  if (checklistPill) {
    checklistPill.className = `badge ${agLower.includes('pass') ? 'badge-pass' : agLower.includes('unsure') ? 'badge-unsure' : 'badge-fail'}`;
    checklistPill.textContent = sub.agent_grade;
  }
  const evalBannerBadge = document.getElementById('eval-banner-verdict-badge');
  if (evalBannerBadge) {
    evalBannerBadge.className = `banner-count-badge ${agLower.includes('pass') ? 'is-pass' : agLower.includes('unsure') ? 'is-unsure' : 'is-fail'}`;
    evalBannerBadge.textContent = sub.agent_grade || 'AI';
  }

  const checklistArea = document.getElementById('stage-checklist-container');
  if (checklistArea) {
    checklistArea.innerHTML = '';
  }

  let rawList = [];
  if (sub.checklist_items && sub.checklist_items.length > 0) {
    rawList = sub.checklist_items;
  } else if (sub.agent_checklist) {
    rawList = sub.agent_checklist.split('\n');
  }

  const parsedItems = rawList.map(parseChecklistItem).filter(Boolean);
  if (parsedItems.length > 0) {
    parsedItems.forEach((item) => {
      const row = document.createElement('div');
      row.className = `checklist-item-card ${item.passed ? 'is-pass' : 'is-fail'}`;
      row.innerHTML = `
        <div class="check-icon-circle ${item.passed ? 'pass' : 'fail'}">
          ${item.passed ? '<svg class="ic-xs" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>' : '<svg class="ic-xs" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'}
        </div>
        <div>
          <div class="checklist-item-text">${escapeHtml(item.instruction)}</div>
          ${item.comment ? `<div class="checklist-item-comment">${escapeHtml(item.comment)}</div>` : ''}
        </div>
      `;
      checklistArea.appendChild(row);
    });
  } else {
    checklistArea.innerHTML = `
      <div style="color: var(--text-muted); font-size: 12.5px; padding: 12px 0;">
        No granular checklist items recorded for this submission.
      </div>
    `;
  }

  // 4. Decision Dock
  const passBtn = document.getElementById('btn-grade-pass');
  const failBtn = document.getElementById('btn-grade-fail');
  if (passBtn) passBtn.className = `grade-toggle-btn ${cardState.grade === 'Pass' ? 'is-active-pass' : ''}`;
  if (failBtn) failBtn.className = `grade-toggle-btn ${cardState.grade === 'Fail' ? 'is-active-fail' : ''}`;

  // Dock Review Status Summary Box
  const dockReviewStatusVal = document.getElementById('dock-review-status-val');
  const dockReviewedByVal = document.getElementById('dock-reviewed-by-val');
  const dockStatusDot = document.getElementById('dock-status-dot');
  const dockUnsavedIndicator = document.getElementById('dock-unsaved-indicator');

  if (dockReviewStatusVal) {
    dockReviewStatusVal.textContent = sub.review_status || 'Pending Review';
  }
  if (dockReviewedByVal) {
    if (sub.mentor_reviewer) {
      dockReviewedByVal.textContent = `Reviewed by ${sub.mentor_reviewer}`;
    } else {
      dockReviewedByVal.textContent = 'Awaiting Mentor Review';
    }
  }
  if (dockStatusDot) {
    dockStatusDot.className = 'status-indicator-dot';
    if (isApproved) dockStatusDot.classList.add('is-approved');
    else if (isOverridden) dockStatusDot.classList.add('is-overridden');
  }

  if (dockUnsavedIndicator) {
    dockUnsavedIndicator.style.display = cardState.isDirty ? 'inline-flex' : 'none';
  }

  const feedbackInput = document.getElementById('stage-feedback-input');
  if (feedbackInput) feedbackInput.value = cardState.feedback;
  const charCount = document.getElementById('stage-char-count');
  if (charCount) charCount.textContent = `${cardState.feedback.length} chars`;
}

// ============================================================================
// Media Preview Carousel (Smart Multi-Tier Preloading & Zero-Flicker Stage)
// ============================================================================
const submissionMediaCache = new Map(); // key -> Array of media items
const imagePreloadPromises = new Map();  // url -> Promise<{ img: HTMLImageElement, ready: boolean }>
const warmImageSet = new Set();          // Set of URLs that are 100% decoded in memory
let prefetchAdjacentTimer = null;
let currentTransitionToken = 0;

function getSubmissionMediaKey(sub) {
  if (!sub) return '';
  return `${sub.name || ''}__${sub.media_folder || ''}`;
}

function preloadAndDecodeImage(url) {
  if (!url) return Promise.resolve({ img: null, ready: false });
  if (imagePreloadPromises.has(url)) {
    return imagePreloadPromises.get(url);
  }

  const promise = new Promise((resolve) => {
    const img = new Image();
    img.decoding = 'async';
    img.src = url;

    const onReady = () => {
      warmImageSet.add(url);
      resolve({ img, ready: true });
    };
    const onError = () => {
      resolve({ img: null, ready: false });
    };

    if (img.decode) {
      img.decode().then(onReady).catch(() => {
        if (img.complete && img.naturalWidth > 0) {
          onReady();
        } else {
          img.onload = onReady;
          img.onerror = onError;
        }
      });
    } else {
      if (img.complete && img.naturalWidth > 0) {
        onReady();
      } else {
        img.onload = onReady;
        img.onerror = onError;
      }
    }
  });

  imagePreloadPromises.set(url, promise);
  return promise;
}

function preloadSubmissionImages(images) {
  if (!images || !images.length) return;

  // Immediate Priority: decode current/first image
  const first = images[0];
  if (first && first.media_type === 'image') {
    if (first.preview_url) preloadAndDecodeImage(first.preview_url);
    if (first.thumb_url && first.thumb_url !== first.preview_url) {
      preloadAndDecodeImage(first.thumb_url);
    }
  }

  // Next Priority: decode remaining images in sequence
  for (let i = 1; i < images.length; i++) {
    const item = images[i];
    if (item.media_type === 'image' && item.preview_url) {
      preloadAndDecodeImage(item.preview_url);
      if (item.thumb_url && item.thumb_url !== item.preview_url) {
        preloadAndDecodeImage(item.thumb_url);
      }
    }
  }
}

function schedulePrefetchAdjacentStudents() {
  if (prefetchAdjacentTimer) clearTimeout(prefetchAdjacentTimer);
  prefetchAdjacentTimer = setTimeout(async () => {
    const subs = state.filteredSubmissions;
    const idx = state.selectedIndex;
    if (!subs || !subs.length) return;

    // Speculatively prefetch next 2 students and previous 1 student
    const neighbors = [subs[idx + 1], subs[idx + 2], subs[idx - 1]].filter(Boolean);
    for (const neighbor of neighbors) {
      const key = getSubmissionMediaKey(neighbor);
      if (!submissionMediaCache.has(key) && neighbor.media_folder) {
        try {
          let data;
          if (typeof API !== 'undefined' && typeof API.getSubmissionMedia === 'function') {
            data = await API.getSubmissionMedia(neighbor.media_folder, neighbor.name);
          } else {
            const params = new URLSearchParams();
            if (neighbor.media_folder) params.set('media_folder', neighbor.media_folder);
            if (neighbor.name) params.set('student_name', neighbor.name);
            const res = await fetch(`/api/submission-media?${params.toString()}`);
            if (res.ok) data = await res.json();
          }
          if (data && data.images) {
            const valid = data.images.filter(item => {
              const name = (item.name || '').toLowerCase();
              return item.media_type !== 'other' && item.media_type !== 'file' && !name.endsWith('.txt');
            });
            submissionMediaCache.set(key, valid);
            // Preload adjacent images into decoded memory cache!
            preloadSubmissionImages(valid);
          }
        } catch (_) {
          // Silent prefetch failure
        }
      }
    }
  }, 200);
}

async function loadSubmissionMedia(sub) {
  state.currentMediaIndex = 0;
  const key = getSubmissionMediaKey(sub);

  const loadingBar = document.getElementById('media-loading-bar');
  const loadingState = document.getElementById('media-loading-state');
  const emptyState = document.getElementById('media-empty-state');
  const counterBadge = document.getElementById('media-counter-badge');

  // 1. Instant Cache Hit: If already in memory, display immediately with 0 delay!
  if (submissionMediaCache.has(key)) {
    state.currentMediaImages = submissionMediaCache.get(key) || [];
    state.isLoadingMedia = false;
    if (loadingBar) loadingBar.classList.remove('is-active');
    if (loadingState) loadingState.style.display = 'none';
    renderCurrentMediaImage(0);
    schedulePrefetchAdjacentStudents();
    return;
  }

  // 2. Cache Miss: Fetch in background while showing sleek accent bar
  state.currentMediaImages = [];
  state.isLoadingMedia = true;

  if (loadingBar) loadingBar.classList.add('is-active');
  if (counterBadge) counterBadge.textContent = 'Loading...';

  // Keep viewport visible if possible for layout stability
  if (emptyState) emptyState.style.display = 'none';

  try {
    let data;
    if (typeof API !== 'undefined' && typeof API.getSubmissionMedia === 'function') {
      data = await API.getSubmissionMedia(sub.media_folder, sub.name);
    } else {
      const params = new URLSearchParams();
      if (sub.media_folder) params.set('media_folder', sub.media_folder);
      if (sub.name) params.set('student_name', sub.name);
      const res = await fetch(`/api/submission-media?${params.toString()}`);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      data = await res.json();
    }

    // Strictly filter out any .txt or non-media files
    const rawImages = data.images || [];
    const validImages = rawImages.filter(item => {
      const name = (item.name || '').toLowerCase();
      const type = item.media_type;
      return type !== 'other' && type !== 'file' && !name.endsWith('.txt');
    });

    state.currentMediaImages = validImages;
    submissionMediaCache.set(key, validImages);

    // Preload all images in background so navigation is 100% instant
    preloadSubmissionImages(validImages);
  } catch (err) {
    console.warn('Failed to load submission media:', err);
    state.currentMediaImages = [];
  } finally {
    state.isLoadingMedia = false;
    if (loadingBar) loadingBar.classList.remove('is-active');
    if (loadingState) loadingState.style.display = 'none';
    renderCurrentMediaImage(0);
    schedulePrefetchAdjacentStudents();
  }
}

function renderCurrentMediaImage(direction = 0) {
  const transitionId = ++currentTransitionToken;
  const images = state.currentMediaImages;
  const idx = state.currentMediaIndex;

  const emptyState = document.getElementById('media-empty-state');
  const carouselViewport = document.getElementById('media-carousel-viewport');
  const counterBadge = document.getElementById('media-counter-badge');
  const canvasWrap = document.getElementById('media-canvas-wrap');
  const skeletonLoader = document.getElementById('media-skeleton-loader');
  const previewImg = document.getElementById('media-preview-img');
  const stagingImg = document.getElementById('media-staging-img');
  const previewVideo = document.getElementById('media-preview-video');
  const audioWrap = document.getElementById('media-preview-audio-wrap');
  const previewAudio = document.getElementById('media-preview-audio');
  const fileWrap = document.getElementById('media-preview-file-wrap');
  const docFilename = document.getElementById('media-doc-filename');
  const prevBtn = document.getElementById('media-nav-prev-btn');
  const nextBtn = document.getElementById('media-nav-next-btn');

  if (!images || images.length === 0) {
    if (emptyState) emptyState.style.display = 'flex';
    if (carouselViewport) carouselViewport.style.display = 'none';
    if (counterBadge) counterBadge.textContent = '0 items';
    if (skeletonLoader) skeletonLoader.classList.remove('is-active');
    return;
  }

  if (emptyState) emptyState.style.display = 'none';
  if (carouselViewport) carouselViewport.style.display = 'flex';

  if (counterBadge) {
    counterBadge.textContent = `${idx + 1} of ${images.length}`;
  }

  const currentMedia = images[idx];
  const mediaType = currentMedia.media_type || 'image';

  // Stop & hide non-image players
  if (previewVideo) {
    previewVideo.pause();
    if (mediaType !== 'video') previewVideo.style.display = 'none';
  }
  if (audioWrap) {
    if (previewAudio) previewAudio.pause();
    if (mediaType !== 'audio') audioWrap.style.display = 'none';
  }
  if (fileWrap && mediaType !== 'other') fileWrap.style.display = 'none';

  if (mediaType === 'image') {
    if (canvasWrap) canvasWrap.style.display = 'flex';
    const targetUrl = currentMedia.preview_url;
    const targetThumb = currentMedia.thumb_url || targetUrl;
    const isWarm = warmImageSet.has(targetUrl);

    if (previewImg && stagingImg) {
      previewImg.style.display = 'block';
      previewImg.style.cursor = 'zoom-in';
      previewImg.title = 'Click to open in Google Drive';
      previewImg.alt = currentMedia.name || 'Submission photo';

      previewImg.onclick = () => {
        window.open(currentMedia.drive_url || targetUrl, '_blank');
      };
      stagingImg.onclick = () => {
        window.open(currentMedia.drive_url || targetUrl, '_blank');
      };

      previewImg.onerror = () => {
        if (currentMedia.api_stream_url && previewImg.src !== currentMedia.api_stream_url) {
          previewImg.src = currentMedia.api_stream_url;
        }
      };

      // DIRECTIONAL CAROUSEL TRANSITION (User clicked Next or Prev)
      if (direction !== 0 && previewImg.src && previewImg.style.display !== 'none') {
        if (skeletonLoader) skeletonLoader.classList.remove('is-active');

        // Prepare incoming staging image off-center
        const offset = direction > 0 ? '24px' : '-24px';
        const exitOffset = direction > 0 ? '-24px' : '24px';

        stagingImg.style.transition = 'none';
        stagingImg.src = targetUrl;
        stagingImg.className = 'media-preview-img media-staging-img is-crisp';
        stagingImg.style.opacity = '0';
        stagingImg.style.transform = `translate(calc(-50% + ${offset}), -50%) scale(0.98)`;
        stagingImg.style.display = 'block';

        preloadAndDecodeImage(targetUrl).then(() => {
          if (currentTransitionToken !== transitionId) return;

          requestAnimationFrame(() => {
            stagingImg.style.transition = 'opacity 0.2s cubic-bezier(0.16, 1, 0.3, 1), transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)';
            previewImg.style.transition = 'opacity 0.2s cubic-bezier(0.16, 1, 0.3, 1), transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)';

            stagingImg.style.opacity = '1';
            stagingImg.style.transform = 'translate(-50%, -50%) scale(1)';
            previewImg.style.opacity = '0';
            previewImg.style.transform = `translate(calc(-50% + ${exitOffset}), -50%) scale(0.98)`;

            setTimeout(() => {
              if (currentTransitionToken !== transitionId) return;
              previewImg.src = targetUrl;
              previewImg.style.transition = 'none';
              previewImg.style.opacity = '1';
              previewImg.style.transform = 'translate(-50%, -50%)';
              previewImg.className = 'media-preview-img is-crisp';
              stagingImg.style.display = 'none';
            }, 210);
          });
        });
      } else {
        // INITIAL LOAD / STUDENT SWITCH (direction === 0)
        stagingImg.style.display = 'none';
        previewImg.style.transition = 'none';

        if (isWarm) {
          // 100% INSTANT: Decoded in memory cache! Zero flash, zero wait!
          if (skeletonLoader) skeletonLoader.classList.remove('is-active');
          previewImg.src = targetUrl;
          previewImg.className = 'media-preview-img is-crisp';
          previewImg.style.opacity = '1';
          previewImg.style.transform = 'translate(-50%, -50%)';
        } else {
          // FAST PROGRESSIVE: Show micro-thumb immediately + blur-up while high-res decodes
          if (skeletonLoader) skeletonLoader.classList.add('is-active');
          previewImg.src = targetThumb;
          previewImg.className = 'media-preview-img is-blur-up';
          previewImg.style.opacity = '0.92';
          previewImg.style.transform = 'translate(-50%, -50%)';

          preloadAndDecodeImage(targetUrl).then(({ ready }) => {
            if (currentTransitionToken !== transitionId) return;
            if (ready && state.currentMediaImages[state.currentMediaIndex] === currentMedia) {
              if (skeletonLoader) skeletonLoader.classList.remove('is-active');
              previewImg.style.transition = 'opacity 0.2s ease, filter 0.22s ease';
              previewImg.src = targetUrl;
              previewImg.className = 'media-preview-img is-crisp';
              previewImg.style.opacity = '1';
            }
          });
        }
      }
    }
  } else {
    // Hide image canvas for video/audio
    if (canvasWrap) canvasWrap.style.display = 'none';
    if (skeletonLoader) skeletonLoader.classList.remove('is-active');

    if (mediaType === 'video' && previewVideo) {
      previewVideo.style.display = 'block';
      previewVideo.src = currentMedia.preview_url;
    } else if (mediaType === 'audio' && audioWrap && previewAudio) {
      audioWrap.style.display = 'flex';
      previewAudio.src = currentMedia.preview_url;
    } else if (fileWrap) {
      fileWrap.style.display = 'flex';
      if (docFilename) docFilename.textContent = currentMedia.name || 'media';
    }
  }


  if (prevBtn) {
    prevBtn.disabled = idx <= 0;
  }
  if (nextBtn) {
    nextBtn.disabled = idx >= images.length - 1;
  }
}

function navigateMedia(step) {
  const images = state.currentMediaImages;
  if (!images || images.length <= 1) return;
  const newIndex = state.currentMediaIndex + step;
  if (newIndex >= 0 && newIndex < images.length) {
    state.currentMediaIndex = newIndex;
    renderCurrentMediaImage(step);
  }
}

// ============================================================================
// Actions & Review Handlers
// ============================================================================
function setStageGrade(grade) {
  const sub = state.filteredSubmissions[state.selectedIndex];
  if (!sub) return;
  const key = `${sub.name}_${sub.attempt_number}`;
  if (!state.cardStates[key]) {
    const defaultGrade = sub.grade.toLowerCase().includes('pass') ? 'Pass' : 'Fail';
    state.cardStates[key] = {
      grade: defaultGrade,
      feedback: sub.feedback_comment || '',
      originalGrade: defaultGrade,
      originalFeedback: sub.feedback_comment || '',
      isDirty: false,
    };
  }

  const prevGrade = state.cardStates[key].grade || (sub.grade.toLowerCase().includes('pass') ? 'Pass' : 'Fail');
  const gradeChanged = prevGrade !== grade;

  state.cardStates[key].grade = grade;
  const isDirty = computeIsDirty(state.cardStates[key], sub);
  state.cardStates[key].isDirty = isDirty;

  const passBtn = document.getElementById('btn-grade-pass');
  const failBtn = document.getElementById('btn-grade-fail');
  if (passBtn) passBtn.className = `grade-toggle-btn ${grade === 'Pass' ? 'is-active-pass' : ''}`;
  if (failBtn) failBtn.className = `grade-toggle-btn ${grade === 'Fail' ? 'is-active-fail' : ''}`;

  const unsavedBadge = document.getElementById('stage-unsaved-badge');
  if (unsavedBadge) unsavedBadge.style.display = isDirty ? 'inline-flex' : 'none';
  const dockUnsaved = document.getElementById('dock-unsaved-indicator');
  if (dockUnsaved) dockUnsaved.style.display = isDirty ? 'inline-flex' : 'none';
  renderRosterList();

  // Normalize AI grade baseline
  const rawAiGrade = (sub.agent_grade || sub.grade || '').trim();
  const aiGradeNorm = rawAiGrade.toLowerCase().includes('pass') ? 'Pass' : 'Fail';

  // Only prompt when DRIFTING AWAY from AI's grade (not when coming back to AI's grade)
  const isDriftingFromAi = grade !== aiGradeNorm;

  if (gradeChanged && isDriftingFromAi) {
    openRulePromptModal(aiGradeNorm, grade);
  }
}

function onStageFeedbackInput(val) {
  const sub = state.filteredSubmissions[state.selectedIndex];
  if (!sub) return;
  const key = `${sub.name}_${sub.attempt_number}`;
  if (!state.cardStates[key]) {
    const defaultGrade = sub.grade.toLowerCase().includes('pass') ? 'Pass' : 'Fail';
    state.cardStates[key] = {
      grade: defaultGrade,
      feedback: sub.feedback_comment || '',
      originalGrade: defaultGrade,
      originalFeedback: sub.feedback_comment || '',
      isDirty: false,
    };
  }

  state.cardStates[key].feedback = val;
  const isDirty = computeIsDirty(state.cardStates[key], sub);
  state.cardStates[key].isDirty = isDirty;

  const charCountEl = document.getElementById('stage-char-count');
  if (charCountEl) charCountEl.textContent = `${val.length} chars`;
  const unsavedBadge = document.getElementById('stage-unsaved-badge');
  if (unsavedBadge) unsavedBadge.style.display = isDirty ? 'inline-flex' : 'none';
  const dockUnsaved = document.getElementById('dock-unsaved-indicator');
  if (dockUnsaved) dockUnsaved.style.display = isDirty ? 'inline-flex' : 'none';
  renderRosterList();
}

function revertStageFeedback() {
  const sub = state.filteredSubmissions[state.selectedIndex];
  if (!sub) return;
  const key = `${sub.name}_${sub.attempt_number}`;
  if (!state.cardStates[key]) return;

  state.cardStates[key].feedback = state.cardStates[key].originalFeedback || sub.feedback_comment || '';
  state.cardStates[key].grade = state.cardStates[key].originalGrade || (sub.grade.toLowerCase().includes('pass') ? 'Pass' : 'Fail');
  state.cardStates[key].isDirty = false;

  const dockUnsaved = document.getElementById('dock-unsaved-indicator');
  if (dockUnsaved) dockUnsaved.style.display = 'none';
  const unsavedBadge = document.getElementById('stage-unsaved-badge');
  if (unsavedBadge) unsavedBadge.style.display = 'none';

  renderActiveStage();
  renderRosterList();
  showToast('Reverted to AI comment', 'info');
}

async function submitStageReview(reviewStatus, advanceToNext = false) {
  const sub = state.filteredSubmissions[state.selectedIndex];
  if (!sub) return;
  const key = `${sub.name}_${sub.attempt_number}`;
  const cardState = state.cardStates[key] || { grade: 'Pass', feedback: '' };

  const approveBtn = document.getElementById('stage-approve-next-btn');
  const overrideBtn = document.getElementById('stage-override-btn');
  approveBtn.disabled = true;
  overrideBtn.disabled = true;

  showToast(`Saving review for ${sub.name}...`, 'info');

  try {
    await API.submitReview(state.activeActivity, {
      student_name: sub.name,
      attempt_number: sub.attempt_number,
      grade: cardState.grade,
      feedback_comment: cardState.feedback.trim(),
      review_status: reviewStatus,
      mentor_name: state.mentorName,
    });

    showToast(`Saved for ${sub.name}!`, 'success');
    if (state.cardStates[key]) state.cardStates[key].isDirty = false;

    // Refresh activity data
    await selectActivity(state.activeActivity);

    // If advance requested, move to next item
    if (advanceToNext && state.filteredSubmissions.length > 0) {
      if (state.selectedIndex < state.filteredSubmissions.length) {
        renderActiveStage();
      }
    }
  } catch (err) {
    console.error('Save failed:', err);
    showToast('Failed to save: ' + err.message, 'error');
  } finally {
    approveBtn.disabled = false;
    overrideBtn.disabled = false;
  }
}

// Navigation
function navigateRoster(delta) {
  if (state.filteredSubmissions.length === 0) return;
  const nextIdx = state.selectedIndex + delta;
  if (nextIdx >= 0 && nextIdx < state.filteredSubmissions.length) {
    state.selectedIndex = nextIdx;
    renderRosterList();
    renderActiveStage();
  }
}

// Copy Text
function copySubmissionText() {
  const sub = state.filteredSubmissions[state.selectedIndex];
  if (!sub || !sub.online_text) return;
  navigator.clipboard.writeText(sub.online_text);
  showToast('Copied student text to clipboard', 'success');
}

// Batch Actions
async function triggerBatchApprove() {
  if (!confirm(`Batch approve all confident pending passes for '${state.activeActivity}'?`)) {
    return;
  }
  showToast('Running batch approval...', 'info');
  try {
    const res = await API.batchApprove(state.activeActivity, state.mentorName);
    showToast(`Batch approved ${res.approved_count || 0} submission(s)!`, 'success');
    await selectActivity(state.activeActivity);
  } catch (err) {
    showToast('Batch approve error: ' + err.message, 'error');
  }
}

async function triggerSyncLocal() {
  const folder = prompt('Enter assignment folder path (or leave blank to sync latest download):', '');
  if (folder === null) return;
  showToast('Syncing folder to registry...', 'info');
  try {
    const res = await API.syncLocal(state.activeActivity, folder.trim() || null);
    showToast(`Synced ${res.total_synced || 0} student(s)!`, 'success');
    await selectActivity(state.activeActivity);
  } catch (err) {
    showToast('Sync error: ' + err.message, 'error');
  }
}

// ============================================================================
// Activity Edge Case & Guideline Prompt Modal
// ============================================================================
let activeModalGradeChange = null;

function toggleExistingRulesBanner() {
  const bannerContent = document.getElementById('existing-rules-banner-content');
  const chevron = document.getElementById('existing-rules-chevron');
  const toggleHint = document.getElementById('existing-rules-toggle-hint');
  const bannerToggle = document.getElementById('existing-rules-banner-toggle');
  if (!bannerContent) return;

  const isOpen = bannerContent.style.display !== 'none';
  bannerContent.style.display = isOpen ? 'none' : 'block';
  if (chevron) chevron.classList.toggle('is-open', !isOpen);
  if (toggleHint) toggleHint.textContent = isOpen ? 'View' : 'Hide';
  if (bannerToggle) bannerToggle.setAttribute('aria-expanded', String(!isOpen));
}

function openRulePromptModal(fromGrade, toGrade, isManual = false) {
  const modal = document.getElementById('rule-prompt-modal');
  if (!modal) return;

  activeModalGradeChange = { fromGrade, toGrade, isManual };

  const gradeBadge = document.getElementById('rule-modal-grade-change-badge');
  const actBadge = document.getElementById('rule-modal-activity-badge');
  const titleEl = document.getElementById('rule-modal-title');
  const edgeInput = document.getElementById('rule-edge-case-input');
  const guideInput = document.getElementById('rule-guideline-input');
  const prevEdgeEl = document.getElementById('preview-existing-edge-cases');
  const prevGuideEl = document.getElementById('preview-existing-guidelines');
  const countText = document.getElementById('existing-rules-count-text');
  const bannerContent = document.getElementById('existing-rules-banner-content');
  const bannerToggle = document.getElementById('existing-rules-banner-toggle');
  const chevron = document.getElementById('existing-rules-chevron');
  const toggleHint = document.getElementById('existing-rules-toggle-hint');

  if (gradeBadge) {
    if (isManual) {
      gradeBadge.textContent = 'Rule Editor';
      gradeBadge.className = 'badge badge-subtle';
    } else {
      gradeBadge.textContent = `${fromGrade} ➔ ${toGrade}`;
      gradeBadge.className = `badge ${toGrade === 'Pass' ? 'badge-pass' : 'badge-fail'}`;
    }
  }

  if (titleEl) {
    titleEl.textContent = `Add Rule for ${state.activeActivity || 'Activity'}`;
  }

  if (edgeInput) edgeInput.value = '';
  if (guideInput) guideInput.value = '';

  const curEdge = (state.activityEdgeCases || '').trim();
  const curGuide = (state.activityGuidelines || '').trim();

  if (prevEdgeEl) prevEdgeEl.textContent = curEdge || '(None recorded yet)';
  if (prevGuideEl) prevGuideEl.textContent = curGuide || '(None recorded yet)';

  const hasEdge = Boolean(curEdge);
  const hasGuide = Boolean(curGuide);
  const activeCount = (hasEdge ? 1 : 0) + (hasGuide ? 1 : 0);

  if (countText) {
    countText.textContent = activeCount > 0 ? `(${activeCount} saved)` : '(None yet)';
  }

  // Reset dropdown banner to collapsed on modal open
  if (bannerContent) bannerContent.style.display = 'none';
  if (chevron) chevron.classList.remove('is-open');
  if (toggleHint) toggleHint.textContent = 'View';
  if (bannerToggle) bannerToggle.setAttribute('aria-expanded', 'false');

  setRuleTab('both');

  modal.style.display = 'flex';
  modal.setAttribute('aria-hidden', 'false');

  setTimeout(() => {
    if (edgeInput) edgeInput.focus();
  }, 100);
}

function closeRulePromptModal() {
  const modal = document.getElementById('rule-prompt-modal');
  if (!modal) return;
  modal.style.display = 'none';
  modal.setAttribute('aria-hidden', 'true');
  activeModalGradeChange = null;
}

function setRuleTab(tab) {
  const groupEdge = document.getElementById('group-edge-case');
  const groupGuide = document.getElementById('group-guideline');
  const tabBoth = document.getElementById('tab-btn-both');
  const tabEdge = document.getElementById('tab-btn-edge');
  const tabGuide = document.getElementById('tab-btn-guide');
  const edgeInput = document.getElementById('rule-edge-case-input');
  const guideInput = document.getElementById('rule-guideline-input');

  if (tabBoth) tabBoth.classList.toggle('is-active', tab === 'both');
  if (tabEdge) tabEdge.classList.toggle('is-active', tab === 'edge');
  if (tabGuide) tabGuide.classList.toggle('is-active', tab === 'guide');

  if (groupEdge) groupEdge.style.display = (tab === 'both' || tab === 'edge') ? 'flex' : 'none';
  if (groupGuide) groupGuide.style.display = (tab === 'both' || tab === 'guide') ? 'flex' : 'none';

  // Expand textarea in single mode, keep compact in both mode
  if (edgeInput) {
    edgeInput.classList.toggle('is-expanded', tab === 'edge');
    edgeInput.rows = tab === 'both' ? 2 : 4;
  }
  if (guideInput) {
    guideInput.classList.toggle('is-expanded', tab === 'guide');
    guideInput.rows = tab === 'both' ? 2 : 4;
  }
}

async function saveActivityRuleFromModal() {
  const edgeInput = document.getElementById('rule-edge-case-input');
  const guideInput = document.getElementById('rule-guideline-input');
  const saveBtn = document.getElementById('rule-modal-save-btn');

  const edgeVal = (edgeInput ? edgeInput.value : '').trim();
  const guideVal = (guideInput ? guideInput.value : '').trim();

  if (!edgeVal && !guideVal) {
    showToast('Please enter an Edge Case or Guideline (or click Skip)', 'warning');
    if (edgeInput) edgeInput.focus();
    return;
  }

  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<span>Saving to Sheet...</span>';
  }

  try {
    const res = await API.addActivityRule(state.activeActivity, {
      edge_case: edgeVal || null,
      guideline: guideVal || null,
      mentor_name: state.mentorName,
    });

    if (res.edge_cases) state.activityEdgeCases = res.edge_cases;
    if (res.guidelines) state.activityGuidelines = res.guidelines;

    showToast(`Saved to Activity Details tab for '${state.activeActivity}'!`, 'success');
    closeRulePromptModal();
  } catch (err) {
    console.error('Failed to save rule:', err);
    showToast('Failed to save rule to sheet: ' + err.message, 'error');
  } finally {
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.innerHTML = `
        <svg class="ic-xs" viewBox="0 0 24 24"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
        <span>Save Rule</span>
      `;
    }
  }
}

function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ============================================================================
// Event Listeners & Keyboard Shortcuts
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initApp();

  // Back to Dashboard button
  const backToDashBtn = document.getElementById('back-to-activities-btn');
  if (backToDashBtn) {
    backToDashBtn.onclick = showDashboardView;
  }

  // Navigation to Statistics & Moodle Sync (Dashboard only)
  const dashToStatsBtn = document.getElementById('dash-to-stats-btn');
  if (dashToStatsBtn) {
    dashToStatsBtn.onclick = showSyncStatsView;
  }

  const statsBackBtn = document.getElementById('stats-back-to-dash-btn');
  if (statsBackBtn) {
    statsBackBtn.onclick = showDashboardView;
  }

  const statsRefreshBtn = document.getElementById('stats-refresh-btn');
  if (statsRefreshBtn) {
    statsRefreshBtn.onclick = async () => {
      await refreshActivitiesData(true);
      renderSyncStats();
    };
  }

  const statsThemeBtn = document.getElementById('stats-theme-toggle-btn');
  if (statsThemeBtn) {
    statsThemeBtn.onclick = () => applyTheme(state.theme === 'dark' ? 'light' : 'dark');
  }

  // Moodle Action Buttons: Action 1 (Pull & Review) and Action 2 (Push to Moodle)
  const btnMoodlePull = document.getElementById('btn-moodle-pull');
  if (btnMoodlePull) {
    btnMoodlePull.onclick = async () => {
      btnMoodlePull.classList.add('is-loading');
      const origHtml = btnMoodlePull.innerHTML;
      btnMoodlePull.innerHTML = `
        <svg class="ic spin" viewBox="0 0 24 24"><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/></svg>
        <span>Ingesting & Reviewing...</span>
      `;
      try {
        const targetActivity = state.activeActivity || (state.activitiesOverview && state.activitiesOverview.length > 0 ? state.activitiesOverview[0].name : 'System Identification');
        const startRes = await API.moodlePull({ activity_name: targetActivity });
        showToast(startRes.message || 'Moodle ingestion & AI review started in background...', 'info');

        const pollInterval = setInterval(async () => {
          try {
            const statusData = await API.getMoodleJobStatus();
            const pullJob = statusData.pull || {};
            if (pullJob.status === 'completed') {
              clearInterval(pollInterval);
              btnMoodlePull.classList.remove('is-loading');
              btnMoodlePull.innerHTML = origHtml;
              showToast(pullJob.message || 'Pipeline complete! Submissions synced to Google Sheet.', 'success');
              await refreshActivitiesData(true);
              renderSyncStats();
            } else if (pullJob.status === 'failed') {
              clearInterval(pollInterval);
              btnMoodlePull.classList.remove('is-loading');
              btnMoodlePull.innerHTML = origHtml;
              showToast('Pull failed: ' + (pullJob.error || pullJob.message), 'error');
            }
          } catch (pollErr) {
            console.warn('Poll error:', pollErr);
          }
        }, 3000);
      } catch (err) {
        btnMoodlePull.classList.remove('is-loading');
        btnMoodlePull.innerHTML = origHtml;
        showToast('Moodle Pull error: ' + err.message, 'error');
      }
    };
  }

  const btnMoodlePush = document.getElementById('btn-moodle-push');
  if (btnMoodlePush) {
    btnMoodlePush.onclick = async () => {
      btnMoodlePush.classList.add('is-loading');
      const origHtml = btnMoodlePush.innerHTML;
      btnMoodlePush.innerHTML = `
        <svg class="ic spin" viewBox="0 0 24 24"><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/></svg>
        <span>Publishing to Moodle...</span>
      `;
      try {
        const targetActivity = state.activeActivity || (state.activitiesOverview && state.activitiesOverview.length > 0 ? state.activitiesOverview[0].name : 'System Identification');
        const startRes = await API.moodlePush({ activity_name: targetActivity, only_reviewed: false });
        showToast(startRes.message || 'Moodle gradebook upload started in background...', 'info');

        const pollInterval = setInterval(async () => {
          try {
            const statusData = await API.getMoodleJobStatus();
            const pushJob = statusData.push || {};
            if (pushJob.status === 'completed') {
              clearInterval(pollInterval);
              btnMoodlePush.classList.remove('is-loading');
              btnMoodlePush.innerHTML = origHtml;
              showToast(pushJob.message || 'Approved grades published to Moodle Gradebook!', 'success');
              await refreshActivitiesData(true);
              renderSyncStats();
            } else if (pushJob.status === 'failed') {
              clearInterval(pollInterval);
              btnMoodlePush.classList.remove('is-loading');
              btnMoodlePush.innerHTML = origHtml;
              showToast('Push failed: ' + (pushJob.error || pushJob.message), 'error');
            }
          } catch (pollErr) {
            console.warn('Poll error:', pollErr);
          }
        }, 3000);
      } catch (err) {
        btnMoodlePush.classList.remove('is-loading');
        btnMoodlePush.innerHTML = origHtml;
        showToast('Moodle Push error: ' + err.message, 'error');
      }
    };
  }

  // Dashboard controls
  const dashRefreshBtn = document.getElementById('dash-refresh-btn');
  if (dashRefreshBtn) {
    dashRefreshBtn.onclick = () => refreshActivitiesData(true);
  }

  const dashThemeBtn = document.getElementById('dash-theme-toggle-btn');
  if (dashThemeBtn) {
    dashThemeBtn.onclick = () => applyTheme(state.theme === 'dark' ? 'light' : 'dark');
  }

  const dashSearch = document.getElementById('dash-activity-search');
  if (dashSearch) {
    dashSearch.oninput = (e) => {
      state.dashboardSearchQuery = e.target.value;
      renderDashboard();
    };
  }

  document.querySelectorAll('.dash-tab').forEach((tab) => {
    tab.onclick = () => {
      document.querySelectorAll('.dash-tab').forEach((t) => t.classList.remove('is-active'));
      tab.classList.add('is-active');
      state.dashboardFilter = tab.dataset.filter || 'all';
      renderDashboard();
    };
  });

  // Activity change
  document.getElementById('activity-select').onchange = (e) => {
    selectActivity(e.target.value);
  };

  // Search input
  const searchInput = document.getElementById('roster-search');
  searchInput.oninput = (e) => {
    state.searchQuery = e.target.value;
    applyFilterAndRender();
  };

  // Filter pills
  document.querySelectorAll('.filter-pill').forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll('.filter-pill').forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      state.activeFilter = btn.dataset.filter;
      state.selectedIndex = 0;
      applyFilterAndRender();
    };
  });

  // Theme switch
  document.getElementById('theme-toggle-btn').onclick = () => {
    applyTheme(state.theme === 'dark' ? 'light' : 'dark');
  };

  // Sidebar collapse toggles (Side banner click uncollapses, toolbar button collapses)
  const collapsedBanner = document.getElementById('roster-collapsed-banner');
  if (collapsedBanner) {
    collapsedBanner.onclick = () => applySidebarState(false);
  }
  const sidebarCollapseBtn = document.getElementById('sidebar-collapse-btn');
  if (sidebarCollapseBtn) {
    sidebarCollapseBtn.onclick = () => applySidebarState(true);
  }

  // AI Evaluation panel collapse toggles
  const evalCollapsedBanner = document.getElementById('eval-collapsed-banner');
  if (evalCollapsedBanner) {
    evalCollapsedBanner.onclick = () => applyEvalSidebarState(false);
  }
  const evalCollapseBtn = document.getElementById('eval-collapse-btn');
  if (evalCollapseBtn) {
    evalCollapseBtn.onclick = () => applyEvalSidebarState(true);
  }

  // Media carousel navigation
  const mediaPrevBtn = document.getElementById('media-nav-prev-btn');
  if (mediaPrevBtn) mediaPrevBtn.onclick = () => navigateMedia(-1);
  const mediaNextBtn = document.getElementById('media-nav-next-btn');
  if (mediaNextBtn) mediaNextBtn.onclick = () => navigateMedia(1);

  // Grade buttons
  document.getElementById('btn-grade-pass').onclick = () => setStageGrade('Pass');
  document.getElementById('btn-grade-fail').onclick = () => setStageGrade('Fail');

  // Feedback input
  document.getElementById('stage-feedback-input').oninput = (e) => {
    onStageFeedbackInput(e.target.value);
  };

  // Revert button
  document.getElementById('stage-revert-btn').onclick = revertStageFeedback;

  // Copy text button (if present)
  const copyBtn = document.getElementById('copy-text-btn');
  if (copyBtn) copyBtn.onclick = copySubmissionText;

  // Stage Navigation buttons
  document.getElementById('prev-student-btn').onclick = () => navigateRoster(-1);
  document.getElementById('next-student-btn').onclick = () => navigateRoster(1);

  // In-Roster Mini-Nav buttons
  const rosterPrevBtn = document.getElementById('roster-prev-btn');
  if (rosterPrevBtn) rosterPrevBtn.onclick = () => navigateRoster(-1);
  const rosterNextBtn = document.getElementById('roster-next-btn');
  if (rosterNextBtn) rosterNextBtn.onclick = () => navigateRoster(1);

  // Submit buttons
  document.getElementById('stage-approve-next-btn').onclick = () => submitStageReview('Approved by Mentor', true);
  document.getElementById('stage-override-btn').onclick = () => submitStageReview('Overridden by Mentor', false);

  // Header batch actions
  document.getElementById('batch-approve-btn').onclick = triggerBatchApprove;
  document.getElementById('sync-local-btn').onclick = triggerSyncLocal;

  // Activity Rule Prompt Modal listeners
  const modalCloseBtn = document.getElementById('rule-modal-close-btn');
  if (modalCloseBtn) modalCloseBtn.onclick = closeRulePromptModal;
  const modalBackdrop = document.getElementById('rule-modal-backdrop');
  if (modalBackdrop) modalBackdrop.onclick = closeRulePromptModal;
  const modalSkipBtn = document.getElementById('rule-modal-skip-btn');
  if (modalSkipBtn) modalSkipBtn.onclick = closeRulePromptModal;
  const modalSaveBtn = document.getElementById('rule-modal-save-btn');
  if (modalSaveBtn) modalSaveBtn.onclick = saveActivityRuleFromModal;

  const manualAddRuleBtn = document.getElementById('btn-manual-add-rule');
  if (manualAddRuleBtn) {
    manualAddRuleBtn.onclick = () => {
      const sub = state.filteredSubmissions[state.selectedIndex];
      const g = sub ? (state.cardStates[`${sub.name}_${sub.attempt_number}`]?.grade || (sub.grade.toLowerCase().includes('pass') ? 'Pass' : 'Fail')) : 'Pass';
      openRulePromptModal(g, g, true);
    };
  }

  const tabBoth = document.getElementById('tab-btn-both');
  if (tabBoth) tabBoth.onclick = () => setRuleTab('both');
  const tabEdge = document.getElementById('tab-btn-edge');
  if (tabEdge) tabEdge.onclick = () => setRuleTab('edge');
  const tabGuide = document.getElementById('tab-btn-guide');
  if (tabGuide) tabGuide.onclick = () => setRuleTab('guide');

  const bannerToggle = document.getElementById('existing-rules-banner-toggle');
  if (bannerToggle) bannerToggle.onclick = toggleExistingRulesBanner;

  // GLOBAL KEYBOARD SHORTCUTS (The Linear / Superhuman experience)
  document.addEventListener('keydown', (e) => {
    // If modal is open, Escape closes modal
    if (e.key === 'Escape') {
      const modal = document.getElementById('rule-prompt-modal');
      if (modal && modal.style.display !== 'none') {
        e.preventDefault();
        closeRulePromptModal();
        return;
      }
    }

    const isTyping = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA';

    // Search focus on '/'
    if (e.key === '/' && !isTyping) {
      e.preventDefault();
      searchInput.focus();
      return;
    }

    // Escape exits search or feedback
    if (e.key === 'Escape' && isTyping) {
      e.target.blur();
      return;
    }

    // Cmd+Enter or Ctrl+Enter in feedback -> Approve & Next
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      submitStageReview('Approved by Mentor', true);
      return;
    }

    if (!isTyping) {
      if (e.key === '[' || ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'b' || e.key === '\\'))) {
        e.preventDefault();
        applySidebarState(!state.isSidebarCollapsed);
        return;
      }

      if (e.key === ']') {
        e.preventDefault();
        applyEvalSidebarState(!state.isEvalCollapsed);
        return;
      }

      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        navigateRoster(1);
      } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        navigateRoster(-1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        navigateMedia(-1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        navigateMedia(1);
      } else if (e.key.toLowerCase() === 'p') {
        e.preventDefault();
        setStageGrade('Pass');
      } else if (e.key.toLowerCase() === 'f') {
        e.preventDefault();
        setStageGrade('Fail');
      }
    }
  });
});
