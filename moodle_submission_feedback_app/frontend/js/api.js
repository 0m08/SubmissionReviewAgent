// API client for Moodle Submission Reviewer
async function requestWithAuth(url, options = {}) {
  const res = await fetch(url, options);
  if (res.status === 401) {
    window.location.href = '/login-page';
    throw new Error('Session expired. Redirecting to login...');
  }
  return res;
}

const API = {
  async getMentorInfo() {
    const res = await requestWithAuth('/api/me');
    if (!res.ok) throw new Error('Failed to fetch mentor info');
    return res.json();
  },

  async getActivities() {
    const res = await requestWithAuth('/api/activities');
    if (!res.ok) throw new Error('Failed to fetch activities');
    return res.json();
  },

  async getActivityDetails(activityName) {
    const res = await requestWithAuth(`/api/activity/${encodeURIComponent(activityName)}`);
    if (!res.ok) throw new Error(`Failed to load activity: ${activityName}`);
    return res.json();
  },

  async submitReview(activityName, payload) {
    const res = await requestWithAuth(`/api/activity/${encodeURIComponent(activityName)}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to update review' }));
      throw new Error(err.detail || 'Failed to update review');
    }
    return res.json();
  },

  async batchApprove(activityName, mentorName) {
    const res = await requestWithAuth(`/api/activity/${encodeURIComponent(activityName)}/batch-approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mentor_name: mentorName }),
    });
    if (!res.ok) throw new Error('Failed to batch approve submissions');
    return res.json();
  },

  async syncLocal(activityName, assignmentFolder = null) {
    const res = await requestWithAuth(`/api/activity/${encodeURIComponent(activityName)}/sync-local`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        activity_name: activityName,
        assignment_folder: assignmentFolder,
      }),
    });
    if (!res.ok) throw new Error('Failed to sync local assignment folder');
    return res.json();
  },

  getExportUrl(activityName) {
    return `/api/activity/${encodeURIComponent(activityName)}/export`;
  },

  async getSubmissionMedia(mediaFolder, studentName) {
    const params = new URLSearchParams();
    if (mediaFolder) params.set('media_folder', mediaFolder);
    if (studentName) params.set('student_name', studentName);
    const res = await requestWithAuth(`/api/submission-media?${params.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch submission media');
    return res.json();
  },

  async addActivityRule(activityName, payload) {
    const res = await requestWithAuth(`/api/activity/${encodeURIComponent(activityName)}/rule`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to add activity rule' }));
      throw new Error(err.detail || 'Failed to add activity rule');
    }
    return res.json();
  },
};
