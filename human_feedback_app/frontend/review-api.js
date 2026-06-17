window.HFApi = (function () {
  async function request(path, options) {
    const res = await fetch(path, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", ...(options && options.headers) },
      ...options,
    });
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (_) {}
    if (!res.ok) {
      const detail = data && (data.detail || data.message);
      throw new Error(typeof detail === "string" ? detail : text || res.statusText);
    }
    return data;
  }

  return {
    loadSlides() {
      return request("/api/slides");
    },
    approve(payload) {
      return request("/api/visuals/approve", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },
    selectPool(payload) {
      return request("/api/visuals/select-pool", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },
    revise(payload) {
      return request("/api/visuals/revise", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },
    getJob(jobId) {
      return request("/api/jobs/" + encodeURIComponent(jobId));
    },
    listJobs() {
      return request("/api/jobs");
    },
  };
})();
