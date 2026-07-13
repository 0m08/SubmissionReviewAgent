window.HFYoutube = (function () {
  const players = {};
  const playerRoots = {};
  const mountInflight = {};
  let apiLoading = false;
  let apiReady = false;
  const readyQueue = [];

  function firstHttpUrl(text) {
    if (!text) return "";
    const match = String(text).trim().match(/https?:\/\/[^\s)>"]+/);
    return match ? match[0].replace(/[.,);"']+$/, "") : String(text).trim();
  }

  function parseYoutubeEmbed(url) {
    const raw = firstHttpUrl(url);
    if (!raw) return null;
    let parsed;
    try {
      parsed = new URL(raw);
    } catch (_) {
      return null;
    }

    let videoId = "";
    const host = parsed.hostname.toLowerCase();
    const path = parsed.pathname || "";

    if (host.includes("youtube.com") && path.includes("/embed/")) {
      videoId = path.split("/embed/")[1].split("/")[0];
    } else if (host.includes("youtu.be")) {
      videoId = path.replace(/^\//, "").split("/")[0];
    } else if (host.includes("youtube.com") && path === "/watch") {
      videoId = parsed.searchParams.get("v") || "";
    }

    if (!videoId) return null;

    let start = parsed.searchParams.get("start");
    if (start == null) {
      const t = parsed.searchParams.get("t");
      if (t) {
        const m = String(t).match(/^(\d+)/);
        start = m ? m[1] : "0";
      } else {
        start = "0";
      }
    }

    const end = parsed.searchParams.get("end");
    let startVal = 0;
    let endVal = null;
    try {
      startVal = parseFloat(start);
      if (Number.isNaN(startVal)) startVal = 0;
    } catch (_) {
      startVal = 0;
    }
    if (end != null && end !== "") {
      try {
        endVal = parseFloat(end);
        if (Number.isNaN(endVal)) endVal = null;
      } catch (_) {
        endVal = null;
      }
    }

    return { videoId, start: startVal, end: endVal };
  }

  function isYoutubeUrl(url) {
    const meta = parseYoutubeEmbed(url);
    return !!meta;
  }

  function ensureApi(callback) {
    if (window.YT && window.YT.Player) {
      apiReady = true;
      callback();
      return;
    }
    readyQueue.push(callback);
    if (apiLoading) return;
    apiLoading = true;

    const prev = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = function () {
      apiReady = true;
      if (typeof prev === "function") prev();
      const queue = readyQueue.splice(0, readyQueue.length);
      queue.forEach((fn) => {
        try {
          fn();
        } catch (_) {}
      });
    };

    if (!document.getElementById("yt-iframe-api-script")) {
      const tag = document.createElement("script");
      tag.id = "yt-iframe-api-script";
      tag.src = "https://www.youtube.com/iframe_api";
      document.body.appendChild(tag);
    }
  }

  function destroyPlayer(mountId) {
    delete mountInflight[mountId];
    const player = players[mountId];
    const root = playerRoots[mountId];
    if (root && root.parentNode) {
      try { root.parentNode.removeChild(root); } catch (_) {}
    }
    delete playerRoots[mountId];
    if (!player) return;
    try {
      if (player._checkInterval) {
        clearInterval(player._checkInterval);
      }
      player.destroy();
    } catch (_) {}
    delete players[mountId];
  }

  function destroyAll() {
    Object.keys(players).forEach(destroyPlayer);
    Object.keys(playerRoots).forEach(function (id) {
      delete playerRoots[id];
    });
  }

  function reattachPlayer(mountId) {
    const el = document.getElementById(mountId);
    const root = playerRoots[mountId];
    if (!el || !root || !players[mountId]) return false;
    if (root.parentNode === el && el.contains(root)) {
      if (!el.getAttribute("data-yt-url") && root.getAttribute("data-yt-url")) {
        el.setAttribute("data-yt-url", root.getAttribute("data-yt-url"));
      }
      return true;
    }
    try {
      while (el.firstChild) el.removeChild(el.firstChild);
      el.appendChild(root);
      const url = root.getAttribute("data-yt-url");
      if (url) el.setAttribute("data-yt-url", url);
      return true;
    } catch (_) {
      return false;
    }
  }

  function reattachAll() {
    Object.keys(playerRoots).forEach(reattachPlayer);
  }

  function cueOrLoadVideo(player, meta, play) {
    if (!player || !meta) return false;
    const payload = {
      videoId: meta.videoId,
      startSeconds: meta.start || 0,
    };
    if (meta.end != null) payload.endSeconds = meta.end;
    try {
      if (play && typeof player.loadVideoById === "function") {
        player.loadVideoById(payload);
        return true;
      }
      if (typeof player.cueVideoById === "function") {
        player.cueVideoById(payload);
        if (play && typeof player.playVideo === "function") player.playVideo();
        return true;
      }
    } catch (_) {}
    return false;
  }

  function mount(mountId, url, height) {
    const meta = parseYoutubeEmbed(url);
    if (!meta) return false;

    const el = document.getElementById(mountId);
    if (!el) return false;

    if (players[mountId]) {
      const existingUrl = el.getAttribute("data-yt-url");
      if (existingUrl === url) return true;
      destroyPlayer(mountId);
    }

    el.setAttribute("data-yt-url", url);
    el.innerHTML = "";

    const isSmall = height < 150;
    const scale = isSmall ? 2.5 : 1.5;
    const invScale = 1 / scale;

    const playerWrapper = document.createElement("div");
    playerWrapper.style.width = (scale * 100) + "%";
    playerWrapper.style.height = (scale * 100) + "%";
    playerWrapper.style.transform = "scale(" + invScale + ")";
    playerWrapper.style.transformOrigin = "top left";
    playerWrapper.style.position = "absolute";
    playerWrapper.style.top = "0";
    playerWrapper.style.left = "0";
    
    el.appendChild(playerWrapper);

    const playerHost = document.createElement("div");
    playerHost.id = mountId + "-player";
    playerHost.style.width = "100%";
    playerHost.style.height = "100%";
    playerWrapper.appendChild(playerHost);

    const start = Math.floor(meta.start || 0);
    const end = meta.end != null ? Math.floor(meta.end) : null;

    ensureApi(function () {
      const host = document.getElementById(mountId + "-player");
      if (!host) return;

      let parsedUrl = null;
      try { parsedUrl = new URL(url); } catch (_) {}
      
      const urlControls = parsedUrl ? parsedUrl.searchParams.get("controls") : null;

      const playerVars = {
        start: start,
        autoplay: 1,
        controls: urlControls !== null ? parseInt(urlControls) : 1,
        rel: 0,
        playsinline: 1,
        mute: 1,
        enablejsapi: 1,
        modestbranding: 1,
        disablekb: (parsedUrl && parsedUrl.searchParams.get("disablekb") === "1") ? 1 : 0,
      };
      if (end != null) playerVars.end = end;

      let checkInterval = null;
      function startChecking(p) {
        if (end == null) return;
        if (checkInterval) return;
        checkInterval = setInterval(function () {
          try {
            if (p && typeof p.getCurrentTime === "function" && typeof p.seekTo === "function") {
              const cur = p.getCurrentTime();
              if (cur >= end - 0.2) {
                p.seekTo(start, true);
                p.playVideo();
              }
            }
          } catch (_) {}
        }, 150);
        p._checkInterval = checkInterval;
      }
      function stopChecking(p) {
        if (checkInterval) {
          clearInterval(checkInterval);
          checkInterval = null;
          if (p) p._checkInterval = null;
        }
      }

      const player = new window.YT.Player(mountId + "-player", {
        height: String(Math.round(height * scale) || 176),
        width: "100%",
        videoId: meta.videoId,
        playerVars: playerVars,
        events: {
          onReady: function (event) {
            try {
              event.target.mute();
              event.target.playVideo();
            } catch (_) {}
          },
          onStateChange: function (event) {
            try {
              if (event.data === window.YT.PlayerState.PLAYING) {
                const container = document.getElementById(mountId);
                if (container) {
                  container.classList.add("is-playing");
                }
                startChecking(event.target);
              } else {
                stopChecking(event.target);
              }
            } catch (_) {}
            try {
              if (event.data === window.YT.PlayerState.ENDED) {
                event.target.seekTo(start, true);
                event.target.playVideo();
              }
            } catch (_) {}
          },
        },
      });
      players[mountId] = player;
    });

    return true;
  }

  function mountWithRetry(mountId, url, height, attempt) {
    const el = document.getElementById(mountId);
    if (!el) {
      if ((attempt || 0) < 30) {
        setTimeout(function () {
          mountWithRetry(mountId, url, height, (attempt || 0) + 1);
        }, 50);
      }
      return;
    }
    mount(mountId, url, height);
  }

  function cleanOrphanedPlayers() {
    Object.keys(players).forEach(function (mountId) {
      if (!document.getElementById(mountId)) {
        destroyPlayer(mountId);
      }
    });
  }

  function scheduleMount(mountId, url, height) {
    cleanOrphanedPlayers();
    window.requestAnimationFrame(function () {
      mountWithRetry(mountId, url, height, 0);
    });
  }

  function endTimeForUrl(url) {
    const meta = parseYoutubeEmbed(url);
    if (!meta) return 0;
    if (meta.end != null) return Math.max(meta.start || 0, meta.end - 0.12);
    return meta.start || 0;
  }

  function freezePlayerAtEnd(mountId, url) {
    const p = players[mountId];
    if (!p) return;
    const t = endTimeForUrl(url);
    try {
      if (typeof p.seekTo === "function") p.seekTo(t, true);
      if (typeof p.pauseVideo === "function") p.pauseVideo();
    } catch (_) {}
  }

  function mountControlled(mountId, url, height, opts) {
    opts = opts || {};
    const meta = parseYoutubeEmbed(url);
    if (!meta) return false;

    const el = document.getElementById(mountId);
    if (!el) return false;

    if (players[mountId]) {
      reattachPlayer(mountId);
      const existingUrl = (el.getAttribute("data-yt-url") || (playerRoots[mountId] && playerRoots[mountId].getAttribute("data-yt-url")) || "");
      if (existingUrl === url && isPlayerMounted(mountId)) {
        if (opts.seekTo != null) {
          try {
            const p = players[mountId];
            if (p && typeof p.seekTo === "function") p.seekTo(opts.seekTo, true);
            if (!opts.autoplay && p && typeof p.pauseVideo === "function") p.pauseVideo();
            if (opts.autoplay && p && typeof p.playVideo === "function") p.playVideo();
          } catch (_) {}
        }
        if (typeof opts.onReady === "function") {
          try { opts.onReady(players[mountId]); } catch (_) {}
        }
        return true;
      }
      // Same player host, different clip — swap via API (avoids 2s remount).
      if (isPlayerMounted(mountId) && cueOrLoadVideo(players[mountId], meta, !!opts.autoplay)) {
        el.setAttribute("data-yt-url", url);
        if (playerRoots[mountId]) playerRoots[mountId].setAttribute("data-yt-url", url);
        if (!opts.autoplay) {
          try {
            const p = players[mountId];
            if (opts.seekTo != null && p && typeof p.seekTo === "function") p.seekTo(opts.seekTo, true);
            if (p && typeof p.pauseVideo === "function") p.pauseVideo();
          } catch (_) {}
        }
        if (typeof opts.onReady === "function") {
          try { opts.onReady(players[mountId]); } catch (_) {}
        }
        return true;
      }
      destroyPlayer(mountId);
    }

    el.setAttribute("data-yt-url", url);
    while (el.firstChild) el.removeChild(el.firstChild);
    try {
      el.style.backgroundImage = "url(https://img.youtube.com/vi/" + meta.videoId + "/hqdefault.jpg)";
      el.style.backgroundSize = "cover";
      el.style.backgroundPosition = "center";
    } catch (_) {}

    const coverScale = opts.cover ? 1.65 : 1.0;
    const cropScaleX = 1.12;
    const cropScaleY = 1.22;

    const playerWrapper = document.createElement("div");
    playerWrapper.setAttribute("data-yt-url", url);
    playerWrapper.style.width = (coverScale * cropScaleX * 100) + "%";
    playerWrapper.style.height = (coverScale * cropScaleY * 100) + "%";
    playerWrapper.style.position = "absolute";
    playerWrapper.style.left = "50%";
    playerWrapper.style.top = "50%";
    playerWrapper.style.transform = "translate(-50%, -50%) scale(" + (1 / coverScale) + ")";
    playerWrapper.style.transformOrigin = "center center";
    el.style.overflow = "hidden";
    el.style.position = el.style.position || "relative";
    el.appendChild(playerWrapper);
    playerRoots[mountId] = playerWrapper;

    const playerHost = document.createElement("div");
    playerHost.id = mountId + "-player";
    playerHost.style.width = "100%";
    playerHost.style.height = "100%";
    playerWrapper.appendChild(playerHost);

    const start = Math.floor(meta.start || 0);
    const end = meta.end != null ? Math.floor(meta.end) : null;

    ensureApi(function () {
      if (!document.getElementById(mountId + "-player")) {
        // Host was wiped before API ready — reattach root then continue if possible.
        if (!reattachPlayer(mountId) || !document.getElementById(mountId + "-player")) return;
      }

      const playerVars = {
        start: start,
        autoplay: opts.autoplay ? 1 : 0,
        controls: 0,
        rel: 0,
        playsinline: 1,
        mute: 1,
        enablejsapi: 1,
        modestbranding: 1,
        disablekb: 1,
        cc_load_policy: 0,
        iv_load_policy: 3,
      };
      if (end != null) playerVars.end = end;

      const player = new window.YT.Player(mountId + "-player", {
        height: String(Math.round(height * coverScale * cropScaleY) || 480),
        width: "100%",
        videoId: meta.videoId,
        playerVars: playerVars,
        events: {
          onReady: function (event) {
            try {
              event.target.mute();
            } catch (_) {}
            if (opts.seekTo != null) {
              try {
                event.target.seekTo(opts.seekTo, true);
              } catch (_) {}
            }
            try {
              if (opts.autoplay) event.target.playVideo();
              else event.target.pauseVideo();
            } catch (_) {}
            try {
              if (typeof event.target.unloadModule === "function") {
                event.target.unloadModule("captions");
                event.target.unloadModule("cc");
              }
            } catch (_) {}
            if (typeof opts.onReady === "function") {
              try { opts.onReady(event.target); } catch (_) {}
            }
          },
          onStateChange: function (event) {
            if (!opts.loop && event.data === window.YT.PlayerState.ENDED) {
              try {
                const currentUrl = (playerRoots[mountId] && playerRoots[mountId].getAttribute("data-yt-url")) || url;
                const holdAt = endTimeForUrl(currentUrl);
                event.target.seekTo(holdAt, true);
                event.target.pauseVideo();
              } catch (_) {}
            }
          },
        },
      });
      players[mountId] = player;
    });

    return true;
  }

  function scheduleControlledMount(mountId, url, height, opts) {
    cleanOrphanedPlayers();
    window.requestAnimationFrame(function () {
      mountWithRetryControlled(mountId, url, height, opts, 0);
    });
  }

  function mountWithRetryControlled(mountId, url, height, opts, attempt) {
    const el = document.getElementById(mountId);
    if (!el) {
      if ((attempt || 0) < 60) {
        setTimeout(function () {
          mountWithRetryControlled(mountId, url, height, opts, (attempt || 0) + 1);
        }, 50);
      } else if (opts && typeof opts.onMountFailed === "function") {
        opts.onMountFailed();
      }
      return;
    }
    mountControlled(mountId, url, height, opts);
  }

  function isPlayerMounted(mountId) {
    const el = document.getElementById(mountId);
    if (!el) return false;
    if (playerRoots[mountId] && players[mountId]) {
      if (playerRoots[mountId].parentNode !== el) reattachPlayer(mountId);
    }
    const host = document.getElementById(mountId + "-player");
    return !!(host && el.contains(host));
  }

  function scheduleControlledMountAsync(mountId, url, height, opts) {
    opts = opts || {};
    reattachPlayer(mountId);
    const el = document.getElementById(mountId);
    const existingUrl = el
      ? (el.getAttribute("data-yt-url") || (playerRoots[mountId] && playerRoots[mountId].getAttribute("data-yt-url")) || "")
      : "";
    if (el && players[mountId] && existingUrl === url && isPlayerMounted(mountId)) {
      if (opts.seekTo != null) {
        try {
          const p = players[mountId];
          if (p && typeof p.seekTo === "function") p.seekTo(opts.seekTo, true);
          if (!opts.autoplay && p && typeof p.pauseVideo === "function") p.pauseVideo();
        } catch (_) {}
      }
      return Promise.resolve();
    }
    const inflight = mountInflight[mountId];
    if (inflight && inflight.url === url) return inflight.promise;
    // If a player already exists, mountControlled will reattach/swap instead of full remount.
    const promise = new Promise(function (resolve, reject) {
      const timeoutMs = opts.timeoutMs != null ? opts.timeoutMs : 6000;
      let settled = false;
      function finish(ok) {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (mountInflight[mountId] && mountInflight[mountId].promise === promise) {
          delete mountInflight[mountId];
        }
        if (ok) resolve();
        else reject(new Error("YouTube mount failed"));
      }
      const timer = setTimeout(function () { finish(false); }, timeoutMs);
      const merged = Object.assign({}, opts, {
        onReady: function () {
          if (typeof opts.onReady === "function") {
            try { opts.onReady(); } catch (_) {}
          }
          finish(true);
        },
        onMountFailed: function () { finish(false); },
      });
      window.requestAnimationFrame(function () {
        mountWithRetryControlled(mountId, url, height, merged, 0);
      });
    });
    mountInflight[mountId] = { url: url, promise: promise };
    return promise;
  }

  function mountFrozenAtEndAsync(mountId, url, height, opts) {
    opts = opts || {};
    reattachPlayer(mountId);
    const el = document.getElementById(mountId);
    const existingUrl = el
      ? (el.getAttribute("data-yt-url") || (playerRoots[mountId] && playerRoots[mountId].getAttribute("data-yt-url")) || "")
      : "";
    if (el && players[mountId] && existingUrl === url && isPlayerMounted(mountId)) {
      freezePlayerAtEnd(mountId, url);
      return Promise.resolve();
    }
    return scheduleControlledMountAsync(mountId, url, height, {
      autoplay: false,
      loop: false,
      cover: opts.cover !== false,
      seekTo: endTimeForUrl(url),
      timeoutMs: opts.timeoutMs,
    });
  }

  function mountPausedAtStartAsync(mountId, url, height, opts) {
    opts = opts || {};
    reattachPlayer(mountId);
    const el = document.getElementById(mountId);
    const existingUrl = el
      ? (el.getAttribute("data-yt-url") || (playerRoots[mountId] && playerRoots[mountId].getAttribute("data-yt-url")) || "")
      : "";
    if (el && players[mountId] && existingUrl === url && isPlayerMounted(mountId)) {
      return Promise.resolve();
    }
    const inflight = mountInflight[mountId];
    if (inflight && inflight.url === url) return inflight.promise;
    const meta = parseYoutubeEmbed(url);
    const start = meta ? (meta.start || 0) : 0;
    return scheduleControlledMountAsync(mountId, url, height, {
      autoplay: false,
      loop: false,
      cover: opts.cover !== false,
      seekTo: start,
      timeoutMs: opts.timeoutMs,
    });
  }

  function playPlayer(mountId, url) {
    reattachPlayer(mountId);
    const p = players[mountId];
    if (!p || typeof p.playVideo !== "function") return false;
    if (!isPlayerMounted(mountId)) return false;
    const meta = url ? parseYoutubeEmbed(url) : null;
    const el = document.getElementById(mountId);
    const existingUrl = el
      ? (el.getAttribute("data-yt-url") || (playerRoots[mountId] && playerRoots[mountId].getAttribute("data-yt-url")) || "")
      : "";
    try { if (typeof p.mute === "function") p.mute(); } catch (_) {}
    try {
      if (meta && existingUrl && existingUrl !== url) {
        if (!cueOrLoadVideo(p, meta, true)) return false;
        if (el) el.setAttribute("data-yt-url", url);
        if (playerRoots[mountId]) playerRoots[mountId].setAttribute("data-yt-url", url);
        return true;
      }
      const start = meta ? (meta.start || 0) : 0;
      if (typeof p.seekTo === "function") p.seekTo(start, true);
      p.playVideo();
      return true;
    } catch (_) {
      return false;
    }
  }

  function isReady(mountId, url) {
    reattachPlayer(mountId);
    const el = document.getElementById(mountId);
    if (!el || !players[mountId] || !isPlayerMounted(mountId)) return false;
    if (url) {
      const existingUrl = el.getAttribute("data-yt-url") || (playerRoots[mountId] && playerRoots[mountId].getAttribute("data-yt-url")) || "";
      if (existingUrl !== url) return false;
    }
    return true;
  }

  function pausePlayer(mountId) {
    reattachPlayer(mountId);
    const p = players[mountId];
    if (p && typeof p.pauseVideo === "function") {
      try { p.pauseVideo(); } catch (_) {}
    }
  }

  return {
    parseYoutubeEmbed,
    isYoutubeUrl,
    mount,
    mountControlled,
    scheduleMount,
    scheduleControlledMount,
    scheduleControlledMountAsync,
    mountFrozenAtEndAsync,
    mountPausedAtStartAsync,
    freezePlayerAtEnd,
    endTimeForUrl,
    playPlayer,
    pausePlayer,
    isReady,
    reattachPlayer,
    reattachAll,
    destroyAll,
    destroyPlayer,
  };
})();