window.HFYoutube = (function () {
  const players = {};
  const playerRoots = {};
  const mountInflight = {};
  const preloadByUrl = {};
  let apiLoading = false;
  let apiReady = false;
  const readyQueue = [];
  const VIDEO_ASPECT = 16 / 9;

  function firstHttpUrl(text) {
    if (!text) return "";
    const normalized = String(text).trim()
      .replace(/&amp;/gi, "&")
      .replace(/%26/gi, "&")
      .replace(/%3D/gi, "=");
    const match = normalized.match(/https?:\/\/[^\s)>"]+/);
    return match ? match[0].replace(/[.,);"']+$/, "") : normalized;
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

  function hashText(text) {
    const raw = String(text || "");
    let hash = 2166136261;
    for (let i = 0; i < raw.length; i++) {
      hash ^= raw.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(36);
  }

  function preloadKey(url) {
    return firstHttpUrl(url);
  }

  function normalizeFit(opts) {
    opts = opts || {};
    if (opts.fit === "contain" || opts.cover === false) return "contain";
    return "cover";
  }

  function ensurePreloadRoot() {
    let root = document.getElementById("hf-youtube-preload-root");
    if (root) return root;
    root = document.createElement("div");
    root.id = "hf-youtube-preload-root";
    root.setAttribute("aria-hidden", "true");
    root.style.position = "fixed";
    root.style.left = "-10000px";
    root.style.top = "-10000px";
    root.style.width = "320px";
    root.style.height = "180px";
    root.style.overflow = "hidden";
    root.style.opacity = "0.01";
    root.style.pointerEvents = "none";
    root.style.zIndex = "-1";
    document.body.appendChild(root);
    return root;
  }

  function installBlurBackdrop(container, meta, fit) {
    if (!container || !meta) return;
    const old = container.querySelector(".hf-player-video-backdrop");
    if (old && old.parentNode) {
      try { old.parentNode.removeChild(old); } catch (_) {}
    }
    if (fit !== "contain") {
      container.style.backgroundImage = "";
      return;
    }
    const bg = document.createElement("div");
    bg.className = "hf-player-video-backdrop";
    bg.style.position = "absolute";
    bg.style.inset = "0";
    bg.style.backgroundImage = "url(https://img.youtube.com/vi/" + meta.videoId + "/hqdefault.jpg)";
    bg.style.backgroundSize = "cover";
    bg.style.backgroundPosition = "center";
    bg.style.filter = "blur(24px) brightness(0.58)";
    bg.style.transform = "scale(1.08)";
    bg.style.zIndex = "0";
    bg.style.pointerEvents = "none";
    container.appendChild(bg);
  }

  function stylePlayerRoot(container, root, fit) {
    if (!container || !root) return;
    const cw = Math.max(1, container.clientWidth || 0);
    const ch = Math.max(1, container.clientHeight || 0);
    const ratio = cw > 1 && ch > 1 ? (cw / ch) : VIDEO_ASPECT;
    let widthPct = 100;
    let heightPct = 100;

    if (fit === "contain") {
      if (ratio > VIDEO_ASPECT) widthPct = (VIDEO_ASPECT / ratio) * 100;
      else heightPct = (ratio / VIDEO_ASPECT) * 100;
    } else {
      if (ratio > VIDEO_ASPECT) heightPct = (ratio / VIDEO_ASPECT) * 100;
      else widthPct = (VIDEO_ASPECT / ratio) * 100;
    }

    root.style.width = widthPct + "%";
    root.style.height = heightPct + "%";
    root.style.position = "absolute";
    root.style.left = "50%";
    root.style.top = "50%";
    root.style.transform = "translate(-50%, -50%)";
    root.style.transformOrigin = "center center";
    root.style.zIndex = "2";
    root.style.pointerEvents = "none";
    root.setAttribute("data-yt-fit", fit);
  }

  function cleanYoutubeIframe(mountId) {
    let iframe = document.getElementById(mountId + "-player");
    if (!iframe && playerRoots[mountId]) {
      iframe = playerRoots[mountId].querySelector("iframe");
    }
    if (!iframe) return;
    try {
      iframe.style.width = "100%";
      iframe.style.height = "100%";
      iframe.style.border = "0";
      iframe.style.display = "block";
      iframe.style.pointerEvents = "none";
      iframe.setAttribute("tabindex", "-1");
      iframe.setAttribute("aria-hidden", "true");
    } catch (_) {}
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
    Object.keys(preloadByUrl).forEach(function (key) {
      if (preloadByUrl[key] && preloadByUrl[key].mountId === mountId) delete preloadByUrl[key];
    });
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
    Object.keys(preloadByUrl).forEach(function (key) {
      delete preloadByUrl[key];
    });
    const preloadRoot = document.getElementById("hf-youtube-preload-root");
    if (preloadRoot) preloadRoot.innerHTML = "";
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
      const fit = root.getAttribute("data-yt-fit") || "cover";
      installBlurBackdrop(el, parseYoutubeEmbed(root.getAttribute("data-yt-url") || ""), fit);
      stylePlayerRoot(el, root, fit);
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
        const fit = normalizeFit(opts);
        installBlurBackdrop(el, meta, fit);
        if (playerRoots[mountId]) {
          playerRoots[mountId].setAttribute("data-yt-fit", fit);
          stylePlayerRoot(el, playerRoots[mountId], fit);
        }
        cleanYoutubeIframe(mountId);
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
        const fit = normalizeFit(opts);
        installBlurBackdrop(el, meta, fit);
        if (playerRoots[mountId]) {
          playerRoots[mountId].setAttribute("data-yt-url", url);
          playerRoots[mountId].setAttribute("data-yt-fit", fit);
          stylePlayerRoot(el, playerRoots[mountId], fit);
        }
        cleanYoutubeIframe(mountId);
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
    const fit = normalizeFit(opts);
    installBlurBackdrop(el, meta, fit);

    const playerWrapper = document.createElement("div");
    playerWrapper.setAttribute("data-yt-url", url);
    playerWrapper.setAttribute("data-yt-fit", fit);
    stylePlayerRoot(el, playerWrapper, fit);
    el.style.overflow = "hidden";
    el.style.position = el.style.position || "relative";
    el.style.backgroundColor = "#0d0f12";
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
        fs: 0,
        cc_load_policy: 0,
        iv_load_policy: 3,
        showinfo: 0,
      };
      try {
        if (window.location && window.location.origin) playerVars.origin = window.location.origin;
      } catch (_) {}
      if (end != null) playerVars.end = end;

      const player = new window.YT.Player(mountId + "-player", {
        height: String(Math.round(height) || 480),
        width: "100%",
        videoId: meta.videoId,
        playerVars: playerVars,
        events: {
          onReady: function (event) {
            try {
              event.target.mute();
            } catch (_) {}
            cleanYoutubeIframe(mountId);
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
            const completeReady = function () {
              try {
                if (!opts.autoplay && typeof event.target.pauseVideo === "function") event.target.pauseVideo();
                if (!opts.autoplay && typeof event.target.seekTo === "function") {
                  const holdAt = opts.seekTo != null ? opts.seekTo : start;
                  event.target.seekTo(holdAt, true);
                }
              } catch (_) {}
              cleanYoutubeIframe(mountId);
              if (typeof opts.onReady === "function") {
                try { opts.onReady(event.target); } catch (_) {}
              }
            };
            if (opts.primePlayback && !opts.autoplay) {
              try {
                event.target.mute();
                event.target.playVideo();
              } catch (_) {}
              setTimeout(completeReady, opts.primeMs != null ? opts.primeMs : 450);
            } else {
              completeReady();
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
    const root = playerRoots[mountId];
    return !!(root && players[mountId] && el.contains(root));
  }

  function adoptPreloadedPlayer(mountId, url, height, opts) {
    opts = opts || {};
    const key = preloadKey(url);
    const entry = preloadByUrl[key];
    if (!entry || !players[entry.mountId] || !playerRoots[entry.mountId]) return false;
    const target = document.getElementById(mountId);
    if (!target) return false;

    const sourceMountId = entry.mountId;
    const player = players[sourceMountId];
    const root = playerRoots[sourceMountId];
    const meta = parseYoutubeEmbed(url);
    const fit = normalizeFit(opts);

    try {
      while (target.firstChild) target.removeChild(target.firstChild);
      target.setAttribute("data-yt-url", url);
      target.style.overflow = "hidden";
      target.style.position = target.style.position || "relative";
      target.style.backgroundColor = "#0d0f12";
      installBlurBackdrop(target, meta, fit);
      root.setAttribute("data-yt-url", url);
      root.setAttribute("data-yt-fit", fit);
      stylePlayerRoot(target, root, fit);
      target.appendChild(root);

      players[mountId] = player;
      playerRoots[mountId] = root;
      delete players[sourceMountId];
      delete playerRoots[sourceMountId];
      delete mountInflight[sourceMountId];
      delete preloadByUrl[key];
      if (entry.container && entry.container.parentNode) {
        try { entry.container.parentNode.removeChild(entry.container); } catch (_) {}
      }

      const start = meta ? (meta.start || 0) : 0;
      try {
        if (player && typeof player.mute === "function") player.mute();
        if (player && typeof player.seekTo === "function") player.seekTo(start, true);
        if (player && typeof player.pauseVideo === "function") player.pauseVideo();
      } catch (_) {}
      cleanYoutubeIframe(mountId);
      return true;
    } catch (_) {
      return false;
    }
  }

  function scheduleControlledMountAsync(mountId, url, height, opts) {
    opts = opts || {};
    if (!opts.preloadOnly && adoptPreloadedPlayer(mountId, url, height, opts)) {
      return Promise.resolve();
    }
    reattachPlayer(mountId);
    const el = document.getElementById(mountId);
    const existingUrl = el
      ? (el.getAttribute("data-yt-url") || (playerRoots[mountId] && playerRoots[mountId].getAttribute("data-yt-url")) || "")
      : "";
    const inflight = mountInflight[mountId];
    if (inflight && inflight.url === url) return inflight.promise;
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
      fit: opts.fit,
      seekTo: endTimeForUrl(url),
      timeoutMs: opts.timeoutMs,
    });
  }

  function mountPausedAtStartAsync(mountId, url, height, opts) {
    opts = opts || {};
    const entry = preloadByUrl[preloadKey(url)];
    if (entry && entry.promise && !opts.preloadOnly && !players[entry.mountId]) {
      return entry.promise.then(function () {
        return mountPausedAtStartAsync(mountId, url, height, opts);
      }).catch(function () {
        return scheduleControlledMountAsync(mountId, url, height, {
          autoplay: false,
          loop: false,
          cover: opts.cover !== false,
          fit: opts.fit,
          seekTo: (parseYoutubeEmbed(url) || {}).start || 0,
          primePlayback: opts.primePlayback,
          timeoutMs: opts.timeoutMs,
        });
      });
    }
    reattachPlayer(mountId);
    const el = document.getElementById(mountId);
    const existingUrl = el
      ? (el.getAttribute("data-yt-url") || (playerRoots[mountId] && playerRoots[mountId].getAttribute("data-yt-url")) || "")
      : "";
    const inflight = mountInflight[mountId];
    if (inflight && inflight.url === url) return inflight.promise;
    if (el && players[mountId] && existingUrl === url && isPlayerMounted(mountId)) {
      return Promise.resolve();
    }
    const meta = parseYoutubeEmbed(url);
    const start = meta ? (meta.start || 0) : 0;
    return scheduleControlledMountAsync(mountId, url, height, {
      autoplay: false,
      loop: false,
      cover: opts.cover !== false,
      fit: opts.fit,
      seekTo: start,
      primePlayback: opts.primePlayback,
      timeoutMs: opts.timeoutMs,
    });
  }

  function preloadClip(url, height, opts) {
    opts = opts || {};
    const meta = parseYoutubeEmbed(url);
    if (!meta) return Promise.resolve();
    const key = preloadKey(url);
    const existing = preloadByUrl[key];
    if (existing && existing.promise) return existing.promise;
    const mountId = "hf-yt-preload-" + hashText(key);
    const root = ensurePreloadRoot();
    let holder = document.getElementById(mountId);
    if (!holder) {
      holder = document.createElement("div");
      holder.id = mountId;
      holder.style.position = "relative";
      holder.style.width = "320px";
      holder.style.height = "180px";
      holder.style.overflow = "hidden";
      holder.style.background = "#0d0f12";
      holder.style.pointerEvents = "none";
      root.appendChild(holder);
    }
    const timeoutMs = opts.timeoutMs != null ? opts.timeoutMs : 10000;
    const promise = scheduleControlledMountAsync(mountId, url, height || 180, {
      autoplay: false,
      loop: false,
      cover: false,
      fit: "contain",
      seekTo: meta.start || 0,
      primePlayback: true,
      preloadOnly: true,
      timeoutMs: timeoutMs,
    }).then(function () {
      // Mounting can succeed while YouTube is still showing only its thumbnail.
      // Require real playback once before this clip counts as Player-ready.
      return playPlayerAsync(mountId, url, { timeoutMs: timeoutMs, retryMs: 250 });
    }).then(function (playing) {
      if (!playing) throw new Error("YouTube clip could not be warmed for playback");
      const player = players[mountId];
      try {
        if (player && typeof player.pauseVideo === "function") player.pauseVideo();
        if (player && typeof player.seekTo === "function") player.seekTo(meta.start || 0, true);
      } catch (_) {}
      return true;
    }).catch(function (error) {
      if (preloadByUrl[key] && preloadByUrl[key].mountId === mountId) delete preloadByUrl[key];
      destroyPlayer(mountId);
      if (holder && holder.parentNode) {
        try { holder.parentNode.removeChild(holder); } catch (_) {}
      }
      throw error;
    });
    preloadByUrl[key] = { mountId: mountId, promise: promise, container: holder };
    return promise;
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

  function playPlayerAsync(mountId, url, opts) {
    opts = opts || {};
    const timeoutMs = opts.timeoutMs != null ? opts.timeoutMs : 10000;
    const retryMs = opts.retryMs != null ? opts.retryMs : 250;
    const startedAt = Date.now();
    let requested = false;

    return new Promise(function (resolve) {
      function check() {
        reattachPlayer(mountId);
        const p = players[mountId];
        if (typeof opts.shouldCancel === "function" && opts.shouldCancel()) {
          try { if (p && typeof p.pauseVideo === "function") p.pauseVideo(); } catch (_) {}
          resolve(false);
          return;
        }
        if (p && isPlayerMounted(mountId)) {
          try {
            const playingState = window.YT && window.YT.PlayerState
              ? window.YT.PlayerState.PLAYING
              : 1;
            if (typeof p.getPlayerState === "function" && p.getPlayerState() === playingState) {
              resolve(true);
              return;
            }
          } catch (_) {}

          if (!requested) {
            requested = true;
            playPlayer(mountId, url);
          } else {
            // A muted play request is permitted by browser autoplay policy. Retry
            // without seeking again so an iframe in BUFFERING/CUED can advance.
            try {
              if (typeof p.mute === "function") p.mute();
              if (typeof p.playVideo === "function") p.playVideo();
            } catch (_) {}
          }
        }

        if (Date.now() - startedAt >= timeoutMs) {
          try { if (p && typeof p.pauseVideo === "function") p.pauseVideo(); } catch (_) {}
          resolve(false);
          return;
        }
        setTimeout(check, retryMs);
      }
      check();
    });
  }

  function resumePlayer(mountId) {
    reattachPlayer(mountId);
    const p = players[mountId];
    if (!p || typeof p.playVideo !== "function") return false;
    if (!isPlayerMounted(mountId)) return false;
    try {
      if (typeof p.mute === "function") p.mute();
      p.playVideo();
      return true;
    } catch (_) {
      return false;
    }
  }

  function setPlaybackRate(mountId, rate) {
    reattachPlayer(mountId);
    const p = players[mountId];
    if (!p || typeof p.setPlaybackRate !== "function") return false;
    try {
      p.setPlaybackRate(Number(rate) || 1);
      return true;
    } catch (_) {
      return false;
    }
  }

  function freezeCurrentFrame(mountId) {
    reattachPlayer(mountId);
    const p = players[mountId];
    if (!p) return;
    try {
      if (typeof p.pauseVideo === "function") p.pauseVideo();
    } catch (_) {}
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
    playPlayerAsync,
    pausePlayer,
    resumePlayer,
    setPlaybackRate,
    freezeCurrentFrame,
    preloadClip,
    isReady,
    reattachPlayer,
    reattachAll,
    destroyAll,
    destroyPlayer,
  };
})();
