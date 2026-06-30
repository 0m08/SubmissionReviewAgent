window.HFYoutube = (function () {
  const players = {};
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
    const player = players[mountId];
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

  return {
    parseYoutubeEmbed,
    isYoutubeUrl,
    mount,
    scheduleMount,
    destroyAll,
    destroyPlayer,
  };
})();