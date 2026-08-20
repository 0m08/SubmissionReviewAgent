window.HFPlayer = (function () {
  function findStep(slide, visualId) {
    if (!slide || !visualId) return null;
    let found = null;
    (slide.segments || []).forEach(function (seg) {
      (seg.steps || []).forEach(function (st) {
        if (st.visualId === visualId) found = st;
      });
    });
    return found;
  }

  function buildCues(slides) {
    const cues = [];
    (slides || []).forEach(function (slide, slideIdx) {
      const scenes = (slide.scenes || []).filter(function (sc) {
        return (sc.visualIds || []).some(function (vid) { return !!findStep(slide, vid); });
      });
      if (!scenes.length) return;
      scenes.forEach(function (scene, sceneIdx) {
        const visualIds = scene.visualIds || [];
        visualIds.forEach(function (vid, partIdx) {
          const step = findStep(slide, vid);
          if (!step) return;
          const vo = String(step.voiceover || "").trim();
          if (!vo) return;
          cues.push({
            slideIdx: slideIdx,
            sceneIdx: sceneIdx,
            sceneId: scene.id != null ? String(scene.id) : String(sceneIdx + 1),
            partIdx: partIdx,
            partCount: visualIds.length,
            visualId: vid,
            voiceover: vo,
            step: step,
            slideTitle: slide.title || ("Slide " + (slideIdx + 1)),
            rowIndex: step.rowIndex,
            sceneTemplate: scene.template || "",
            animationType: scene.animationType || "none",
            labelText: scene.labelText || "",
            bboxHighlights: scene.bboxHighlights || [],
            iconOverlays: scene.iconOverlays || [],
            slideType: slide.slideType || "",
            slideChunk: slide.slideChunk || "",
            topic: slide.topic || "",
          });
        });
      });
    });
    return cues;
  }

  function isDriveVideoUrl(url) {
    const u = String(url || "").toLowerCase();
    if (u.indexOf("drive.google.com") === -1 && u.indexOf("docs.google.com") === -1) return false;
    // Drive *video* clips carry a trailing "(start=..)" / "(start=..&end=..)" suffix, which is what distinguishes them from Drive *image* URLs.
    return /\(start=\d+(?:&end=\d+)?\)/.test(String(url || ""));
  }

  function afterPaint(fn) {
    if (typeof window.requestAnimationFrame === "function") {
      window.requestAnimationFrame(function () {
        window.requestAnimationFrame(fn);
      });
    } else {
      setTimeout(fn, 32);
    }
  }

  // Two visuals covering parts of the same sentence live in one scene. Handoff
  // between those parts must not insert a pause — it is still one spoken line.
  function sameSceneParts(a, b) {
    return !!(a && b
      && a.slideIdx === b.slideIdx
      && String(a.sceneId) === String(b.sceneId)
      && Math.max(a.partCount || 0, b.partCount || 0) > 1);
  }

  function Controller(options) {
    this.slides = options.slides || [];
    this.resolveImageUrl = options.resolveImageUrl || function () { return ""; };
    this.getYtMountId = options.getYtMountId || function () { return "hf-player-yt-mount"; };
    this.getYtMountHeight = options.getYtMountHeight || function () { return 480; };
    this.getDriveVideoId = options.getDriveVideoId || function () { return "hf-player-drivevid-active"; };
    this.onBeforeCuePlay = options.onBeforeCuePlay || null;
    this.onState = options.onState || function () {};
    this.cues = buildCues(this.slides);
    this.cueIndex = 0;
    this.playing = false;
    this.speed = 1;
    this.volume = 1;
    this.muted = false;
    this.ccOn = true;
    this.pauseForReview = false;
    this.assetsReady = false;
    this.preloadProgress = 0;
    this._preloading = false;
    this._audio = null;
    this._token = 0;
    this._ttsCache = Object.create(null);
    this._ttsInflight = Object.create(null);
    this._ttsPad = Object.create(null);
    this._ttsDur = Object.create(null);
    this._ttsWords = Object.create(null);
    this._pendingSeek = null;
    this._primedAudio = null;
    this._audioCtx = null;
    this._tickTimer = null;
    this._cueElapsed = 0;
    this._cueDuration = 0;
    this._activeCueIndex = -1;
    this.emit();
    this.prefetchAll();
  }

  Controller.prototype.setSlides = function (slides) {
    this.slides = slides || [];
    this.cues = buildCues(this.slides);
    if (this.cueIndex >= this.cues.length) this.cueIndex = Math.max(0, this.cues.length - 1);
    this.assetsReady = false;
    this.prefetchAll();
    this.emit();
  };

  Controller.prototype.emit = function () {
    const cue = this.cues[this.cueIndex] || null;
    let sceneCues = [];
    if (cue) {
      sceneCues = this.cues.filter(function (c) {
        return c.slideIdx === cue.slideIdx && c.sceneId === cue.sceneId;
      });
    }
    const slideTiming = this._slideTiming();
    const pad = this._ttsPad[String((cue && cue.voiceover) || "").trim()] || {};
    this.onState({
      cues: this.cues,
      cueIndex: this.cueIndex,
      cue: cue,
      sceneCues: sceneCues,
      playing: this.playing,
      speed: this.speed,
      volume: this.volume,
      muted: this.muted,
      ccOn: this.ccOn,
      pauseForReview: this.pauseForReview,
      cueElapsed: this._cueElapsed,
      cueDuration: this._cueDuration,
      cueLead: pad.lead || 0,
      cueTrail: pad.trail || 0,
      cueWords: this._ttsWords[String((cue && cue.voiceover) || "").trim()] || null,
      slideElapsed: slideTiming.elapsed,
      slideDuration: slideTiming.duration,
      hasCues: this.cues.length > 0,
      assetsReady: this.assetsReady,
      preloadProgress: this.preloadProgress,
      preloading: this._preloading,
    });
  };

  Controller.prototype._rememberDuration = function (cueOrText, duration) {
    const key = typeof cueOrText === "string"
      ? String(cueOrText || "").trim()
      : String((cueOrText && cueOrText.voiceover) || "").trim();
    if (!key || !(duration > 0) || !isFinite(duration)) return;
    this._ttsDur[key] = duration;
  };

  Controller.prototype._durationForCue = function (cue, index) {
    if (!cue) return 0;
    if (index === this.cueIndex && this._cueDuration > 0) return this._cueDuration;
    const key = String(cue.voiceover || "").trim();
    if (this._ttsDur[key] > 0) return this._ttsDur[key];
    const pad = this._ttsPad[key];
    if (pad && pad.duration > 0) return pad.duration;
    return Math.max(1.2, key.length / 14);
  };

  Controller.prototype._slideTiming = function () {
    const cue = this.cues[this.cueIndex];
    if (!cue) return { elapsed: 0, duration: 0 };
    const slideIdx = cue.slideIdx;
    let duration = 0;
    let elapsed = 0;
    for (let i = 0; i < this.cues.length; i++) {
      const c = this.cues[i];
      if (c.slideIdx !== slideIdx) continue;
      const d = this._durationForCue(c, i);
      if (i < this.cueIndex) elapsed += d;
      duration += d;
    }
    const currentDur = this._durationForCue(cue, this.cueIndex);
    elapsed += Math.max(0, Math.min(currentDur, this._cueElapsed || 0));
    return { elapsed: elapsed, duration: duration };
  };

  Controller.prototype._ytMountIdForCue = function (cue) {
    return this.getYtMountId(cue || this.cues[this.cueIndex]);
  };

  // --- Native Drive-video clip control ---------------------------------------
  // Drive clips render as native <video> elements (not the YouTube iframe API), so we drive them imperatively via the DOM to hard-sync playback with the TTS narration timeline — play on cue start, pause on pause, freeze when the cue ends — mirroring the YouTube path (mounted paused, freezePlayerAtEnd).
  Controller.prototype._isDriveVideoCue = function (cue) {
    if (!cue || !cue.step) return false;
    const rawUrl = cue.step.assetUrl || cue.step.url || "";
    return cue.step.type === "video" && isDriveVideoUrl(rawUrl);
  };

  Controller.prototype._driveVidElForCue = function (cue) {
    if (!this._isDriveVideoCue(cue)) return null;
    try {
      return document.getElementById(this.getDriveVideoId(cue));
    } catch (_) {
      return null;
    }
  };

  Controller.prototype._playDriveVideo = function (cue, fromStart) {
    const el = this._driveVidElForCue(cue);
    if (!el) return false;
    try {
      el.muted = true;
      if (fromStart) {
        try { el.currentTime = 0; } catch (_) {}
      }
      const p = el.play();
      if (p && p.catch) p.catch(function () {});
    } catch (_) {}
    return true;
  };

  Controller.prototype._pauseDriveVideo = function (cue) {
    const el = this._driveVidElForCue(cue);
    if (!el) return false;
    try { el.pause(); } catch (_) {}
    return true;
  };

  Controller.prototype._clearTick = function () {
    if (this._tickTimer) {
      clearInterval(this._tickTimer);
      this._tickTimer = null;
    }
  };

  function withTimeout(promise, ms, fallback) {
    return new Promise(function (resolve) {
      let settled = false;
      const timer = setTimeout(function () {
        if (settled) return;
        settled = true;
        resolve(fallback);
      }, ms);
      Promise.resolve(promise).then(function (val) {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(val);
      }).catch(function () {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(fallback);
      });
    });
  }

  Controller.prototype._stopMedia = function (opts) {
    opts = opts || {};
    this._clearTick();
    this._token += 1;
    if (this._audio) {
      try {
        this._audio.pause();
        this._audio.src = "";
      } catch (_) {}
      this._audio = null;
    }
    if (opts.destroyVideos) {
      if (window.HFYoutube) window.HFYoutube.destroyAll();
      // Native Drive clips aren't destroyable like YouTube iframes; just halt the active one so it doesn't keep playing after a stop/jump.
      this._pauseDriveVideo(this.cues[this.cueIndex]);
      this._activeCueIndex = -1;
    }
    this._cueElapsed = 0;
    if (opts.resetDuration !== false) {
      this._cueDuration = 0;
    }
  };

  Controller.prototype._freezeCueVideo = function (cue) {
    if (!cue || !cue.step) return;
    // Drive clip: freeze by pausing on the current frame (parity with YouTube's freezePlayerAtEnd, which halts the iframe on its final frame).
    if (this._isDriveVideoCue(cue)) {
      this._pauseDriveVideo(cue);
      return;
    }
    if (!window.HFYoutube || !window.HFYoutube.freezePlayerAtEnd) return;
    const step = cue.step;
    const rawUrl = step.assetUrl || step.url || "";
    const isVideo = step.type === "video" && window.HFYoutube.isYoutubeUrl(rawUrl);
    if (!isVideo) return;
    window.HFYoutube.freezePlayerAtEnd(this._ytMountIdForCue(cue), rawUrl);
  };

  Controller.prototype._freezePrevCueVideo = function (prevIdx, newIdx) {
    if (prevIdx < 0 || prevIdx === newIdx) return;
    const prevCue = this.cues[prevIdx];
    const newCue = this.cues[newIdx];
    if (!prevCue || !newCue) return;
    if (prevCue.slideIdx !== newCue.slideIdx || prevCue.sceneId !== newCue.sceneId) return;
    this._freezeCueVideo(prevCue);
  };

  Controller.prototype.stop = function () {
    this.playing = false;
    this._primedAudio = null;
    this._stopMedia({ destroyVideos: true });
    this.emit();
  };

  Controller.prototype._fetchTtsUrl = function (text) {
    const key = String(text || "").trim();
    if (!key) return Promise.reject(new Error("empty voiceover"));
    if (this._ttsCache[key]) return Promise.resolve(this._ttsCache[key]);
    if (this._ttsInflight[key]) return this._ttsInflight[key];
    const audioUrl = "/api/tts?voiceover=" + encodeURIComponent(key);
    const wordsUrl = "/api/tts/words?voiceover=" + encodeURIComponent(key);
    const promise = fetch(audioUrl, { credentials: "same-origin" }).then(function (res) {
      if (!res.ok) throw new Error("TTS request failed");
      let marks = [];
      const encoded = res.headers.get("X-TTS-Words");
      if (encoded) {
        try { marks = JSON.parse(atob(encoded)); } catch (_) { marks = []; }
      }
      return res.blob().then(function (blob) {
        return { blob: blob, marks: marks };
      });
    }).then(function (pair) {
      const blob = pair.blob;
      const marks = pair.marks || [];
      this._ttsWords[key] = marks;
      const objUrl = URL.createObjectURL(blob);
      this._ttsCache[key] = objUrl;
      this._analyzeTtsSilence(key, blob);
      delete this._ttsInflight[key];
      if (!marks.length) {
        fetch(wordsUrl, { credentials: "same-origin" }).then(function (res) {
          if (!res.ok) return { words: [] };
          return res.json();
        }).then(function (body) {
          this._ttsWords[key] = (body && body.words) || [];
          if (this.cues[this.cueIndex] && String(this.cues[this.cueIndex].voiceover || "").trim() === key) {
            this.emit();
          }
        }.bind(this)).catch(function () {});
      }
      if (this.cues[this.cueIndex] && String(this.cues[this.cueIndex].voiceover || "").trim() === key) {
        this.emit();
      }
      return objUrl;
    }.bind(this)).catch(function (err) {
      delete this._ttsInflight[key];
      throw err;
    }.bind(this));
    this._ttsInflight[key] = promise;
    return promise;
  };

  Controller.prototype._analyzeTtsSilence = function (key, blob) {
    const self = this;
    if (!key || self._ttsPad[key] || !blob || !blob.arrayBuffer) return;
    blob.arrayBuffer().then(function (ab) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      if (!self._audioCtx) self._audioCtx = new AC();
      return self._audioCtx.decodeAudioData(ab.slice(0));
    }).then(function (buf) {
      if (!buf) return;
      const data = buf.getChannelData(0);
      const sr = buf.sampleRate || 24000;
      const thresh = 0.02;
      let first = 0;
      let last = data.length - 1;
      for (let i = 0; i < data.length; i++) {
        if (Math.abs(data[i]) > thresh) { first = i; break; }
      }
      for (let i = data.length - 1; i >= 0; i--) {
        if (Math.abs(data[i]) > thresh) { last = i; break; }
      }
      // Keep a tiny pad so the first/last phoneme is not clipped.
      const hold = 0.04;
      self._ttsPad[key] = {
        lead: Math.max(0, first / sr - hold),
        trail: Math.max(0, (data.length - 1 - last) / sr - hold),
        duration: buf.duration || (data.length / sr),
      };
      self._rememberDuration(key, self._ttsPad[key].duration);
    }).catch(function () {});
  };

  Controller.prototype._primeNextAudio = function (fromIndex) {
    const self = this;
    const nextIdx = fromIndex + 1;
    const cue = self.cues[nextIdx];
    if (!cue) {
      self._primedAudio = null;
      return;
    }
    const vo = String(cue.voiceover || "").trim();
    if (!vo) return;
    self._fetchTtsUrl(vo).then(function (url) {
      if (self.cueIndex !== fromIndex) return;
      const audio = new Audio(url);
      audio.preload = "auto";
      self._bindAudio(audio);
      self._primedAudio = { index: nextIdx, audio: audio, url: url, voiceover: vo };
      try { audio.load(); } catch (_) {}
    }).catch(function () {});
  };

  Controller.prototype._uniqueVoiceovers = function (fromIndex, count) {
    const seen = Object.create(null);
    const texts = [];
    const start = Math.max(0, fromIndex || 0);
    const end = count == null ? this.cues.length : Math.min(this.cues.length, start + count);
    for (let i = start; i < end; i++) {
      const t = String(this.cues[i].voiceover || "").trim();
      if (t && !seen[t]) {
        seen[t] = true;
        texts.push(t);
      }
    }
    return texts;
  };

  Controller.prototype.prefetchVoiceovers = function (fromIndex, count) {
    const texts = this._uniqueVoiceovers(fromIndex, count);
    this._prefetchTexts(texts);
  };

  Controller.prototype.prefetchAll = function () {
    this._prefetchTexts(this._uniqueVoiceovers(0, null));
  };

  Controller.prototype._prefetchTexts = function (texts) {
    const self = this;
    const pending = (texts || []).filter(function (t) {
      return t && !self._ttsCache[t] && !self._ttsInflight[t];
    });
    if (!pending.length) return;
    const concurrency = 4;
    let cursor = 0;
    function pump() {
      while (cursor < pending.length) {
        const batch = pending.slice(cursor, cursor + concurrency);
        cursor += concurrency;
        Promise.all(batch.map(function (t) {
          return self._fetchTtsUrl(t).catch(function () {});
        })).then(function () {
          if (cursor < pending.length) pump();
        });
        return;
      }
    }
    pump();
  };

  Controller.prototype.prefetchAllAsync = function (onProgress) {
    const self = this;
    const texts = self._uniqueVoiceovers(0, null);
    if (!texts.length) {
      self.preloadProgress = 100;
      return Promise.resolve();
    }
    let done = 0;
    const total = texts.length;
    function tick() {
      done += 1;
      self.preloadProgress = Math.round((done / total) * 100);
      if (typeof onProgress === "function") onProgress(self.preloadProgress);
      self.emit();
    }
    const concurrency = 4;
    let cursor = 0;
    return new Promise(function (resolve) {
      function pump() {
        if (cursor >= texts.length) {
          resolve();
          return;
        }
        const batch = texts.slice(cursor, cursor + concurrency);
        cursor += concurrency;
        Promise.all(batch.map(function (t) {
          return self._fetchTtsUrl(t).then(tick).catch(tick);
        })).then(pump);
      }
      pump();
    });
  };

  Controller.prototype.isCueAudioReady = function (index) {
    const cue = this.cues[index];
    if (!cue) return true;
    const t = String(cue.voiceover || "").trim();
    return !t || !!this._ttsCache[t];
  };

  Controller.prototype._waitMs = function (ms, token) {
    return new Promise(function (resolve) {
      setTimeout(function () {
        resolve(token === this._token);
      }.bind(this), ms);
    }.bind(this));
  };

  Controller.prototype._waitForVisuals = function (cue, token) {
    const self = this;
    const waitMs = self.onBeforeCuePlay ? 2500 : 64;
    const visualReady = self.onBeforeCuePlay
      ? withTimeout(Promise.resolve().then(function () {
        return afterPaintAsync().then(function () {
          if (token !== self._token) return false;
          return Promise.resolve(self.onBeforeCuePlay(cue)).then(function () {
            return token === self._token;
          });
        });
      }), waitMs, true)
      : afterPaintAsync().then(function () { return token === self._token; });
    return visualReady.then(function (ok) {
      if (!ok) return false;
      return afterPaintAsync().then(function () { return token === self._token; });
    });
  };

  Controller.prototype._mountVideoAsync = function (cue, token) {
    const self = this;
    const step = cue.step;
    const rawUrl = step.assetUrl || step.url || "";
    const isVideo = step.type === "video" && window.HFYoutube && window.HFYoutube.isYoutubeUrl(rawUrl);
    if (!isVideo || !window.HFYoutube) {
      return Promise.resolve(token === self._token);
    }
    const mountId = self._ytMountIdForCue(cue);
    if (window.HFYoutube.reattachPlayer) window.HFYoutube.reattachPlayer(mountId);
    if (window.HFYoutube.isReady && window.HFYoutube.isReady(mountId, rawUrl)) {
      return Promise.resolve(token === self._token);
    }
    const height = self.getYtMountHeight(cue) || 480;
    const mountFn = window.HFYoutube.mountPausedAtStartAsync
      || window.HFYoutube.scheduleControlledMountAsync;
    if (!mountFn) return Promise.resolve(token === self._token);
    return mountFn.call(window.HFYoutube, mountId, rawUrl, height, {
      autoplay: false,
      loop: false,
      cover: true,
      timeoutMs: 4500,
    }).then(function () {
      return token === self._token;
    }).catch(function () {
      return token === self._token;
    });
  };

  Controller.prototype._prefetchUpcomingVideos = function (fromIndex, count) {
    const self = this;
    if (!window.HFYoutube || !window.HFYoutube.mountPausedAtStartAsync) return;
    const start = Math.max(0, fromIndex || 0);
    const end = Math.min(self.cues.length, start + (count || 2));
    for (let i = start; i < end; i++) {
      const cue = self.cues[i];
      if (!cue || !cue.step || cue.step.type !== "video") continue;
      const rawUrl = cue.step.assetUrl || cue.step.url || "";
      if (!window.HFYoutube.isYoutubeUrl(rawUrl)) continue;
      const mountId = self._ytMountIdForCue(cue);
      if (!document.getElementById(mountId)) continue;
      const height = self.getYtMountHeight(cue) || 480;
      window.HFYoutube.mountPausedAtStartAsync(mountId, rawUrl, height, {
        cover: true,
        timeoutMs: 8000,
      }).catch(function () {});
    }
  };

  function afterPaintAsync() {
    return new Promise(function (resolve) {
      afterPaint(resolve);
    });
  }

  Controller.prototype._playCue = function (index) {
    const self = this;
    if (!self.cues.length) return Promise.resolve();
    const idx = Math.max(0, Math.min(index, self.cues.length - 1));
    const prevCue = self.cues[self._activeCueIndex];
    self._freezePrevCueVideo(self._activeCueIndex, idx);
    self.cueIndex = idx;
    self._activeCueIndex = idx;
    const cue = self.cues[idx];
    const nextCue = self.cues[idx + 1];
    const tightHandoff = sameSceneParts(prevCue, cue);
    const tightToNext = sameSceneParts(cue, nextCue);
    self.prefetchVoiceovers(idx + 1, 8);
    self._stopMedia({ destroyVideos: false, resetDuration: true });
    const token = self._token;
    self.emit();

    const step = cue.step;
    const rawUrl = step.assetUrl || step.url || "";
    const isVideo = step.type === "video" && window.HFYoutube && window.HFYoutube.isYoutubeUrl(rawUrl);
    const isDriveVideo = self._isDriveVideoCue(cue);
    const mountId = self._ytMountIdForCue(cue);
    const voKey = String(cue.voiceover || "").trim();

    const primed = (self._primedAudio && self._primedAudio.index === idx) ? self._primedAudio : null;
    self._primedAudio = null;

    // Same-scene image parts are already on screen — don't wait for preload/rAF.
    const visualsPromise = (tightHandoff && !isVideo)
      ? Promise.resolve(true)
      : self._waitForVisuals(cue, token);
    const mountPromise = isVideo
      ? visualsPromise.then(function (alive) {
        if (!alive) return false;
        return self._mountVideoAsync(cue, token);
      })
      : visualsPromise;

    const ttsPromise = primed && primed.url
      ? Promise.resolve(primed.url)
      : visualsPromise.then(function (alive) {
        if (!alive) return null;
        return self._fetchTtsUrl(cue.voiceover);
      });

    return withTimeout(Promise.all([
      ttsPromise,
      mountPromise,
    ]).then(function (results) {
      return results[0];
    }), tightHandoff ? 2500 : 8000, primed && primed.url ? primed.url : null).then(function (audioUrl) {
      if (token !== self._token || audioUrl == null) return;
      return new Promise(function (resolve) {
        const audio = (primed && primed.audio && primed.url === audioUrl)
          ? primed.audio
          : new Audio(audioUrl);
        audio.preload = "auto";
        self._bindAudio(audio);
        self._audio = audio;

        let started = false;
        let finished = false;

        function onDone() {
          if (token !== self._token || finished) return;
          finished = true;
          self._clearTick();
          if (isVideo && window.HFYoutube) {
            window.HFYoutube.freezePlayerAtEnd(mountId, rawUrl);
          }
          if (isDriveVideo) self._pauseDriveVideo(cue);
          self._cueElapsed = self._cueDuration;
          self.emit();
          resolve();
        }

        function startPlayback() {
          if (started || token !== self._token) return;
          started = true;
          if (isVideo && window.HFYoutube) {
            if (!window.HFYoutube.playPlayer(mountId, rawUrl)) {
              self._mountVideoAsync(cue, token).then(function () {
                if (token !== self._token) return;
                window.HFYoutube.playPlayer(mountId, rawUrl);
              });
            }
          }
          if (isDriveVideo && !self._playDriveVideo(cue, true)) {
            afterPaint(function () {
              if (token !== self._token) return;
              self._playDriveVideo(cue, true);
            });
          }
          const playPromise = audio.play();
          if (playPromise && playPromise.catch) playPromise.catch(onDone);
          self._tickTimer = setInterval(function () {
            if (token !== self._token || !self._audio) return;
            self._cueElapsed = self._audio.currentTime || 0;
            if (tightToNext) {
              const pad = self._ttsPad[voKey] || {};
              const trail = pad.trail || 0;
              if (trail > 0.08 && self._cueDuration > 0 && self._cueElapsed >= self._cueDuration - trail) {
                onDone();
                return;
              }
            }
            self.emit();
          }, 50);
          self._prefetchUpcomingVideos(idx + 1, 2);
          self._primeNextAudio(idx);
        }

        function onMetaReady() {
          if (token !== self._token || started) return;
          self._cueDuration = audio.duration || 0;
          self._rememberDuration(cue, self._cueDuration);
          const pad = self._ttsPad[voKey] || {};
          const pending = self._pendingSeek;
          self._pendingSeek = null;
          if (pending != null && pending >= 0) {
            const t = Math.max(0, Math.min(self._cueDuration || pending, pending));
            try { audio.currentTime = t; } catch (_) {}
            self._cueElapsed = t;
          } else {
            const lead = tightHandoff ? (pad.lead || 0) : 0;
            if (lead > 0.05) {
              try { audio.currentTime = lead; } catch (_) {}
              self._cueElapsed = lead;
            } else {
              self._cueElapsed = 0;
            }
          }
          self.emit();
          if (isVideo) {
            afterPaint(function () {
              if (token !== self._token || started) return;
              self._mountVideoAsync(cue, token).then(function () {
                afterPaint(startPlayback);
              });
            });
          } else if (tightHandoff) {
            startPlayback();
          } else {
            afterPaint(startPlayback);
          }
        }

        audio.addEventListener("ended", onDone);
        audio.addEventListener("error", onDone);
        if (audio.readyState >= 1) {
          onMetaReady();
        } else {
          audio.addEventListener("loadedmetadata", onMetaReady);
          audio.addEventListener("canplay", onMetaReady);
          setTimeout(function () {
            if (!started && token === self._token) {
              if (!self._cueDuration) self._cueDuration = audio.duration || 0;
              onMetaReady();
            }
          }, tightHandoff ? 800 : 4000);
          try { audio.load(); } catch (_) {}
        }
      });
    }).catch(function () {
      return self._waitMs(800, token);
    });
  };

  Controller.prototype.prepareAssets = function (preloadImagesFn, onProgress) {
    const self = this;
    if (self._preloading) return self._preloadPromise || Promise.resolve();
    self._preloading = true;
    self.assetsReady = false;
    self.preloadProgress = 0;
    self.emit();

    const imageProgress = function (pct) {
      self.preloadProgress = Math.round(pct);
      if (typeof onProgress === "function") onProgress(self.preloadProgress);
      self.emit();
    };

    self._preloadPromise = self.prefetchAllAsync(function (ttsPct) {
      const blended = Math.round(ttsPct * 0.55);
      self.preloadProgress = blended;
      if (typeof onProgress === "function") onProgress(blended);
    }).then(function () {
      self.assetsReady = true;
      self.preloadProgress = Math.max(self.preloadProgress, 55);
      self._preloading = false;
      self.emit();
      if (typeof preloadImagesFn !== "function") return;
      return preloadImagesFn(function (imgPct) {
        const blended = Math.round(55 + imgPct * 0.45);
        imageProgress(blended);
      });
    }).then(function () {
      self.preloadProgress = 100;
      self.emit();
    }).catch(function () {
      self.assetsReady = true;
      self.preloadProgress = 100;
      self._preloading = false;
      self.emit();
    });

    return self._preloadPromise;
  };

  Controller.prototype.play = function () {
    if (!this.cues.length) return;
    const self = this;
    if (!self.assetsReady) {
      self.playing = true;
      self.emit();
      self.prepareAssets().then(function () {
        if (!self.playing) return;
        self._runLoop();
      });
      return;
    }
    self.playing = true;
    self.emit();
    self._runLoop();
  };

  Controller.prototype._runLoop = function () {
    const self = this;
    if (!self.playing) return;
    self._playCue(self.cueIndex).then(function () {
      if (!self.playing) return;
      if (self.cueIndex < self.cues.length - 1) {
        const cur = self.cues[self.cueIndex];
        self.cueIndex += 1;
        self._cueElapsed = 0;
        self._cueDuration = 0;
        self.emit();
        const nxt = self.cues[self.cueIndex];
        if (sameSceneParts(cur, nxt)) {
          if (!self.playing) return;
          self._runLoop();
          return;
        }
        afterPaint(function () {
          if (!self.playing) return;
          self._runLoop();
        });
      } else {
        self.playing = false;
        self.emit();
      }
    });
  };

  Controller.prototype.togglePlay = function () {
    if (!this.cues.length) return;
    if (this.playing) {
      this.playing = false;
      if (this._audio) {
        try { this._audio.pause(); } catch (_) {}
      }
      const cue = this.cues[this.cueIndex];
      if (window.HFYoutube && cue) window.HFYoutube.pausePlayer(this._ytMountIdForCue(cue));
      if (cue) this._pauseDriveVideo(cue);
      this._clearTick();
      this.emit();
      return;
    }
    if (this._audio && this._cueDuration > 0 && this._cueElapsed > 0 && this._cueElapsed < this._cueDuration - 0.05) {
      this.playing = true;
      this._bindAudio(this._audio);
      const playPromise = this._audio.play();
      if (playPromise && playPromise.catch) playPromise.catch(function () {});
      const cue = this.cues[this.cueIndex];
      if (window.HFYoutube && cue) {
        const step = cue.step;
        const rawUrl = step.assetUrl || step.url || "";
        window.HFYoutube.playPlayer(this._ytMountIdForCue(cue), rawUrl);
      }
      // Resume the Drive clip from where it froze (do not restart from 0).
      if (cue) this._playDriveVideo(cue, false);
      const token = this._token;
      this._tickTimer = setInterval(function () {
        if (token !== this._token || !this._audio) return;
        this._cueElapsed = this._audio.currentTime || 0;
        this.emit();
      }.bind(this), 120);
      this.emit();
      return;
    }
    this.play();
  };

  Controller.prototype.prev = function () {
    if (!this.cues.length) return;
    const wasPlaying = this.playing;
    this.playing = false;
    // Halt the current native Drive clip before stepping back (YouTube is handled by _stopMedia + reattach; native <video> persists per slot, so pause it).
    this._pauseDriveVideo(this.cues[this.cueIndex]);
    this._primedAudio = null;
    this._stopMedia({ destroyVideos: false });
    this.cueIndex = Math.max(0, this.cueIndex - 1);
    this._activeCueIndex = this.cueIndex;
    this.emit();
    if (wasPlaying) {
      this.playing = true;
      this._runLoop();
    }
  };

  Controller.prototype.next = function () {
    if (!this.cues.length) return;
    const wasPlaying = this.playing;
    this.playing = false;
    this._freezeCueVideo(this.cues[this.cueIndex]);
    this._primedAudio = null;
    this._stopMedia({ destroyVideos: false });
    this.cueIndex = Math.min(this.cues.length - 1, this.cueIndex + 1);
    this._activeCueIndex = this.cueIndex;
    this.emit();
    if (wasPlaying) {
      this.playing = true;
      this._runLoop();
    }
  };

  Controller.prototype.setSpeed = function (speed) {
    this.speed = speed;
    if (this._audio) this._audio.playbackRate = speed;
    this.emit();
  };

  Controller.prototype._audioVolume = function () {
    if (this.muted) return 0;
    return Math.max(0, Math.min(1, this.volume));
  };

  Controller.prototype._bindAudio = function (audio) {
    if (!audio) return;
    audio.playbackRate = this.speed;
    audio.volume = this._audioVolume();
  };

  Controller.prototype.setVolume = function (v) {
    this.volume = Math.max(0, Math.min(1, Number(v)));
    if (this.volume > 0) this.muted = false;
    if (this._audio) this._audio.volume = this._audioVolume();
    this.emit();
  };

  Controller.prototype.toggleMute = function () {
    this.muted = !this.muted;
    if (this._audio) this._audio.volume = this._audioVolume();
    this.emit();
  };

  Controller.prototype.seek = function (ratio) {
    const cue = this.cues[this.cueIndex];
    if (!cue) return;
    const slideIdx = cue.slideIdx;
    const indices = [];
    const durs = [];
    let total = 0;
    for (let i = 0; i < this.cues.length; i++) {
      if (this.cues[i].slideIdx !== slideIdx) continue;
      const d = this._durationForCue(this.cues[i], i);
      indices.push(i);
      durs.push(d);
      total += d;
    }
    if (!(total > 0)) return;
    let target = Math.max(0, Math.min(1, Number(ratio) || 0)) * total;
    for (let k = 0; k < indices.length; k++) {
      const d = Math.max(0.01, durs[k]);
      const isLast = k === indices.length - 1;
      if (!isLast && target > d) {
        target -= d;
        continue;
      }
      const idx = indices[k];
      const offset = Math.max(0, Math.min(d, target));
      if (idx === this.cueIndex && this._audio && this._cueDuration > 0) {
        try { this._audio.currentTime = offset; } catch (_) {}
        this._cueElapsed = offset;
        this._pendingSeek = null;
        this.emit();
        return;
      }
      this._seekToSlideCue(idx, offset);
      return;
    }
  };

  Controller.prototype._seekToSlideCue = function (idx, offset) {
    if (idx < 0 || idx >= this.cues.length) return;
    const wasPlaying = this.playing;
    this.playing = false;
    this._primedAudio = null;
    this._pauseDriveVideo(this.cues[this.cueIndex]);
    this._stopMedia({ destroyVideos: false });
    this.cueIndex = idx;
    this._activeCueIndex = -1;
    this._pendingSeek = offset;
    this._cueElapsed = offset;
    this.emit();
    if (wasPlaying) {
      this.playing = true;
      this._runLoop();
    }
  };

  Controller.prototype.replay = function () {
    if (!this.cues.length) return;
    this.playing = false;
    this._primedAudio = null;
    this._pauseDriveVideo(this.cues[this.cueIndex]);
    this._stopMedia({ destroyVideos: false });
    this._activeCueIndex = -1;
    this.play();
  };

  Controller.prototype.toggleCc = function () {
    this.ccOn = !this.ccOn;
    this.emit();
  };

  Controller.prototype.togglePauseForReview = function () {
    this.pauseForReview = !this.pauseForReview;
    this.emit();
  };

  Controller.prototype.jumpToSlide = function (slideIdx) {
    const idx = this.cues.findIndex(function (c) { return c.slideIdx === slideIdx; });
    if (idx < 0) return;
    this.playing = false;
    this._primedAudio = null;
    this._pendingSeek = null;
    this._stopMedia({ destroyVideos: true });
    this.cueIndex = idx;
    this._activeCueIndex = idx;
    this.emit();
  };

  Controller.prototype.getVisualPresentation = function (resolveImageUrl) {
    const cue = this.cues[this.cueIndex];
    if (!cue) return { type: "empty" };
    const step = cue.step;
    const rawUrl = step.assetUrl || step.url || "";
    if (step.type === "video" && window.HFYoutube && window.HFYoutube.isYoutubeUrl(rawUrl)) {
      return { type: "video", mountId: this._ytMountIdForCue(cue), rawUrl: rawUrl, label: step.label || "" };
    }

    const isDriveVideo = rawUrl
      && rawUrl.toLowerCase().indexOf("drive.google.com") !== -1
      && /\(start=\d+(?:&end=\d+)?\)/.test(rawUrl);
    if (step.type === "video" && isDriveVideo) {
      return { type: "image", url: "/api/assets/image?thumb=1&url=" + encodeURIComponent(rawUrl), label: step.label || "" };
    }
    const img = resolveImageUrl(step);
    return { type: "image", url: img, label: step.label || "" };
  };

  return {
    buildCues: buildCues,
    Controller: Controller,
  };
})();
