// hero-stills.js
// Single-hero still treatments: zoom in, still (no motion), center-split reveal.
// center_split: two half-width shutters cover the still on first paint, then
// slide outward from the midline. Do not clip/tween the image itself.
window.HFHeroStills = (function() {
  const REVEAL_DUR = 0.8;

  let activeTween = null;
  let activeTweenDuration = 0;
  let currentTarget = null;
  let currentWrap = null;
  let currentCueKey = '';
  let splitDone = false;

  function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function isStill(treatment) {
    return treatment === 'still' || treatment === 'off' || treatment === 'none';
  }

  function overlayTimingOpts(opts, cue, duration) {
    return {
      cueWords: opts && opts.cueWords,
      voiceover: cue && cue.voiceover,
      duration: duration,
      cueLead: (opts && opts.cueLead) || 0,
    };
  }

  function resolveOverlayTime(phrase, timingOpts, fallbackSeconds) {
    if (window.HFOverlayTiming && window.HFOverlayTiming.resolveSeconds) {
      return window.HFOverlayTiming.resolveSeconds(phrase, timingOpts, fallbackSeconds);
    }
    return Math.max(0, Number(fallbackSeconds) || 0);
  }

  // Must stay in sync with agent _BBOX_MIN_GAP / size floors (normalized 0–1000 coords).
  const BBOX_MIN_GAP = 8;
  const BBOX_MIN_WIDTH = 55;
  const BBOX_MIN_HEIGHT = 55;
  const BBOX_MIN_AREA = 6000;

  function bboxPairInterferes(a, b, minGap) {
    const gap = minGap == null ? BBOX_MIN_GAP : minGap;
    const aY0 = a[0], aX0 = a[1], aY1 = a[2], aX1 = a[3];
    const bY0 = b[0], bX0 = b[1], bY1 = b[2], bX1 = b[3];
    const hGap = Math.max(aX0, bX0) - Math.min(aX1, bX1);
    const vGap = Math.max(aY0, bY0) - Math.min(aY1, bY1);
    if (hGap < 0 && vGap < 0) return true; // area overlap
    if (hGap < 0) return vGap < gap; // vertical neighbors
    if (vGap < 0) return hGap < gap; // horizontal neighbors
    return false; // diagonal separation
  }

  function isBboxTooSmall(box) {
    if (!box || box.length !== 4) return true;
    const width = box[3] - box[1];
    const height = box[2] - box[0];
    if (!(width > 0) || !(height > 0)) return true;
    return width < BBOX_MIN_WIDTH || height < BBOX_MIN_HEIGHT || (width * height) < BBOX_MIN_AREA;
  }

  function filterValidBboxHighlights(cue) {
    const validHighlights = [];
    (cue.bboxHighlights || []).forEach(function (highlight) {
      const box = highlight.box_2d;
      if (!box || box.length !== 4) return;
      if (isBboxTooSmall(box)) return;
      let interferes = false;
      for (let i = 0; i < validHighlights.length; i++) {
        if (bboxPairInterferes(box, validHighlights[i].box_2d, BBOX_MIN_GAP)) {
          interferes = true;
          break;
        }
      }
      if (!interferes) validHighlights.push(highlight);
    });
    return validHighlights;
  }

  function bboxBoxMetrics(rect, box) {
    const ymin = box[0], xmin = box[1], ymax = box[2], xmax = box[3];
    return {
      left: rect.x + (xmin / 1000) * rect.w,
      top: rect.y + (ymin / 1000) * rect.h,
      width: ((xmax - xmin) / 1000) * rect.w,
      height: ((ymax - ymin) / 1000) * rect.h,
      relLeft: (xmin / 1000) * rect.w,
      relTop: (ymin / 1000) * rect.h,
    };
  }

  function appendSplitBboxPaths(svgEl, highlight, width, height, strokeColor, strokeWidth) {
    let d1 = '';
    let d2 = '';
    if (highlight.shape === 'circle') {
      const rx = width / 2;
      const ry = height / 2;
      d1 = 'M ' + rx + ' 0 A ' + rx + ' ' + ry + ' 0 0 0 ' + rx + ' ' + height;
      d2 = 'M ' + rx + ' 0 A ' + rx + ' ' + ry + ' 0 0 1 ' + rx + ' ' + height;
    } else {
      const halfW = width / 2;
      d1 = 'M ' + halfW + ' 0 L 0 0 L 0 ' + height + ' L ' + halfW + ' ' + height;
      d2 = 'M ' + halfW + ' 0 L ' + width + ' 0 L ' + width + ' ' + height + ' L ' + halfW + ' ' + height;
    }
    const path1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path1.setAttribute('d', d1);
    path1.setAttribute('fill', 'none');
    path1.setAttribute('stroke', strokeColor);
    path1.setAttribute('stroke-width', String(strokeWidth));
    path1.setAttribute('stroke-linecap', 'round');
    path1.setAttribute('stroke-linejoin', 'round');

    const path2 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path2.setAttribute('d', d2);
    path2.setAttribute('fill', 'none');
    path2.setAttribute('stroke', strokeColor);
    path2.setAttribute('stroke-width', String(strokeWidth));
    path2.setAttribute('stroke-linecap', 'round');
    path2.setAttribute('stroke-linejoin', 'round');

    svgEl.appendChild(path1);
    svgEl.appendChild(path2);
    return { path1: path1, path2: path2 };
  }

  function animateSplitBboxDraw(tl, svgEl, paths, width, height, startTime, drawDuration) {
    const len1 = (typeof paths.path1.getTotalLength === 'function' && paths.path1.getTotalLength() > 0)
      ? paths.path1.getTotalLength()
      : (width + height);
    const len2 = (typeof paths.path2.getTotalLength === 'function' && paths.path2.getTotalLength() > 0)
      ? paths.path2.getTotalLength()
      : (width + height);
    paths.path1.style.strokeDasharray = len1;
    paths.path1.style.strokeDashoffset = len1;
    paths.path2.style.strokeDasharray = len2;
    paths.path2.style.strokeDashoffset = len2;
    tl.fromTo(svgEl, { opacity: 0 }, { opacity: 1, duration: 0.15, ease: 'power1.out' }, startTime);
    tl.fromTo(paths.path1, { strokeDashoffset: len1 }, { strokeDashoffset: 0, duration: drawDuration, ease: 'power1.out' }, startTime);
    tl.fromTo(paths.path2, { strokeDashoffset: len2 }, { strokeDashoffset: 0, duration: drawDuration, ease: 'power1.out' }, startTime);
  }

  function parseDashPeriod(dashPattern) {
    const parts = String(dashPattern || '18 10').trim().split(/[\s,]+/).map(parseFloat).filter(function (n) {
      return Number.isFinite(n) && n > 0;
    });
    if (!parts.length) return 28;
    if (parts.length === 1) return parts[0] * 2;
    return parts[0] + parts[1];
  }

  function animateSpotlightBorderMarch(tl, borderShape, startTime, sceneDuration, dashPattern, marchCycleSec) {
    const period = parseDashPeriod(dashPattern);
    borderShape.style.strokeDasharray = dashPattern;
    borderShape.style.strokeDashoffset = '0';
    const remaining = Math.max(0.1, sceneDuration - startTime);
    const cycle = Math.max(0.35, marchCycleSec || 1.1);
    const repeats = Math.max(0, Math.ceil(remaining / cycle) - 1);
    tl.fromTo(
      borderShape,
      { strokeDashoffset: 0 },
      { strokeDashoffset: -period, duration: cycle, ease: 'none', repeat: repeats },
      startTime
    );
  }

  function themeGetters() {
    const theme = window.HFPlayerTheme;
    return {
      tGet: function (key, fallback) {
        return theme && theme.get ? (theme.get(key) || fallback) : fallback;
      },
      tSize: function (key, fallback) {
        return theme && theme.fontSizePx ? theme.fontSizePx(key) : fallback;
      },
    };
  }

  function isTopicTransition(cue) {
    if (!cue) return false;
    if ((cue.slideType || '').trim().toLowerCase() !== 'transition') return false;
    const t = String(cue.topic || '').trim().toLowerCase().replace(/[^a-z0-9]/g, '');
    const c = String(cue.slideChunk || '').trim().toLowerCase().replace(/[^a-z0-9]/g, '');
    const st = String(cue.slideTitle || '').trim().toLowerCase().replace(/[^a-z0-9]/g, '');
    return t && (c === t || st === t);
  }

  function queryShutters(wrap) {
    if (!wrap) return null;
    const left = wrap.querySelector('.hf-hero-shutter.is-left');
    const right = wrap.querySelector('.hf-hero-shutter.is-right');
    if (!left || !right) return null;
    return { left: left, right: right };
  }

  function setShuttersClosed(s) {
    gsap.set(s.left, { xPercent: 0 });
    gsap.set(s.right, { xPercent: 0 });
  }

  function setShuttersOpen(s) {
    gsap.set(s.left, { xPercent: -100 });
    gsap.set(s.right, { xPercent: 100 });
  }

  function killTween() {
    if (activeTween) {
      activeTween.kill();
      activeTween = null;
    }
  }

  function kill() {
    killTween();
    activeTweenDuration = 0;
    if (currentTarget) {
      gsap.killTweensOf(currentTarget);
      gsap.set(currentTarget, { clearProps: 'transform,clipPath,filter,opacity', scale: 1, x: 0, y: 0, xPercent: 0, yPercent: 0 });
      currentTarget.classList.remove('is-active-motion');
      currentTarget = null;
    }
    if (currentWrap) {
      const s = queryShutters(currentWrap);
      if (s) {
        gsap.killTweensOf([s.left, s.right]);
        setShuttersClosed(s);
      }
      const overlay = currentWrap.querySelector('.hf-hero-overlay-container');
      if (overlay) {
        overlay.remove();
      }
    }
    currentWrap = null;
    currentCueKey = '';
    splitDone = false;
  }

  function restartSplit() {
    killTween();
    splitDone = false;
    const s = queryShutters(currentWrap);
    if (s) {
      gsap.killTweensOf([s.left, s.right]);
      setShuttersClosed(s);
    }
  }

  function syncCenterSplit(wrap, playing, forceRestart) {
    const s = queryShutters(wrap);
    if (!s) return;
    wrap.style.overflow = 'hidden';

    if (forceRestart) restartSplit();

    if (splitDone) {
      setShuttersOpen(s);
      return;
    }

    if (!playing) {
      if (activeTween) {
        activeTween.pause();
        return;
      }
      setShuttersClosed(s);
      return;
    }

    if (activeTween) {
      activeTween.play();
      return;
    }

    setShuttersClosed(s);
    const tl = gsap.timeline({
      defaults: { ease: 'power2.inOut', overwrite: 'auto' },
      onComplete: function () {
        splitDone = true;
      },
    });
    tl.fromTo(s.left, { xPercent: 0 }, { xPercent: -100, duration: REVEAL_DUR }, 0);
    tl.fromTo(s.right, { xPercent: 0 }, { xPercent: 100, duration: REVEAL_DUR }, 0);
    activeTween = tl;
  }

  function getVisibleImageRect(container, img) {
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (!img || !img.naturalWidth || !img.naturalHeight) {
      return { x: 0, y: 0, w: cw, h: ch };
    }
    const iw = img.naturalWidth;
    const ih = img.naturalHeight;
    const containerRatio = cw / ch;
    const imageRatio = iw / ih;

    let w, h, x, y;
    if (imageRatio > containerRatio) {
      w = cw;
      h = cw / imageRatio;
      x = 0;
      y = (ch - h) / 2;
    } else {
      h = ch;
      w = ch * imageRatio;
      x = (cw - w) / 2;
      y = 0;
    }
    return { x: x, y: y, w: w, h: h };
  }

  // Resolve inset like "6%" or "24" against a reference length (usually image width).
  function resolveInsetPx(insetRaw, refLength) {
    const raw = String(insetRaw || '6%').trim();
    if (/%$/.test(raw)) {
      const pct = parseFloat(raw);
      return (isFinite(pct) ? pct : 6) / 100 * refLength;
    }
    const n = parseFloat(raw);
    return isFinite(n) ? n : 0.06 * refLength;
  }

  function parseCssLengthPx(raw, fallback) {
    const n = parseFloat(String(raw || '').trim());
    return isFinite(n) && n > 0 ? n : fallback;
  }

  // Keep overlays inside the sharp object-fit:contain image, not the blurred letterbox.
  // Do NOT use the `inset` shorthand here — it resets left/top/right/bottom.
  function placeInVisibleImage(el, container, img) {
    const rect = getVisibleImageRect(container, img);
    el.style.position = 'absolute';
    el.style.left = rect.x + 'px';
    el.style.top = rect.y + 'px';
    el.style.width = rect.w + 'px';
    el.style.height = rect.h + 'px';
    el.style.right = 'auto';
    el.style.bottom = 'auto';
    return rect;
  }

  // Full motion wrapper / frame — includes blurred letterbox pillars/bars.
  // Use for callouts so narrow portrait heroes still get readable card width.
  function placeInFullFrame(el, container) {
    const w = container.clientWidth;
    const h = container.clientHeight;
    el.style.position = 'absolute';
    el.style.left = '0px';
    el.style.top = '0px';
    el.style.width = w + 'px';
    el.style.height = h + 'px';
    el.style.right = 'auto';
    el.style.bottom = 'auto';
    return { x: 0, y: 0, w: w, h: h };
  }

  // Callout/overlay icons are generated as an orange glyph on a solid white square.
  // The design guide wants the glyph transparent, so knock out near-white pixels.
  // Icons are same-origin via /api/assets/image, so the canvas is not tainted.
  const calloutIconSrcCache = Object.create(null);
  const calloutIconPending = Object.create(null);

  function knockOutWhiteFromSource(source) {
    if (!source) return '';
    const w = source.naturalWidth;
    const h = source.naturalHeight;
    if (!w || !h) return '';
    try {
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(source, 0, 0);
      const imageData = ctx.getImageData(0, 0, w, h);
      const d = imageData.data;
      const HARD = 236;
      const SOFT = 200;
      for (let i = 0; i < d.length; i += 4) {
        const m = Math.min(d[i], d[i + 1], d[i + 2]);
        if (m >= HARD) {
          d[i + 3] = 0;
        } else if (m >= SOFT) {
          d[i + 3] = Math.round(d[i + 3] * (HARD - m) / (HARD - SOFT));
        }
      }
      ctx.putImageData(imageData, 0, 0);
      return canvas.toDataURL('image/png');
    } catch (e) {
      return '';
    }
  }

  function knockOutWhiteToTransparent(img) {
    if (!img || img.dataset.hfKnocked === '1') return;
    const dataUrl = knockOutWhiteFromSource(img);
    img.dataset.hfKnocked = '1';
    if (dataUrl) img.src = dataUrl;
  }

  function ensureCalloutIconReady(url, knockoutWhite) {
    if (!url) return Promise.resolve('');
    if (!knockoutWhite) {
      calloutIconSrcCache[url] = url;
      return Promise.resolve(url);
    }
    if (calloutIconSrcCache[url]) return Promise.resolve(calloutIconSrcCache[url]);
    if (calloutIconPending[url]) return calloutIconPending[url];

    calloutIconPending[url] = new Promise(function (resolve) {
      const loader = new Image();
      loader.crossOrigin = 'anonymous';
      loader.onload = function () {
        const knocked = knockOutWhiteFromSource(loader);
        const finalSrc = knocked || url;
        calloutIconSrcCache[url] = finalSrc;
        delete calloutIconPending[url];
        resolve(finalSrc);
      };
      loader.onerror = function () {
        calloutIconSrcCache[url] = url;
        delete calloutIconPending[url];
        resolve(url);
      };
      loader.src = url;
    });
    return calloutIconPending[url];
  }

  function warmCalloutIcons(urls, knockoutWhite) {
    const knockout = knockoutWhite !== false;
    const jobs = [];
    (urls || []).forEach(function (raw) {
      const url = getProxyUrl(raw);
      if (!url) return;
      jobs.push(ensureCalloutIconReady(url, knockout));
    });
    return jobs.length ? Promise.all(jobs) : Promise.resolve();
  }

  function getProxyUrl(url) {
    if (!url) return '';
    if (url.indexOf('drive.google.com') !== -1 || url.indexOf('docs.google.com') !== -1 || url.indexOf('googleusercontent.com') !== -1) {
      return '/api/assets/image?url=' + encodeURIComponent(url);
    }
    return url;
  }

  function normalizeCalloutPosition(position) {
    if (window.HFTreatmentRotator && window.HFTreatmentRotator.normalizeCalloutPosition) {
      return window.HFTreatmentRotator.normalizeCalloutPosition(position);
    }
    const pos = String(position || 'center_left').toLowerCase().replace(/[\s-]+/g, '_');
    const allowed = [
      'top_left', 'top_center', 'top_right',
      'center_left', 'center', 'center_right',
      'bottom_left', 'bottom_center', 'bottom_right',
    ];
    return allowed.indexOf(pos) >= 0 ? pos : 'center_left';
  }

  var CALLOUT_CORNER_POSITIONS = ['top_left', 'top_right', 'bottom_left', 'bottom_right'];

  function resolveCalloutPosition(position, cardIdx, cardCount) {
    if (cardCount <= 1) {
      return normalizeCalloutPosition(position);
    }
    return CALLOUT_CORNER_POSITIONS[cardIdx % CALLOUT_CORNER_POSITIONS.length];
  }

  function resolveCalloutMaxWidth(cardCount, tGet) {
    if (cardCount >= 3) {
      return tGet('callout_card_max_width_multi_3', '26%');
    }
    if (cardCount === 2) {
      return tGet('callout_card_max_width_multi_2', '30%');
    }
    return tGet('callout_card_max_width', '42%');
  }

  function calloutSlideFrom(position) {
    const pos = normalizeCalloutPosition(position);
    if (pos === 'top_left' || pos === 'center_left' || pos === 'bottom_left' || pos === 'center') {
      return { x: -36, y: 0, opacity: 0 };
    }
    if (pos === 'top_right' || pos === 'center_right' || pos === 'bottom_right') {
      return { x: 36, y: 0, opacity: 0 };
    }
    if (pos === 'top_center') {
      return { x: 0, y: -24, opacity: 0 };
    }
    if (pos === 'bottom_center') {
      return { x: 0, y: 24, opacity: 0 };
    }
    return { opacity: 0 };
  }

  function scheduleCalloutCardEntrance(card, iconReady, entrance, position, tl, startTime) {
    const launch = function () {
      const when = Math.max(startTime, tl.time());
      animateCalloutCardEntrance(card, entrance, position, tl, when);
    };
    if (iconReady && typeof iconReady.then === 'function') {
      iconReady.then(launch);
      return;
    }
    launch();
  }

  function animateCalloutCardEntrance(card, entrance, position, tl, startTime) {
    if (entrance === 'callout_none' || entrance === 'none') {
      gsap.set(card, { x: 0, y: 0, opacity: 1 });
      return;
    }
    if (entrance === 'callout_slide') {
      const from = calloutSlideFrom(position);
      gsap.set(card, {
        x: from.x || 0,
        y: from.y || 0,
        opacity: from.opacity != null ? from.opacity : 0,
      });
      tl.to(
        card,
        { x: 0, y: 0, opacity: 1, duration: 0.6, ease: 'power2.out' },
        startTime
      );
      return;
    }
    gsap.set(card, { x: 0, y: 0, opacity: 0 });
    tl.to(card, { opacity: 1, duration: 0.55, ease: 'power2.out' }, startTime);
  }

  function calloutGridPlacement(position) {
    const pos = normalizeCalloutPosition(position);
    if (pos === 'top_right' || pos === 'center_right') {
      return { col: 2, row: 1, align: 'start', justify: 'end' };
    }
    if (pos === 'bottom_left' || pos === 'bottom_center') {
      return { col: 1, row: 2, align: 'end', justify: 'start' };
    }
    if (pos === 'bottom_right') {
      return { col: 2, row: 2, align: 'end', justify: 'end' };
    }
    if (pos === 'top_center') {
      return { col: 1, row: 1, align: 'start', justify: 'center' };
    }
    if (pos === 'center_left') {
      return { col: 1, row: 1, align: 'center', justify: 'start' };
    }
    return { col: 1, row: 1, align: 'start', justify: 'start' };
  }

  function parseCalloutInsetPx(bounds, inset) {
    const raw = String(inset || '6%').trim();
    const h = bounds.clientHeight || 0;
    if (raw.endsWith('%')) {
      return (parseFloat(raw) / 100) * h;
    }
    const n = parseFloat(raw);
    return Number.isFinite(n) ? n : 0;
  }

  function resetCalloutFonts(root) {
    root.querySelectorAll('[data-hf-base-fs]').forEach(function (el) {
      const base = parseFloat(el.dataset.hfBaseFs);
      if (Number.isFinite(base) && base > 0) {
        el.style.fontSize = base + 'px';
      }
    });
  }

  function shrinkCalloutFonts(root, floorPx) {
    let changed = false;
    root.querySelectorAll('[data-hf-base-fs]').forEach(function (el) {
      const fs = parseFloat(getComputedStyle(el).fontSize);
      if (!Number.isFinite(fs) || fs <= floorPx) return;
      el.style.fontSize = Math.max(floorPx, fs - 2) + 'px';
      changed = true;
    });
    return changed;
  }

  function fitCalloutCardToCell(cell) {
    const card = cell.querySelector('.hf-callout-card-outer, .hf-callout-card');
    if (!card || !cell.clientHeight) return;
    resetCalloutFonts(card);
    let guard = 0;
    while (card.scrollHeight > cell.clientHeight + 1 && guard < 14) {
      if (!shrinkCalloutFonts(card, 18)) break;
      guard += 1;
    }
  }

  function buildCalloutCard(callout, helpers) {
    const tGet = helpers.tGet;
    const tSize = helpers.tSize;
    const tWeight = helpers.tWeight;
    const tFont = helpers.tFont;

    const borderW = tSize('callout_card_border_width', '4px');
    const radius = tSize('callout_card_radius', '30px');
    const borderGradient = tGet(
      'callout_card_border_gradient',
      'linear-gradient(135deg, #FF5C26 0%, #FF5C26 28%, rgba(255,92,38,0.55) 70%, rgba(255,92,38,0.4) 100%)'
    );
    // Divider fades like the guide (solid orange → transparent), not a flat bar.
    const dividerBg = tGet(
      'callout_card_divider',
      tGet(
        'callout_card_divider_color',
        'linear-gradient(90deg, #FF5C26 0%, rgba(255, 92, 38, 0.35) 70%, rgba(255, 92, 38, 0) 100%)'
      )
    );

    // Card = translucent fill (image shows through) + gradient ring border via mask.
    const card = document.createElement('div');
    card.className = 'hf-callout-card-outer';
    card.style.position = 'relative';
    card.style.background = tGet(
      'callout_card_bg',
      'linear-gradient(90deg, rgba(0,0,0,0.72) 0%, rgba(0,0,0,0.45) 55%, rgba(0,0,0,0.12) 100%)'
    );
    card.style.borderRadius = radius;
    card.style.boxShadow = tGet('callout_card_shadow', '0 8px 24px rgba(0, 0, 0, 0.35)');
    card.style.boxSizing = 'border-box';
    card.style.width = '100%';
    card.style.maxWidth = '100%';
    card.style.overflow = 'hidden';

    // Gradient border ring: only the stroke shows, fill stays translucent underneath.
    const ring = document.createElement('div');
    ring.className = 'hf-callout-card-ring';
    ring.style.position = 'absolute';
    ring.style.left = '0';
    ring.style.top = '0';
    ring.style.width = '100%';
    ring.style.height = '100%';
    ring.style.borderRadius = radius;
    ring.style.padding = borderW;
    ring.style.background = borderGradient;
    ring.style.pointerEvents = 'none';
    ring.style.boxSizing = 'border-box';
    ring.style.webkitMask =
      'linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0)';
    ring.style.webkitMaskComposite = 'xor';
    ring.style.mask =
      'linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0)';
    ring.style.maskComposite = 'exclude';
    card.appendChild(ring);

    const paddingValue = tGet('callout_card_padding', '20px 24px');
    const inner = document.createElement('div');
    inner.className = 'hf-callout-card-inner';
    inner.style.position = 'relative';
    inner.style.padding = paddingValue;
    inner.style.boxSizing = 'border-box';
    inner.style.display = 'flex';
    inner.style.flexDirection = 'column';
    inner.style.alignItems = 'stretch';
    inner.style.minWidth = '0';
    inner.style.maxWidth = '100%';
    inner.style.borderRadius = radius;
    inner.style.overflow = 'hidden';

    // CSS padding shorthand → left inset (divider must cancel this to touch the card edge).
    const padParts = String(paddingValue).trim().split(/\s+/).filter(Boolean);
    let padLeft = '0px';
    if (padParts.length === 1) padLeft = padParts[0];
    else if (padParts.length === 2 || padParts.length === 3) padLeft = padParts[1];
    else if (padParts.length >= 4) padLeft = padParts[3];

    const headerText = String(callout.header || '').trim();
    const bodyText = String(callout.body || '').trim();
    const iconUrl = callout.url ? getProxyUrl(callout.url) : '';
    let iconReady = null;

    if (headerText || iconUrl) {
      const headerRow = document.createElement('div');
      headerRow.className = 'hf-callout-card-header';
      headerRow.style.display = 'flex';
      headerRow.style.flexWrap = 'nowrap';
      headerRow.style.alignItems = 'center';
      headerRow.style.gap = tSize('callout_card_header_gap', '12px');
      headerRow.style.minWidth = '0';

      if (iconUrl) {
        const iconWrap = document.createElement('div');
        iconWrap.style.flexShrink = '0';
        const iconSize = tSize('callout_card_icon_size', '75px');
        const img = document.createElement('img');
        img.alt = '';
        img.style.width = iconSize;
        img.style.height = iconSize;
        img.style.objectFit = 'contain';
        img.style.display = 'block';
        const knockoutWhite = tGet('callout_card_icon_knockout_white', 'true') !== 'false';
        if (calloutIconSrcCache[iconUrl]) {
          img.src = calloutIconSrcCache[iconUrl];
          img.dataset.hfKnocked = '1';
        } else if (knockoutWhite) {
          iconReady = ensureCalloutIconReady(iconUrl, true).then(function (src) {
            img.src = src;
            img.dataset.hfKnocked = '1';
          });
        } else {
          img.src = iconUrl;
        }
        iconWrap.appendChild(img);
        headerRow.appendChild(iconWrap);
      }

      if (headerText) {
        const headerEl = document.createElement('div');
        headerEl.textContent = headerText;
        headerEl.style.fontFamily = tFont('callout_card_header_font', "'Fira Sans', Arial, sans-serif");
        headerEl.style.fontWeight = tWeight('callout_card_header_font_weight', '700');
        headerEl.style.fontSize = tSize('callout_card_header_font_size', '32px');
        headerEl.style.color = tGet('callout_card_header_text_color', '#FF5C26');
        headerEl.style.textTransform = tGet('callout_card_header_text_transform', 'uppercase');
        headerEl.style.letterSpacing = '0.02em';
        headerEl.style.lineHeight = '1.1';
        // Wrap only at spaces — never mid-word (avoids orphan letters like "…MEN" / "T").
        headerEl.style.overflowWrap = 'normal';
        headerEl.style.wordBreak = 'normal';
        headerEl.style.hyphens = 'none';
        headerEl.style.minWidth = '0';
        headerEl.style.flex = '1 1 auto';
        headerEl.dataset.hfBaseFs = String(parseFloat(headerEl.style.fontSize) || 32);
        headerRow.appendChild(headerEl);
      }

      inner.appendChild(headerRow);
    }

    if (headerText || bodyText || iconUrl) {
      const divider = document.createElement('div');
      divider.className = 'hf-callout-card-divider';
      // Bleed to the card's left inner edge (guide: line touches the box on the left).
      const dividerWidth = tGet('callout_card_divider_width', '85%');
      divider.style.width = 'calc(' + dividerWidth + ' + ' + padLeft + ')';
      divider.style.height = tSize('callout_card_divider_thickness', '2px');
      divider.style.background = dividerBg;
      divider.style.border = 'none';
      divider.style.margin = tGet('callout_card_divider_margin', '12px 0');
      divider.style.marginLeft = '-' + padLeft;
      divider.style.flexShrink = '0';
      inner.appendChild(divider);
    }

    if (bodyText) {
      const bodyEl = document.createElement('div');
      bodyEl.className = 'hf-callout-card-body';
      // Strip trailing sentence periods so body never ends with a full stop.
      bodyEl.textContent = bodyText.replace(/\.+$/, '');
      bodyEl.style.fontFamily = tFont('callout_card_body_font', "'Fira Sans', Arial, sans-serif");
      bodyEl.style.fontWeight = tWeight('callout_card_body_font_weight', '700');
      bodyEl.style.fontSize = tSize('callout_card_body_font_size', '36px');
      bodyEl.style.color = tGet('callout_card_body_text_color', '#FFFFFF');
      bodyEl.style.textAlign = tGet('callout_card_body_text_align', 'left');
      bodyEl.style.lineHeight = tGet('callout_card_body_line_height', '1.25');
      bodyEl.style.letterSpacing = '-0.01em';
      bodyEl.style.minWidth = '0';
      bodyEl.style.maxWidth = '100%';
      bodyEl.style.whiteSpace = 'normal';
      bodyEl.style.overflowWrap = 'break-word';
      bodyEl.style.wordBreak = 'normal';
      bodyEl.style.hyphens = 'none';
      bodyEl.dataset.hfBaseFs = String(parseFloat(bodyEl.style.fontSize) || 36);
      // 0 / empty / "none" = show full body. Positive N still clamps (legacy).
      const maxLinesRaw = String(tGet('callout_card_body_max_lines', '0')).trim().toLowerCase();
      const maxLines = parseInt(maxLinesRaw, 10);
      if (maxLinesRaw && maxLinesRaw !== 'none' && maxLinesRaw !== '0' && maxLines > 0) {
        bodyEl.style.display = '-webkit-box';
        bodyEl.style.webkitBoxOrient = 'vertical';
        bodyEl.style.webkitLineClamp = String(maxLines);
        bodyEl.style.overflow = 'hidden';
      } else {
        bodyEl.style.display = 'block';
        bodyEl.style.overflow = 'visible';
      }
      inner.appendChild(bodyEl);
    }

    card.appendChild(inner);
    return { card: card, iconReady: iconReady };
  }

  function sync(opts) {
    const cue = opts.cue;
    const duration = opts.duration || 0;
    const playing = opts.playing;
    const elapsed = opts.elapsed || 0;
    const forceRestart = !!opts.forceRestart;

    if (!cue) {
      kill();
      return;
    }

    const cueKey = cue.slideIdx + ':' + cue.sceneId + ':' + cue.visualId;
    if (currentCueKey !== cueKey) {
      kill();
      currentCueKey = cueKey;
    }

    const target = document.querySelector('.hf-player-scene-slot.is-active .hf-hero-motion-target');
    if (!target) return;

    currentTarget = target;
    currentWrap = target.closest('.hf-hero-motion-wrapper') || target.parentElement || target;

    const animType = cue.animationType || 'none';
    let treatment = opts.treatment || 'still';
    if (animType !== 'none' && !isTopicTransition(cue)) {
      treatment = 'still';
    }

    if (duration <= 0) {
      if (treatment === 'center_split') {
        const s = queryShutters(currentWrap);
        if (s && !splitDone) {
          setShuttersClosed(s);
        }
      } else if (treatment === 'zoom_in') {
        gsap.set(target, { scale: 1.00, xPercent: 0, yPercent: 0, transformOrigin: '50% 50%' });
      }
      return;
    }

    let overlayContainer = currentWrap.querySelector('.hf-hero-overlay-container');
    if (forceRestart && overlayContainer) {
      overlayContainer.remove();
      overlayContainer = null;
    }

    if (!overlayContainer) {
      overlayContainer = document.createElement('div');
      overlayContainer.className = 'hf-hero-overlay-container';
      overlayContainer.style.position = 'absolute';
      overlayContainer.style.inset = '0';
      overlayContainer.style.zIndex = '20';
      overlayContainer.style.pointerEvents = 'none';
      currentWrap.appendChild(overlayContainer);
    }

    if (activeTween) {
      if (activeTweenDuration === duration) {
        if (playing) {
          activeTween.play();
          if (Math.abs(activeTween.time() - elapsed) > 0.25) activeTween.time(elapsed);
        } else {
          activeTween.pause();
          activeTween.time(elapsed);
        }
        return;
      } else {
        killTween();
        if (overlayContainer) {
          overlayContainer.remove();
          overlayContainer = null;
        }
        overlayContainer = document.createElement('div');
        overlayContainer.className = 'hf-hero-overlay-container';
        overlayContainer.style.position = 'absolute';
        overlayContainer.style.inset = '0';
        overlayContainer.style.zIndex = '20';
        overlayContainer.style.pointerEvents = 'none';
        currentWrap.appendChild(overlayContainer);
      }
    }

    const tl = gsap.timeline({ paused: true });

    let hasMotion = false;
    let baseStart = { scale: 1.00, xPercent: 0, yPercent: 0, transformOrigin: '50% 50%' };
    let baseEnd = {};

    if (treatment === 'zoom_in') {
      baseEnd.scale = 1.10;
      hasMotion = true;
    }

    if (isTopicTransition(cue)) {
      const theme = window.HFPlayerTheme;
      const tGet = (key, fallback) =>
        theme && theme.get ? (theme.get(key) || fallback) : fallback;
      const tSize = (key, fallback) =>
        theme && theme.fontSizePx ? theme.fontSizePx(key) : fallback;
      const tWeight = (key, fallback) =>
        theme && theme.fontWeight ? theme.fontWeight(key) : fallback;
      const tFont = (key, fallback) =>
        theme && theme.fontStack ? theme.fontStack(key) : fallback;

      const topicRaw = String(cue.topic || cue.slideTitle || 'Topic').trim();
      const secondaryColorCfg = String(tGet('topic_card_secondary_text_color', '') || '').trim();
      const useTwoTone =
        !!secondaryColorCfg && secondaryColorCfg.toLowerCase() !== 'none';

      const splitTopicLines = (text) => {
        const t = String(text || '').trim();
        if (!t) return { primary: 'Topic', secondary: '' };
        const amp = t.search(/\s+&\s+/);
        if (amp > 0) {
          return {
            primary: t.slice(0, amp).trim(),
            secondary: '& ' + t.slice(amp).replace(/^\s*&\s*/, '').trim(),
          };
        }
        const inMatch = t.match(/^(.*?)\s+in\s+(.+)$/i);
        if (inMatch) {
          return {
            primary: inMatch[1].trim(),
            secondary: 'IN ' + inMatch[2].trim(),
          };
        }
        const parts = t.split(/\s+/);
        if (parts.length >= 2) {
          return {
            primary: parts.slice(0, -1).join(' '),
            secondary: parts[parts.length - 1],
          };
        }
        return { primary: t, secondary: '' };
      };

      const insetRaw = tGet('topic_card_position_inset', '6%');
      const cardMaxWidthRaw = tGet('topic_card_max_width', '520px');
      const cardMaxWidthPx = parseCssLengthPx(cardMaxWidthRaw, 520);

      const wrap = document.createElement('div');
      wrap.className = 'hf-hero-topic-transition-wrap';
      wrap.style.pointerEvents = 'none';
      wrap.style.zIndex = '11';
      wrap.style.boxSizing = 'border-box';
      wrap.style.overflow = 'hidden';

      // Position shell: always the right side of the player frame.
      const shell = document.createElement('div');
      shell.className = 'hf-hero-topic-transition-shell';
      shell.style.position = 'absolute';
      shell.style.top = '50%';
      shell.style.transform = 'translateY(-50%)';
      shell.style.width = 'fit-content';
      shell.style.boxSizing = 'border-box';
      shell.style.marginLeft = 'auto';

      function layoutTopicInImage() {
        const rect = placeInFullFrame(wrap, currentWrap);
        const insetPx = resolveInsetPx(insetRaw, rect.w);
        const maxW = Math.max(80, Math.min(cardMaxWidthPx, rect.w - 2 * insetPx));
        shell.style.left = 'auto';
        shell.style.right = insetPx + 'px';
        shell.style.maxWidth = maxW + 'px';
      }

      // GSAP animates this layer so slide transforms don't fight shell's translateY(-50%).
      const animWrap = document.createElement('div');
      animWrap.className = 'hf-hero-topic-transition-anim';
      animWrap.style.maxWidth = '100%';
      animWrap.style.width = 'fit-content';
      animWrap.style.pointerEvents = 'none';

      const card = document.createElement('div');
      card.className = 'hf-hero-topic-transition-card';
      card.style.position = 'relative';
      card.style.background = tGet('topic_card_bg', '#000066');
      card.style.border = tGet('topic_card_border', 'none');
      card.style.borderRadius = tSize('topic_card_radius', '22px');
      card.style.padding = tGet('topic_card_padding', '24px 32px');
      card.style.maxWidth = '100%';
      card.style.width = 'fit-content';
      card.style.boxSizing = 'border-box';
      card.style.pointerEvents = 'auto';
      card.style.display = 'block';
      card.style.boxShadow = tGet('topic_card_shadow', '0 10px 30px rgba(0,0,0,0.4)');
      card.style.fontFamily = tFont(
        'topic_card_font',
        "'Public Sans', 'Open Sans', 'Segoe UI', Arial, sans-serif"
      );
      card.style.textAlign = tGet('topic_card_text_align', 'center');

      const topicWeight = tWeight('topic_card_font_weight', '700');
      const topicLineHeight = tGet('topic_card_line_height', '1.15');
      const topicAlign = tGet('topic_card_text_align', 'center');
      const topicTransform = tGet('topic_card_text_transform', 'none');

      if (useTwoTone) {
        const lines = splitTopicLines(topicRaw);
        const primarySize = tSize('topic_card_font_size', '60px');
        const secondarySize = tSize('topic_card_secondary_font_size', '70px');
        const primaryColor = tGet('topic_card_text_color', '#3B3B3B');
        const secondaryColor = secondaryColorCfg || '#FF5C26';
        const primaryHtml =
          `<div style="font-size: ${primarySize}; font-weight: ${topicWeight}; color: ${primaryColor}; ` +
          `line-height: ${topicLineHeight}; font-family: inherit; letter-spacing: -0.02em; ` +
          `overflow-wrap: break-word; word-break: break-word; hyphens: auto; ` +
          `text-align: ${topicAlign}; text-transform: ${topicTransform};">${lines.primary}</div>`;
        const secondaryHtml = lines.secondary
          ? (`<div style="font-size: ${secondarySize}; font-weight: ${topicWeight}; color: ${secondaryColor}; ` +
            `line-height: ${topicLineHeight}; font-family: inherit; letter-spacing: -0.02em; ` +
            `overflow-wrap: break-word; word-break: break-word; hyphens: auto; ` +
            `text-align: ${topicAlign}; text-transform: ${topicTransform}; margin-top: 0.02em;">${lines.secondary}</div>`)
          : '';
        card.innerHTML = primaryHtml + secondaryHtml;
      } else {
        const topicSize = tSize('topic_card_font_size', '44px');
        const topicColor = tGet('topic_card_text_color', '#ffffff');
        card.innerHTML =
          `<div style="font-size: ${topicSize}; font-weight: ${topicWeight}; color: ${topicColor}; ` +
          `line-height: ${topicLineHeight}; font-family: inherit; letter-spacing: -0.015em; ` +
          `overflow-wrap: break-word; word-break: break-word; hyphens: auto; ` +
          `text-align: ${topicAlign}; text-transform: ${topicTransform};">${topicRaw}</div>`;
      }

      animWrap.appendChild(card);
      shell.appendChild(animWrap);
      wrap.appendChild(shell);
      overlayContainer.appendChild(wrap);

      layoutTopicInImage();
      if (!(target.complete && target.naturalWidth > 0)) {
        target.addEventListener('load', layoutTopicInImage, { once: true });
      }
      requestAnimationFrame(layoutTopicInImage);

      // Keep the hero image visible. Fading target 0→1 caused: image flash → blank → image+card.
      gsap.set(target, { opacity: 1 });
      animWrap.style.opacity = '0';

      if (treatment === 'fade_in_right') {
        tl.fromTo(animWrap, { opacity: 0, x: 200 }, { opacity: 1, x: 0, duration: 0.8, ease: 'power2.out' }, 0.15);
      } else if (treatment === 'fade_in_top_right') {
        tl.fromTo(animWrap, { opacity: 0, x: 150, y: -150 }, { opacity: 1, x: 0, y: 0, duration: 0.8, ease: 'power2.out' }, 0.15);
      } else if (treatment === 'fade_in_bottom_right') {
        tl.fromTo(animWrap, { opacity: 0, x: 150, y: 150 }, { opacity: 1, x: 0, y: 0, duration: 0.8, ease: 'power2.out' }, 0.15);
      } else if (treatment === 'fade_in_only') {
        tl.fromTo(animWrap, { opacity: 0 }, { opacity: 1, duration: 0.8, ease: 'power2.out' }, 0.15);
      } else {
        gsap.set(animWrap, { opacity: 1, x: 0, y: 0 });
      }
    }
    else if (animType === 'text_label') {
      const timingOpts = overlayTimingOpts(opts, cue, duration);
      const labelStart = resolveOverlayTime(cue.triggerPhrase, timingOpts, 0.15);
      gsap.set(target, { scale: 1.00, xPercent: 0, yPercent: 0, transformOrigin: '50% 50%' });
      tl.to(target, { xPercent: -15, scale: 0.85, duration: 0.45, ease: 'power2.out' }, labelStart);

      const theme = window.HFPlayerTheme;
      const tGet = (key, fallback) =>
        theme && theme.get ? (theme.get(key) || fallback) : fallback;
      const tSize = (key, fallback) =>
        theme && theme.fontSizePx ? theme.fontSizePx(key) : fallback;
      const tWeight = (key, fallback) =>
        theme && theme.fontWeight ? theme.fontWeight(key) : fallback;
      const tFont = (key, fallback) =>
        theme && theme.fontStack ? theme.fontStack(key) : fallback;

      const labelSize = tSize('hero_label_font_size', '26px');
      const labelWeight = tWeight('hero_label_font_weight', '700');
      const labelColor = tGet('hero_label_text_color', '#ffffff');
      const labelAlign = tGet('hero_label_text_align', 'center');
      const labelLineHeight = tGet('hero_label_line_height', '1.25');
      const labelFont = tFont('hero_label_font', "'Public Sans', Arial, sans-serif");
      const labelWhiteSpace = tGet('hero_label_white_space', 'nowrap');
      const labelText = cue.labelText || 'Information';

      const card = document.createElement('div');
      card.className = 'hf-hero-text-card';
      card.innerHTML =
        `<div style="font-size: ${labelSize}; font-weight: ${labelWeight}; color: ${labelColor}; ` +
        `line-height: ${labelLineHeight}; font-family: inherit; letter-spacing: -0.01em; ` +
        `text-align: ${labelAlign}; white-space: ${labelWhiteSpace};">${labelText}</div>`;
      card.style.position = 'absolute';
      card.style.top = '50%';
      card.style.right = '8%';
      card.style.transform = 'translateY(-50%) translateX(60px)';
      card.style.opacity = '0';
      card.style.background = tGet('hero_label_bg', '#f05725');
      card.style.borderRadius = tSize('hero_label_radius', '24px');
      card.style.padding = tGet('hero_label_padding', '18px 36px');
      card.style.width = 'fit-content';
      card.style.maxWidth = tGet('hero_label_max_width', '90%');
      card.style.display = 'inline-block';
      card.style.boxShadow = tGet('hero_label_shadow', '0 8px 24px rgba(0,0,0,0.28)');
      card.style.boxSizing = 'border-box';
      card.style.pointerEvents = 'auto';
      card.style.fontFamily = labelFont;
      overlayContainer.appendChild(card);

      tl.fromTo(card, { opacity: 0, x: 60 }, { opacity: 1, x: 0, duration: 0.5, ease: 'back.out(1.4)' }, labelStart + 0.05);
    } 
    else if (animType === 'bbox_highlight' && cue.bboxHighlights && cue.bboxHighlights.length > 0) {
      const bboxTimingOpts = overlayTimingOpts(opts, cue, duration);
      const bboxStyle = String(opts.bboxTreatment || 'bbox_draw').toLowerCase();
      const themeHelpers = themeGetters();
      const tGet = themeHelpers.tGet;
      const drawStrokeColor = tGet('bbox_draw_stroke_color', '#f05523');
      const drawStrokeWidth = parseFloat(tGet('bbox_draw_stroke_width', '4.5')) || 4.5;

      const drawBboxDrawStyle = function () {
        overlayContainer.querySelectorAll('.hf-hero-bbox').forEach(function (node) {
          node.remove();
        });

        const rect = getVisibleImageRect(currentWrap, target);
        const validHighlights = filterValidBboxHighlights(cue);
        const numBoxes = validHighlights.length;
        if (numBoxes === 0) return;

        const startOffset = 0.2;
        const stagger = numBoxes > 1 ? Math.min(0.25, (duration - startOffset - 1.5) / (numBoxes - 1)) : 0;
        const safeStagger = Math.max(0, stagger);
        const maxEndTime = duration - 0.8;

        validHighlights.forEach(function (highlight, idx) {
          const metrics = bboxBoxMetrics(rect, highlight.box_2d);
          const width = metrics.width;
          const height = metrics.height;

          const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          svgEl.className = 'hf-hero-bbox';
          svgEl.setAttribute('width', width);
          svgEl.setAttribute('height', height);
          svgEl.style.position = 'absolute';
          svgEl.style.left = metrics.left + 'px';
          svgEl.style.top = metrics.top + 'px';
          svgEl.style.width = width + 'px';
          svgEl.style.height = height + 'px';
          svgEl.style.pointerEvents = 'none';
          svgEl.style.overflow = 'visible';
          svgEl.style.filter = 'drop-shadow(0px 0px 6px rgba(240, 85, 35, 0.65))';

          const paths = appendSplitBboxPaths(svgEl, highlight, width, height, drawStrokeColor, drawStrokeWidth);
          overlayContainer.appendChild(svgEl);

          const lastBoxStart = startOffset + (numBoxes - 1) * safeStagger;
          const drawDuration = Math.max(0.4, Math.min(1.5, maxEndTime - lastBoxStart));
          const startTime = resolveOverlayTime(
            highlight.triggerPhrase,
            bboxTimingOpts,
            startOffset + idx * safeStagger
          );
          animateSplitBboxDraw(tl, svgEl, paths, width, height, startTime, drawDuration);
        });
      };

      const drawBboxSpotlightStyle = function () {
        overlayContainer.querySelectorAll('.hf-hero-bbox-spotlight').forEach(function (node) {
          node.remove();
        });

        const validHighlights = filterValidBboxHighlights(cue);
        if (validHighlights.length === 0) return;

        const veilOpacity = parseFloat(tGet('bbox_spotlight_veil_opacity', '0.75'));
        const safeVeilOpacity = Number.isFinite(veilOpacity) ? veilOpacity : 0.75;
        const borderColor = tGet('bbox_spotlight_border_color', '#F9E400');
        const borderWidth = parseFloat(tGet('bbox_spotlight_border_width', '3')) || 3;
        const borderDash = tGet('bbox_spotlight_border_dash', '18 10');
        const marchCycle = parseFloat(tGet('bbox_spotlight_border_march_cycle', '1.1')) || 1.1;
        const cornerRadius = parseFloat(tGet('bbox_spotlight_corner_radius', '28')) || 28;
        const highlightDelayRaw = parseFloat(tGet('bbox_spotlight_highlight_delay', '0.28'));
        const highlightDelay = Number.isFinite(highlightDelayRaw) ? Math.max(0, highlightDelayRaw) : 0.28;
        const imgSrc = target.currentSrc || target.src;

        const bounds = document.createElement('div');
        bounds.className = 'hf-hero-bbox-spotlight';
        bounds.style.pointerEvents = 'none';
        bounds.style.overflow = 'hidden';
        bounds.style.boxSizing = 'border-box';
        overlayContainer.appendChild(bounds);

        const veil = document.createElement('div');
        veil.className = 'hf-hero-bbox-spotlight-veil';
        veil.style.position = 'absolute';
        veil.style.inset = '0';
        veil.style.pointerEvents = 'none';
        veil.style.background = 'rgba(0, 0, 0, ' + safeVeilOpacity + ')';
        veil.style.opacity = '0';
        bounds.appendChild(veil);
        const numBoxes = validHighlights.length;
        const startOffset = 0.2;
        const stagger = numBoxes > 1 ? Math.min(0.25, (duration - startOffset - 1.5) / (numBoxes - 1)) : 0;
        const safeStagger = Math.max(0, stagger);

        // Precompute start times so the surrounding dark veil is synced to the trigger phrase
        const startTimes = validHighlights.map(function (highlight, idx) {
          return resolveOverlayTime(
            highlight.triggerPhrase,
            bboxTimingOpts,
            startOffset + idx * safeStagger
          );
        });

        const veilStartTime = startTimes.length ? Math.min.apply(null, startTimes) : startOffset;
        const remainingAfterVeil = Math.max(0.2, duration - veilStartTime);
        const veilDuration = Math.min(0.4, remainingAfterVeil);
        tl.to(veil, { opacity: 1, duration: veilDuration, ease: 'power2.out' }, veilStartTime);

        const spotlightNodes = [];

        validHighlights.forEach(function (highlight, idx) {
          const windowWrap = document.createElement('div');
          windowWrap.className = 'hf-hero-bbox-spotlight-window';
          windowWrap.style.position = 'absolute';
          windowWrap.style.overflow = 'hidden';
          windowWrap.style.opacity = '0';
          windowWrap.style.pointerEvents = 'none';

          const brightImg = document.createElement('img');
          brightImg.className = 'hf-hero-bbox-spotlight-bright';
          brightImg.alt = '';
          brightImg.draggable = false;
          brightImg.src = imgSrc;
          brightImg.style.position = 'absolute';
          brightImg.style.display = 'block';
          brightImg.style.maxWidth = 'none';
          brightImg.style.pointerEvents = 'none';
          windowWrap.appendChild(brightImg);

          const borderWrap = document.createElement('div');
          borderWrap.className = 'hf-hero-bbox-spotlight-border';
          borderWrap.style.position = 'absolute';
          borderWrap.style.pointerEvents = 'none';
          borderWrap.style.opacity = '0';
          borderWrap.style.overflow = 'visible';

          const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          svgEl.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
          svgEl.style.position = 'absolute';
          svgEl.style.inset = '0';
          svgEl.style.overflow = 'visible';
          svgEl.style.pointerEvents = 'none';

          const borderShape = document.createElementNS('http://www.w3.org/2000/svg', highlight.shape === 'circle' ? 'ellipse' : 'rect');
          borderShape.setAttribute('fill', 'none');
          borderShape.setAttribute('stroke', borderColor);
          borderShape.setAttribute('stroke-width', String(borderWidth));
          borderShape.setAttribute('stroke-dasharray', borderDash);
          borderShape.setAttribute('stroke-linecap', 'round');
          borderShape.setAttribute('stroke-linejoin', 'round');
          svgEl.appendChild(borderShape);
          borderWrap.appendChild(svgEl);

          bounds.appendChild(windowWrap);
          bounds.appendChild(borderWrap);

          const startTime = startTimes[idx];
          // Darken first, then punch the bright cutout + yellow border after a short beat.
          const highlightTime = Math.min(Math.max(0, startTime + highlightDelay), Math.max(0, duration - 0.12));

          spotlightNodes.push({
            highlight: highlight,
            windowWrap: windowWrap,
            brightImg: brightImg,
            borderWrap: borderWrap,
            svgEl: svgEl,
            borderShape: borderShape,
            startTime: startTime,
          });

          tl.fromTo(windowWrap, { opacity: 0 }, { opacity: 1, duration: 0.22, ease: 'power1.out' }, highlightTime);
          tl.fromTo(borderWrap, { opacity: 0 }, { opacity: 1, duration: 0.28, ease: 'power1.out' }, highlightTime);
          animateSpotlightBorderMarch(tl, borderShape, highlightTime, duration, borderDash, marchCycle);
        });

        function layoutSpotlightWindows() {
          const rect = placeInVisibleImage(bounds, currentWrap, target);
          spotlightNodes.forEach(function (node) {
            const metrics = bboxBoxMetrics(rect, node.highlight.box_2d);
            const width = metrics.width;
            const height = metrics.height;
            const inset = borderWidth / 2;

            node.windowWrap.style.left = metrics.relLeft + 'px';
            node.windowWrap.style.top = metrics.relTop + 'px';
            node.windowWrap.style.width = width + 'px';
            node.windowWrap.style.height = height + 'px';

            node.borderWrap.style.left = metrics.relLeft + 'px';
            node.borderWrap.style.top = metrics.relTop + 'px';
            node.borderWrap.style.width = width + 'px';
            node.borderWrap.style.height = height + 'px';

            node.svgEl.setAttribute('width', width);
            node.svgEl.setAttribute('height', height);
            node.svgEl.style.width = width + 'px';
            node.svgEl.style.height = height + 'px';

            if (node.highlight.shape === 'circle') {
              node.windowWrap.style.borderRadius = '50%';
              node.borderShape.setAttribute('cx', String(width / 2));
              node.borderShape.setAttribute('cy', String(height / 2));
              node.borderShape.setAttribute('rx', String(Math.max(0, width / 2 - inset)));
              node.borderShape.setAttribute('ry', String(Math.max(0, height / 2 - inset)));
            } else {
              const rx = Math.min(cornerRadius, width / 2, height / 2);
              node.windowWrap.style.borderRadius = rx + 'px';
              node.borderShape.setAttribute('x', String(inset));
              node.borderShape.setAttribute('y', String(inset));
              node.borderShape.setAttribute('width', String(Math.max(0, width - borderWidth)));
              node.borderShape.setAttribute('height', String(Math.max(0, height - borderWidth)));
              node.borderShape.setAttribute('rx', String(rx));
              node.borderShape.setAttribute('ry', String(rx));
            }

            node.brightImg.style.width = rect.w + 'px';
            node.brightImg.style.height = rect.h + 'px';
            node.brightImg.style.left = (-metrics.relLeft) + 'px';
            node.brightImg.style.top = (-metrics.relTop) + 'px';
          });
        }

        layoutSpotlightWindows();
        if (!(target.complete && target.naturalWidth > 0)) {
          target.addEventListener('load', layoutSpotlightWindows, { once: true });
        }
        requestAnimationFrame(layoutSpotlightWindows);
      };

      const renderBboxes = function () {
        if (bboxStyle === 'bbox_spotlight') {
          drawBboxSpotlightStyle();
        } else {
          drawBboxDrawStyle();
        }
      };

      if (target.complete && target.naturalWidth > 0) {
        renderBboxes();
      } else {
        target.addEventListener('load', renderBboxes, { once: true });
      }
    } 
    else if (animType === 'callout_card' && cue.calloutCards && cue.calloutCards.length > 0) {
      const theme = window.HFPlayerTheme;
      const tGet = (key, fallback) =>
        theme && theme.get ? (theme.get(key) || fallback) : fallback;
      const tSize = (key, fallback) =>
        theme && theme.fontSizePx ? theme.fontSizePx(key) : fallback;
      const tWeight = (key, fallback) =>
        theme && theme.fontWeight ? theme.fontWeight(key) : fallback;
      const tFont = (key, fallback) =>
        theme && theme.fontStack ? theme.fontStack(key) : fallback;
      const helpers = { tGet: tGet, tSize: tSize, tWeight: tWeight, tFont: tFont };

      const calloutTimingOpts = overlayTimingOpts(opts, cue, duration);
      const cardCount = cue.calloutCards.length;
      const cardTimes = cue.calloutCards.map(function (callout, idx) {
        return resolveOverlayTime(callout.triggerPhrase, calloutTimingOpts, 0.5 + idx * 0.12);
      });
      const sceneStart = cardTimes.length ? Math.min.apply(null, cardTimes) : 0.5;
      const inset = tGet('callout_card_position_inset', '6%');
      const gridGap = parseFloat(tGet('callout_card_stack_gap', '16')) || 16;

      const bounds = document.createElement('div');
      bounds.className = 'hf-callout-image-bounds';
      bounds.style.pointerEvents = 'none';
      bounds.style.overflow = 'hidden';
      bounds.style.boxSizing = 'border-box';
      overlayContainer.appendChild(bounds);

      const sceneOverlay = document.createElement('div');
      sceneOverlay.className = 'hf-callout-scene-overlay';
      sceneOverlay.style.position = 'absolute';
      sceneOverlay.style.left = '0';
      sceneOverlay.style.top = '0';
      sceneOverlay.style.width = '100%';
      sceneOverlay.style.height = '100%';
      sceneOverlay.style.pointerEvents = 'none';
      sceneOverlay.style.background = tGet(
        'callout_scene_overlay',
        'linear-gradient(180deg, rgba(0,0,0,0.68) 0%, rgba(0,0,0,0.31) 55%, rgba(0,0,0,0) 100%)'
      );
      sceneOverlay.style.opacity = '0';
      bounds.appendChild(sceneOverlay);

      const grid = document.createElement('div');
      grid.className = 'hf-callout-grid';
      grid.style.position = 'absolute';
      grid.style.display = 'grid';
      grid.style.gridTemplateColumns = 'minmax(0, 1fr) minmax(0, 1fr)';
      grid.style.gridTemplateRows = 'minmax(0, 1fr) minmax(0, 1fr)';
      grid.style.gap = gridGap + 'px';
      grid.style.boxSizing = 'border-box';
      grid.style.pointerEvents = 'none';
      grid.style.minWidth = '0';
      grid.style.minHeight = '0';
      bounds.appendChild(grid);

      gsap.set(target, { filter: 'brightness(1)' });
      tl.to(target, { filter: 'brightness(0.55)', duration: 0.6, ease: 'power1.out' }, sceneStart);
      tl.to(sceneOverlay, { opacity: 1, duration: 0.6, ease: 'power1.out' }, sceneStart);

      const cells = [];

      cue.calloutCards.forEach(function (callout, idx) {
        const built = buildCalloutCard(callout, helpers);
        const card = built.card;
        card.className = (card.className || '') + ' hf-callout-card';
        card.style.pointerEvents = 'auto';
        card.style.maxHeight = '100%';
        gsap.set(card, { opacity: 0, x: 0, y: 0 });

        const resolvedPos = resolveCalloutPosition(callout.position, idx, cardCount);
        const place = calloutGridPlacement(resolvedPos);
        const cell = document.createElement('div');
        cell.className = 'hf-callout-cell';
        cell.style.gridColumn = String(place.col);
        cell.style.gridRow = String(place.row);
        cell.style.minWidth = '0';
        cell.style.minHeight = '0';
        cell.style.overflow = 'hidden';
        cell.style.display = 'flex';
        cell.style.flexDirection = 'column';
        cell.style.alignItems = place.justify === 'end' ? 'flex-end' : place.justify === 'center' ? 'center' : 'flex-start';
        cell.style.justifyContent =
          place.align === 'end' ? 'flex-end' : place.align === 'center' ? 'center' : 'flex-start';
        cell.appendChild(card);
        grid.appendChild(cell);
        cells.push(cell);

        cell.querySelectorAll('img').forEach(function (img) {
          if (!img.complete) {
            img.addEventListener('load', function () {
              fitCalloutCardToCell(cell);
            }, { once: true });
          }
        });

        const meta =
          window.HFTreatmentRotator && window.HFTreatmentRotator.getCalloutEntrance
            ? window.HFTreatmentRotator.getCalloutEntrance(cue, idx)
            : { entrance: 'callout_fade', position: resolvedPos, occurrence: 0 };
        const cardStart = cardTimes[idx];
        scheduleCalloutCardEntrance(card, built.iconReady, meta.entrance, resolvedPos, tl, cardStart);
      });

      function layoutCalloutGrid() {
        // Default: full frame (sharp image + blur letterbox). Narrow portrait
        // heroes crush cards if bounds are limited to the contain-rect only.
        const boundsMode = String(tGet('callout_card_bounds', 'frame') || 'frame')
          .trim()
          .toLowerCase();
        if (boundsMode === 'image' || boundsMode === 'visible' || boundsMode === 'contain') {
          placeInVisibleImage(bounds, currentWrap, target);
        } else {
          placeInFullFrame(bounds, currentWrap);
        }
        const pad = Math.round(parseCalloutInsetPx(bounds, inset));
        grid.style.top = pad + 'px';
        grid.style.left = pad + 'px';
        grid.style.right = pad + 'px';
        grid.style.bottom = pad + 'px';
        cells.forEach(fitCalloutCardToCell);
      }
      layoutCalloutGrid();
      if (!(target.complete && target.naturalWidth > 0)) {
        target.addEventListener('load', layoutCalloutGrid, { once: true });
      }
      requestAnimationFrame(layoutCalloutGrid);
    }
    else if (animType === 'icon_overlay' && cue.iconOverlays && cue.iconOverlays.length > 0) {
      const iconTimingOpts = overlayTimingOpts(opts, cue, duration);
      const iconTimes = cue.iconOverlays.map(function (icon, idx) {
        return resolveOverlayTime(icon.triggerPhrase, iconTimingOpts, 0.8 + 0.2 + idx * 0.15);
      });
      const sceneStart = iconTimes.length ? Math.min.apply(null, iconTimes) : 0.8;
      gsap.set(target, { filter: 'brightness(1)' });
      tl.to(target, { filter: 'brightness(0.35)', duration: 0.6, ease: 'power1.out' }, sceneStart);

      const flexContainer = document.createElement('div');
      flexContainer.className = 'hf-hero-icons-flex';
      flexContainer.style.position = 'absolute';
      flexContainer.style.inset = '0';
      flexContainer.style.display = 'flex';
      flexContainer.style.alignItems = 'center';
      flexContainer.style.justifyContent = 'center';
      flexContainer.style.gap = '20px';
      flexContainer.style.pointerEvents = 'none';
      overlayContainer.appendChild(flexContainer);

      cue.iconOverlays.forEach((icon, idx) => {
        const cardEl = document.createElement('div');
        cardEl.className = 'hf-hero-icon-card';
        cardEl.style.display = 'flex';
        cardEl.style.alignItems = 'center';
        cardEl.style.justifyContent = 'center';
        cardEl.style.background = 'rgba(25, 29, 38, 0.95)';
        cardEl.style.border = '1px solid rgba(240, 85, 35, 0.25)';
        cardEl.style.borderRadius = '12px';
        cardEl.style.padding = '12px';
        cardEl.style.boxShadow = '0 10px 25px rgba(0,0,0,0.35)';
        cardEl.style.boxSizing = 'border-box';
        cardEl.style.pointerEvents = 'auto';

        const imgUrl = getProxyUrl(icon.url);
        cardEl.innerHTML = `<img src="${imgUrl}" alt="icon" style="width: 80px; height: 80px; border-radius: 6px; object-fit: contain; background: #ffffff; border: 1.5px solid #f05523; display: block;" />`;

        flexContainer.appendChild(cardEl);

        tl.fromTo(cardEl, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.5, ease: 'back.out(1.2)' }, iconTimes[idx]);
      });
    }

    if (hasMotion) {
      gsap.set(target, baseStart);
      tl.to(target, {
        ...baseEnd,
        duration: duration || 1,
        ease: 'sine.inOut',
      }, 0);
    }

    if (treatment === 'center_split') {
      target.classList.add('is-active-motion');
      gsap.set(target, { scale: 1, xPercent: 0, yPercent: 0, transformOrigin: '50% 50%', clipPath: 'none' });
      syncCenterSplit(currentWrap, playing, forceRestart);
    }

    activeTween = tl;
    activeTweenDuration = duration;
    activeTween.time(elapsed);
    if (playing) activeTween.play();
  }

  function setInitialState(target, treatment) {
    const vars = getTweenVars(treatment);
    if (vars) gsap.set(target, vars.start);
  }

  function getTweenVars(treatment) {
    if (treatment === 'zoom_in') {
      return {
        start: { scale: 1.00, xPercent: 0, yPercent: 0, transformOrigin: '50% 50%' },
        end: { scale: 1.10 },
      };
    }
    return null;
  }

  return {
    sync: sync,
    kill: kill,
    warmCalloutIcons: warmCalloutIcons,
    ensureCalloutIconReady: ensureCalloutIconReady,
  };
})();
