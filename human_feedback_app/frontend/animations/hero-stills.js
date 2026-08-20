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

  function getProxyUrl(url) {
    if (!url) return '';
    if (url.indexOf('drive.google.com') !== -1 || url.indexOf('docs.google.com') !== -1 || url.indexOf('googleusercontent.com') !== -1) {
      return '/api/assets/image?url=' + encodeURIComponent(url);
    }
    return url;
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
      overlayContainer.style.zIndex = '10';
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
        overlayContainer.style.zIndex = '10';
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
      const card = document.createElement('div');
      card.className = 'hf-hero-topic-transition-card';
      card.style.position = 'absolute';
      card.style.top = '50%';
      card.style.right = '8%';
      card.style.transform = 'translateY(-50%)';
      card.style.background = '#000066';
      card.style.borderRadius = '18px';
      card.style.padding = '24px 36px';
      card.style.maxWidth = '480px';
      card.style.width = 'fit-content';
      card.style.boxSizing = 'border-box';
      card.style.pointerEvents = 'auto';
      card.style.display = 'inline-block';
      card.style.boxShadow = '0 10px 30px rgba(0,0,0,0.4)';
      card.style.fontFamily = "'Public Sans', 'Open Sans', 'Segoe UI', Arial, sans-serif";
      card.style.textAlign = 'left';
      
      card.innerHTML = `
        <div style="font-size: 32px; font-weight: 800; color: #ffffff; line-height: 1.15; font-family: inherit; letter-spacing: -0.015em; word-wrap: break-word;">${cue.topic || cue.slideTitle || 'Topic'}</div>
      `;
      overlayContainer.appendChild(card);

      if (treatment === 'fade_in_right') {
        tl.fromTo(target, { opacity: 0 }, { opacity: 1, duration: 0.8, ease: 'power2.out' }, 0);
        tl.fromTo(card, { opacity: 0, x: 200 }, { opacity: 1, x: 0, duration: 0.8, ease: 'power2.out' }, 0.8);
      } else if (treatment === 'fade_in_top_right') {
        tl.fromTo(target, { opacity: 0 }, { opacity: 1, duration: 0.8, ease: 'power2.out' }, 0);
        tl.fromTo(card, { opacity: 0, x: 150, y: -150 }, { opacity: 1, x: 0, y: 0, duration: 0.8, ease: 'power2.out' }, 0.8);
      } else if (treatment === 'fade_in_bottom_right') {
        tl.fromTo(target, { opacity: 0 }, { opacity: 1, duration: 0.8, ease: 'power2.out' }, 0);
        tl.fromTo(card, { opacity: 0, x: 150, y: 150 }, { opacity: 1, x: 0, y: 0, duration: 0.8, ease: 'power2.out' }, 0.8);
      } else if (treatment === 'fade_in_only') {
        gsap.set(target, { opacity: 1 });
        tl.fromTo(card, { opacity: 0 }, { opacity: 1, duration: 0.8, ease: 'power2.out' }, 0.2);
      }
    }
    else if (animType === 'text_label') {
      gsap.set(target, { scale: 1.00, xPercent: 0, yPercent: 0, transformOrigin: '50% 50%' });
      tl.to(target, { xPercent: -15, scale: 0.85, duration: 0.45, ease: 'power2.out' }, 0);

      const card = document.createElement('div');
      card.className = 'hf-hero-text-card';
      card.innerHTML = `<div style="font-size: 18px; font-weight: 800; color: #ffffff; line-height: 1.35; font-family: 'Public Sans', Arial, sans-serif; letter-spacing: -0.01em; text-align: center;">${cue.labelText || 'Information'}</div>`;
      card.style.position = 'absolute';
      card.style.top = '50%';
      card.style.right = '8%';
      card.style.transform = 'translateY(-50%) translateX(60px)';
      card.style.opacity = '0';
      card.style.background = '#f05725'; // Solid SkillCat Orange
      card.style.borderRadius = '12px';
      card.style.padding = '16px 24px';
      card.style.width = 'fit-content';
      card.style.maxWidth = '320px';
      card.style.display = 'inline-block';
      card.style.boxShadow = '0 10px 30px rgba(0,0,0,0.4)';
      card.style.boxSizing = 'border-box';
      card.style.pointerEvents = 'auto';
      card.style.fontFamily = "'Public Sans', Arial, sans-serif";
      overlayContainer.appendChild(card);

      tl.fromTo(card, { opacity: 0, x: 60 }, { opacity: 1, x: 0, duration: 0.5, ease: 'back.out(1.4)' }, 0.15);
    } 
    else if (animType === 'bbox_highlight' && cue.bboxHighlights && cue.bboxHighlights.length > 0) {
      const drawBBoxes = () => {
        const oldBoxes = overlayContainer.querySelectorAll('.hf-hero-bbox');
        oldBoxes.forEach(b => b.remove());

        const rect = getVisibleImageRect(currentWrap, target);
        const validHighlights = [];
        
        cue.bboxHighlights.forEach((highlight) => {
          const box = highlight.box_2d;
          if (!box || box.length !== 4) return;
          const ymin = box[0], xmin = box[1], ymax = box[2], xmax = box[3];

          // Check if it overlaps with any previously accepted box to prevent intersection
          let intersectsExisting = false;
          for (const val of validHighlights) {
            const acc = val.box_2d;
            const [a_ymin, a_xmin, a_ymax, a_xmax] = acc;
            const hOverlap = xmin < a_xmax && xmax > a_xmin;
            const vOverlap = ymin < a_ymax && ymax > a_ymin;
            if (hOverlap && vOverlap) {
              intersectsExisting = true;
              break;
            }
          }
          if (!intersectsExisting) {
            validHighlights.push(highlight);
          }
        });

        const numBoxes = validHighlights.length;
        if (numBoxes === 0) return;

        const startOffset = 0.2;
        const stagger = numBoxes > 1 ? Math.min(0.25, (duration - startOffset - 1.5) / (numBoxes - 1)) : 0;
        const safeStagger = Math.max(0, stagger);
        const maxEndTime = duration - 0.8;

        validHighlights.forEach((highlight, idx) => {
          const box = highlight.box_2d;
          const ymin = box[0], xmin = box[1], ymax = box[2], xmax = box[3];

          const left = rect.x + (xmin / 1000) * rect.w;
          const top = rect.y + (ymin / 1000) * rect.h;
          const width = ((xmax - xmin) / 1000) * rect.w;
          const height = ((ymax - ymin) / 1000) * rect.h;

          const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          svgEl.className = 'hf-hero-bbox';
          svgEl.setAttribute('width', width);
          svgEl.setAttribute('height', height);
          svgEl.style.position = 'absolute';
          svgEl.style.left = left + 'px';
          svgEl.style.top = top + 'px';
          svgEl.style.width = width + 'px';
          svgEl.style.height = height + 'px';
          svgEl.style.pointerEvents = 'none';
          svgEl.style.overflow = 'visible';
          svgEl.style.filter = 'drop-shadow(0px 0px 6px rgba(240, 85, 35, 0.65))';

          let d1 = '', d2 = '';
          if (highlight.shape === 'circle') {
            const rx = width / 2;
            const ry = height / 2;
            d1 = `M ${rx} 0 A ${rx} ${ry} 0 0 0 ${rx} ${height}`;
            d2 = `M ${rx} 0 A ${rx} ${ry} 0 0 1 ${rx} ${height}`;
          } else {
            const halfW = width / 2;
            d1 = `M ${halfW} 0 L 0 0 L 0 ${height} L ${halfW} ${height}`;
            d2 = `M ${halfW} 0 L ${width} 0 L ${width} ${height} L ${halfW} ${height}`;
          }

          const path1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          path1.setAttribute('d', d1);
          path1.setAttribute('fill', 'none');
          path1.setAttribute('stroke', '#f05523');
          path1.setAttribute('stroke-width', '4.5');
          path1.setAttribute('stroke-linecap', 'round');
          path1.setAttribute('stroke-linejoin', 'round');

          const path2 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          path2.setAttribute('d', d2);
          path2.setAttribute('fill', 'none');
          path2.setAttribute('stroke', '#f05523');
          path2.setAttribute('stroke-width', '4.5');
          path2.setAttribute('stroke-linecap', 'round');
          path2.setAttribute('stroke-linejoin', 'round');

          svgEl.appendChild(path1);
          svgEl.appendChild(path2);
          overlayContainer.appendChild(svgEl);

          const len1 = (typeof path1.getTotalLength === 'function' && path1.getTotalLength() > 0) 
            ? path1.getTotalLength() 
            : (width + height);
          const len2 = (typeof path2.getTotalLength === 'function' && path2.getTotalLength() > 0) 
            ? path2.getTotalLength() 
            : (width + height);

          path1.style.strokeDasharray = len1;
          path1.style.strokeDashoffset = len1;
          path2.style.strokeDasharray = len2;
          path2.style.strokeDashoffset = len2;

          const lastBoxStart = startOffset + (numBoxes - 1) * safeStagger;
          const drawDuration = Math.max(0.4, Math.min(1.5, maxEndTime - lastBoxStart));
          const startTime = startOffset + idx * safeStagger;

          // Fade in the SVG container slightly as drawing begins to avoid sharp edge pop-ins
          tl.fromTo(svgEl, { opacity: 0 }, { opacity: 1, duration: 0.15, ease: 'power1.out' }, startTime);

          // Animate the two halves of the box/circle drawing outwards and meeting at the bottom
          tl.fromTo(path1, { strokeDashoffset: len1 }, { strokeDashoffset: 0, duration: drawDuration, ease: 'power1.out' }, startTime);
          tl.fromTo(path2, { strokeDashoffset: len2 }, { strokeDashoffset: 0, duration: drawDuration, ease: 'power1.out' }, startTime);
        });
      };

      if (target.complete && target.naturalWidth > 0) {
        drawBBoxes();
      } else {
        target.onload = drawBBoxes;
      }
    } 
    else if (animType === 'icon_overlay' && cue.iconOverlays && cue.iconOverlays.length > 0) {
      const startDimDelay = 0.8;
      gsap.set(target, { filter: 'brightness(1)' });
      tl.to(target, { filter: 'brightness(0.35)', duration: 0.6, ease: 'power1.out' }, startDimDelay);

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

        tl.fromTo(cardEl, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.5, ease: 'back.out(1.2)' }, startDimDelay + 0.2 + idx * 0.15);
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
  };
})();
