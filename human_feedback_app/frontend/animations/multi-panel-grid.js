// multi-panel-grid.js
// GSAP motion for multi_panel_grid (3-up row or 4-up 2x2). Equal peer panels.

window.HFMultiPanelGrid = (function () {
  let sceneKey = '';
  let lastPartIdx = -1;
  let lastTreatment = '';
  let entranceDone = {};
  let labelEntranceDone = {};

  const TEMPLATE = 'multi_panel_grid';
  const SEQUENTIAL = ['reveal', 'slide_in'];

  function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function normalizeTreatment(treatment) {
    if (treatment === 'dim_desaturate') return 'scale_emphasis';
    if (treatment === 'fade_in') return 'reveal';
    return treatment || 'scale_emphasis';
  }

  function isSequential(treatment) {
    return SEQUENTIAL.indexOf(treatment) !== -1;
  }

  function sceneKeyForCue(cue) {
    return cue.slideIdx + ':' + cue.sceneId;
  }

  function getCanvas(cue) {
    return document.querySelector('[data-player-scene="' + sceneKeyForCue(cue) + '"]');
  }

  function getSlots(cue) {
    const canvas = getCanvas(cue);
    if (!canvas) return [];
    return Array.from(canvas.querySelectorAll('.hf-player-scene-slot'));
  }

  function motionTarget(slot) {
    return slot.querySelector('img.hf-split-motion-target')
      || slot.querySelector('.hf-split-motion-target')
      || slot.querySelector('.hf-player-scene-slot-inner')
      || slot;
  }

  function entranceTarget(slot) {
    return slot.querySelector('.hf-split-fill-wrapper')
      || motionTarget(slot);
  }

  function resetSlot(slot) {
    const target = motionTarget(slot);
    const enter = entranceTarget(slot);
    const labelBadge = slot.querySelector('.hf-player-slot-label-badge');
    gsap.killTweensOf([slot, target, enter]);
    if (labelBadge) gsap.killTweensOf(labelBadge);
    slot.classList.remove('is-awaiting-visual');
    gsap.set(slot, { clearProps: 'filter,boxShadow,transform,clipPath,opacity,visibility' });
    gsap.set([target, enter], { clearProps: 'transform,filter,clipPath,opacity,visibility' });
    if (labelBadge) gsap.set(labelBadge, { clearProps: 'transform,opacity,visibility,x,y,autoAlpha' });
  }

  function resetOwnedSlots() {
    if (!sceneKey) return;
    const canvas = document.querySelector('[data-player-scene="' + sceneKey + '"]');
    if (!canvas) return;
    canvas.querySelectorAll('.hf-player-scene-slot').forEach(resetSlot);
  }

  function kill() {
    resetOwnedSlots();
    sceneKey = '';
    lastPartIdx = -1;
    lastTreatment = '';
    entranceDone = {};
    labelEntranceDone = {};
  }

  // 3-up: left / top / right. 4-up: each cell from its corner.
  function slideFrom(idx, count) {
    if (count >= 4) {
      const col = idx % 2;
      const row = Math.floor(idx / 2);
      return {
        xPercent: col === 0 ? -105 : 105,
        yPercent: row === 0 ? -105 : 105,
      };
    }
    if (idx === 0) return { xPercent: -105, yPercent: 0 };
    if (idx === 2) return { xPercent: 105, yPercent: 0 };
    return { xPercent: 0, yPercent: -105 };
  }

  function armSlot(treatment, slot, idx, count) {
    if (!slot) return;
    const enter = entranceTarget(slot);
    if (treatment === 'slide_in') {
      const from = slideFrom(idx, count);
      gsap.set(enter, { autoAlpha: 0, xPercent: from.xPercent, yPercent: from.yPercent });
      return;
    }
    gsap.set(enter, { autoAlpha: 0, xPercent: 0, yPercent: 0 });
  }

  function showSettled(slot) {
    gsap.set(entranceTarget(slot), { autoAlpha: 1, xPercent: 0, yPercent: 0 });
  }

  function runReveal(slot) {
    gsap.fromTo(
      entranceTarget(slot),
      { autoAlpha: 0 },
      { autoAlpha: 1, duration: 0.55, ease: 'power2.out', overwrite: 'auto' }
    );
  }

  function runSlide(slot, idx, count) {
    const from = slideFrom(idx, count);
    gsap.fromTo(
      entranceTarget(slot),
      { autoAlpha: 0, xPercent: from.xPercent, yPercent: from.yPercent },
      {
        autoAlpha: 1,
        xPercent: 0,
        yPercent: 0,
        duration: 0.85,
        ease: 'power3.out',
        overwrite: 'auto',
      }
    );
  }

  function playSequential(treatment, slots, partIdx) {
    const count = slots.length;
    slots.forEach(function (slot, idx) {
      if (idx > partIdx) armSlot(treatment, slot, idx, count);
    });
    for (let i = 0; i <= partIdx; i++) {
      if (entranceDone[i] || !slots[i]) continue;
      if (i < partIdx) {
        showSettled(slots[i]);
        entranceDone[i] = true;
        continue;
      }
      const enter = entranceTarget(slots[i]);
      if (gsap.getProperty(enter, 'autoAlpha') >= 0.99) {
        showSettled(slots[i]);
        entranceDone[i] = true;
        continue;
      }
      if (treatment === 'slide_in') runSlide(slots[i], i, count);
      else runReveal(slots[i]);
      entranceDone[i] = true;
    }
  }

  function applyFocus(treatment, slots, activeIdx, instant) {
    const tween = instant ? gsap.set : gsap.to;
    const dur = instant ? {} : { duration: 0.55, ease: 'power2.inOut' };

    slots.forEach(function (slot, idx) {
      const target = motionTarget(slot);
      const isActive = idx === activeIdx;

      switch (treatment) {
        case 'off':
          resetSlot(slot);
          break;

        case 'label_slide_up': {
          gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
          gsap.set(entranceTarget(slot), { autoAlpha: 1, xPercent: 0, yPercent: 0 });
          const labelBadge = slot.querySelector('.hf-player-slot-label-badge');
          if (labelBadge) {
            if (idx <= activeIdx) {
              if (instant) {
                gsap.set(labelBadge, { autoAlpha: 1, y: 0 });
              } else if (idx === activeIdx && !labelEntranceDone[idx]) {
                gsap.fromTo(labelBadge, 
                  { autoAlpha: 0, y: 60 },
                  { autoAlpha: 1, y: 0, duration: 0.85, ease: 'back.out(1.25)', overwrite: 'auto' }
                );
                labelEntranceDone[idx] = true;
              } else {
                gsap.set(labelBadge, { autoAlpha: 1, y: 0 });
              }
            } else {
              gsap.set(labelBadge, { autoAlpha: 0 });
            }
          }
          break;
        }

        case 'label_corner_slide': {
          gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
          gsap.set(entranceTarget(slot), { autoAlpha: 1, xPercent: 0, yPercent: 0 });
          const labelBadge = slot.querySelector('.hf-player-slot-label-badge');
          if (labelBadge) {
            if (idx <= activeIdx) {
              if (instant) {
                gsap.set(labelBadge, { autoAlpha: 1, x: 0, y: 0 });
              } else if (idx === activeIdx && !labelEntranceDone[idx]) {
                let fromX = 0, fromY = 0;
                if (idx === 0) { fromX = -80; fromY = -80; }
                else if (idx === 1) { fromX = 80; fromY = -80; }
                else if (idx === 2) { fromX = -80; fromY = 80; }
                else if (idx === 3) { fromX = 80; fromY = 80; }
                gsap.fromTo(labelBadge, 
                  { autoAlpha: 0, x: fromX, y: fromY },
                  { autoAlpha: 1, x: 0, y: 0, duration: 0.85, ease: 'back.out(1.25)', overwrite: 'auto' }
                );
                labelEntranceDone[idx] = true;
              } else {
                gsap.set(labelBadge, { autoAlpha: 1, x: 0, y: 0 });
              }
            } else {
              gsap.set(labelBadge, { autoAlpha: 0 });
            }
          }
          break;
        }

        case 'label_fade': {
          gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
          gsap.set(entranceTarget(slot), { autoAlpha: 1, xPercent: 0, yPercent: 0 });
          const labelBadge = slot.querySelector('.hf-player-slot-label-badge');
          if (labelBadge) {
            if (idx <= activeIdx) {
              if (instant) {
                gsap.set(labelBadge, { autoAlpha: 1 });
              } else if (idx === activeIdx && !labelEntranceDone[idx]) {
                gsap.fromTo(labelBadge, 
                  { autoAlpha: 0 },
                  { autoAlpha: 1, duration: 0.75, ease: 'power2.out', overwrite: 'auto' }
                );
                labelEntranceDone[idx] = true;
              } else {
                gsap.set(labelBadge, { autoAlpha: 1 });
              }
            } else {
              gsap.set(labelBadge, { autoAlpha: 0 });
            }
          }
          break;
        }

        case 'scale_emphasis':
          tween(target, Object.assign({
            scale: isActive ? 1.06 : 0.92,
            transformOrigin: '50% 50%',
            overwrite: 'auto',
          }, dur));
          gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
          break;

        case 'reveal':
        case 'slide_in':
          gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
          break;

        default:
          break;
      }
    });
  }

  function poseIdle(treatment, slots, partIdx) {
    if (treatment === 'label_slide_up' || treatment === 'label_corner_slide' || treatment === 'label_fade') {
      slots.forEach(function (slot, idx) {
        gsap.set(entranceTarget(slot), { autoAlpha: 1, xPercent: 0, yPercent: 0 });
        gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
        const labelBadge = slot.querySelector('.hf-player-slot-label-badge');
        if (labelBadge) {
          // Hide active/upcoming labels while idle; play animates them once.
          if (idx < partIdx) {
            gsap.set(labelBadge, { autoAlpha: 1, x: 0, y: 0 });
          } else {
            gsap.set(labelBadge, { autoAlpha: 0, x: 0, y: 0 });
          }
        }
      });
      return;
    }

    slots.forEach(function (slot, idx) {
      const target = motionTarget(slot);
      gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
      gsap.set(target, { scale: 1, transformOrigin: '50% 50%' });
      if (isSequential(treatment)) {
        if (idx <= partIdx) showSettled(slot);
        else armSlot(treatment, slot, idx, slots.length);
      } else {
        gsap.set(entranceTarget(slot), { autoAlpha: 1, xPercent: 0, yPercent: 0 });
      }
    });
  }

  function pauseSlotTweens(slots) {
    (slots || []).forEach(function (slot) {
      gsap.getTweensOf(slot).forEach(function (tw) { tw.pause(); });
      gsap.getTweensOf(motionTarget(slot)).forEach(function (tw) { tw.pause(); });
      gsap.getTweensOf(entranceTarget(slot)).forEach(function (tw) { tw.pause(); });
    });
  }

  function sync(opts) {
    const cue = opts.cue;
    const treatment = normalizeTreatment(opts.treatment);
    const playing = !!opts.playing;
    const forceRestart = !!opts.forceRestart;

    if (!cue) {
      kill();
      return;
    }

    const normTemplate = (cue.sceneTemplate || '').replace(/[\s-]+/g, '_').toLowerCase();
    const isGrid = normTemplate === TEMPLATE
      || normTemplate === 'multi_panel'
      || normTemplate === 'grid';

    if (!isGrid) {
      kill();
      return;
    }

    if (prefersReducedMotion() || treatment === 'off' || treatment === 'none') {
      kill();
      return;
    }

    const slots = getSlots(cue);
    if (slots.length < 3) return;

    const key = sceneKeyForCue(cue);
    const partIdx = cue.partIdx != null ? cue.partIdx : 0;
    const treatmentChanged = lastTreatment !== treatment;

    function resetMotionState() {
      slots.forEach(resetSlot);
      entranceDone = {};
      labelEntranceDone = {};
    }

    if (!playing) {
      if (key !== sceneKey || treatmentChanged || forceRestart) {
        if (key !== sceneKey) kill();
        else resetMotionState();
        sceneKey = key;
        lastTreatment = treatment;
        lastPartIdx = partIdx;
        poseIdle(treatment, slots, partIdx);
      } else {
        pauseSlotTweens(slots);
      }
      return;
    }

    function runCueMotion() {
      if (isSequential(treatment)) playSequential(treatment, slots, partIdx);
      applyFocus(treatment, slots, partIdx);
    }

    if (key !== sceneKey) {
      kill();
      sceneKey = key;
      lastTreatment = treatment;
      lastPartIdx = partIdx;
      if (isSequential(treatment)) {
        slots.forEach(function (slot, idx) {
          if (idx < partIdx) showSettled(slot);
          else armSlot(treatment, slot, idx, slots.length);
        });
      }
      runCueMotion();
      return;
    }

    if (treatmentChanged || forceRestart) {
      resetMotionState();
      lastTreatment = treatment;
      lastPartIdx = partIdx;
      runCueMotion();
      return;
    }

    lastTreatment = treatment;

    // Same scene cue advance: never kill()/clearProps.
    if (partIdx > lastPartIdx && isSequential(treatment)) {
      playSequential(treatment, slots, partIdx);
    }

    if (partIdx !== lastPartIdx) lastPartIdx = partIdx;

    applyFocus(treatment, slots, partIdx);
  }

  return {
    sync: sync,
    kill: kill,
  };
})();
