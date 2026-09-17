// main-inset.js
// GSAP motion treatments for main_plus_supporting_inset layouts.
// Slot 0 = main_visual (dominant). Slot 1 = inset_visual (supporting).
window.HFMainInset = (function () {
  let sceneKey = '';
  let lastPartIdx = -1;
  let lastTreatment = '';
  let entranceDone = {};
  let labelEntranceDone = {};
  let entranceTl = null;

  const TEMPLATE = 'main_plus_supporting_inset';
  const ENTRANCES = ['keep_main_reveal', 'inset_slide'];

  function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function normalizeTreatment(treatment) {
    if (treatment === 'dim_desaturate' || treatment === 'gentle_zoom') return 'keep_main_reveal';
    return treatment || 'keep_main_reveal';
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

  function isEntrance(treatment) {
    return ENTRANCES.indexOf(treatment) !== -1;
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

  function killEntranceTl() {
    if (entranceTl) {
      entranceTl.kill();
      entranceTl = null;
    }
  }

  function resetOwnedSlots() {
    if (!sceneKey) return;
    const canvas = document.querySelector('[data-player-scene="' + sceneKey + '"]');
    if (!canvas) return;
    canvas.querySelectorAll('.hf-player-scene-slot').forEach(resetSlot);
  }

  function kill() {
    killEntranceTl();
    resetOwnedSlots();
    sceneKey = '';
    lastPartIdx = -1;
    lastTreatment = '';
    entranceDone = {};
    labelEntranceDone = {};
  }

  function hideUpcoming(slot) {
    gsap.set(entranceTarget(slot), { autoAlpha: 0 });
  }

  function showSpoken(slot) {
    gsap.set(entranceTarget(slot), { autoAlpha: 1 });
    gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
  }

  function armInset(treatment, inset) {
    if (!inset) return;
    const enter = entranceTarget(inset);
    if (treatment === 'inset_slide') {
      inset.classList.add('is-awaiting-visual');
      gsap.set(enter, { xPercent: 105, autoAlpha: 0 });
      return;
    }
    hideUpcoming(inset);
  }

  function runInsetEntrance(treatment, slots) {
    const inset = slots[1];
    if (!inset) return;
    const enter = entranceTarget(inset);
    inset.classList.remove('is-awaiting-visual');

    if (treatment === 'keep_main_reveal') {
      gsap.fromTo(
        enter,
        { autoAlpha: 0 },
        { autoAlpha: 1, duration: 0.55, ease: 'power2.out', overwrite: 'auto' }
      );
      if (slots[0]) gsap.set(slots[0], { filter: 'brightness(1) saturate(1)' });
      return;
    }

    killEntranceTl();
    entranceTl = gsap.timeline({ defaults: { overwrite: 'auto' } });
    if (treatment === 'inset_slide') {
      entranceTl.fromTo(
        enter,
        { xPercent: 105, autoAlpha: 0 },
        { xPercent: 0, autoAlpha: 1, duration: 0.85, ease: 'power3.out' },
        0
      );
    }
  }

  function applyFocus(treatment, slots, activeIdx, instant) {
    switch (treatment) {
      case 'off':
        slots.forEach(resetSlot);
        break;

      case 'label_slide':
        slots.forEach(function (slot, idx) {
          gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
          gsap.set(entranceTarget(slot), { autoAlpha: 1, xPercent: 0, yPercent: 0 });
          const labelBadge = slot.querySelector('.hf-player-slot-label-badge');
          if (labelBadge) {
            if (idx <= activeIdx) {
              if (instant) {
                gsap.set(labelBadge, { autoAlpha: 1, x: 0 });
              } else if (idx === activeIdx && !labelEntranceDone[idx]) {
                const fromX = (idx === 0) ? -150 : 150;
                gsap.fromTo(labelBadge, 
                  { autoAlpha: 0, x: fromX },
                  { autoAlpha: 1, x: 0, duration: 0.85, ease: 'back.out(1.25)', overwrite: 'auto' }
                );
                labelEntranceDone[idx] = true;
              } else {
                gsap.set(labelBadge, { autoAlpha: 1, x: 0 });
              }
            } else {
              gsap.set(labelBadge, { autoAlpha: 0 });
            }
          }
        });
        break;

      case 'label_fade':
        slots.forEach(function (slot, idx) {
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
        });
        break;

      case 'keep_main_reveal':
        slots.forEach(function (slot, idx) {
          gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
          if (idx < activeIdx) showSpoken(slot);
          else if (idx === activeIdx && activeIdx === 0) showSpoken(slot);
          else if (idx > activeIdx) hideUpcoming(slot);
        });
        break;

      case 'inset_slide': {
        const main = slots[0];
        const tween = instant ? gsap.set : gsap.to;
        const dur = instant ? {} : { duration: 0.45, ease: 'power2.inOut' };
        if (main) {
          tween(main, Object.assign({
            filter: 'brightness(1) saturate(1)',
            overwrite: 'auto',
          }, dur));
        }
        break;
      }

      default:
        break;
    }
  }

  function poseIdle(treatment, slots, partIdx) {
    if (treatment === 'label_slide' || treatment === 'label_fade') {
      slots.forEach(function (slot, idx) {
        gsap.set(entranceTarget(slot), { autoAlpha: 1, xPercent: 0, yPercent: 0 });
        gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
        const labelBadge = slot.querySelector('.hf-player-slot-label-badge');
        if (labelBadge) {
          // Hide active/upcoming labels while idle; play animates them once.
          if (idx < partIdx) {
            gsap.set(labelBadge, { autoAlpha: 1, x: 0 });
          } else {
            gsap.set(labelBadge, { autoAlpha: 0, x: 0 });
          }
        }
      });
      return;
    }

    const main = slots[0];
    const inset = slots[1];
    if (main) {
      main.classList.remove('is-awaiting-visual');
      gsap.set(entranceTarget(main), { xPercent: 0, autoAlpha: 1, scale: 1 });
      gsap.set(motionTarget(main), { scale: 1, transformOrigin: '50% 50%' });
      gsap.set(main, { clipPath: 'inset(0% 0% 0% 0%)', filter: 'brightness(1) saturate(1)' });
    }
    if (isEntrance(treatment) && inset) {
      if (partIdx >= 1) {
        inset.classList.remove('is-awaiting-visual');
        gsap.set(entranceTarget(inset), { xPercent: 0, autoAlpha: 1, scale: 1 });
        gsap.set(inset, { clipPath: 'inset(0% 0% 0% 0%)', filter: 'brightness(1) saturate(1)' });
      } else {
        armInset(treatment, inset);
      }
      return;
    }
    if (inset) {
      inset.classList.remove('is-awaiting-visual');
      gsap.set(entranceTarget(inset), { xPercent: 0, autoAlpha: 1, scale: 1 });
      gsap.set(inset, { clipPath: 'inset(0% 0% 0% 0%)', filter: 'brightness(1) saturate(1)' });
    }
  }

  function pauseSlotTweens(slots) {
    (slots || []).forEach(function (slot) {
      gsap.getTweensOf(slot).forEach(function (tw) { tw.pause(); });
      gsap.getTweensOf(motionTarget(slot)).forEach(function (tw) { tw.pause(); });
      gsap.getTweensOf(entranceTarget(slot)).forEach(function (tw) { tw.pause(); });
    });
    if (entranceTl) entranceTl.pause();
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
    const isInset = normTemplate === TEMPLATE
      || normTemplate === 'main_plus_inset'
      || normTemplate === 'main_plus_supporting'
      || normTemplate === 'main_visual_plus_inset';

    if (!isInset) {
      kill();
      return;
    }

    if (prefersReducedMotion() || treatment === 'off' || treatment === 'none') {
      kill();
      return;
    }

    const slots = getSlots(cue);
    if (slots.length < 2) return;

    const key = sceneKeyForCue(cue);
    const partIdx = cue.partIdx != null ? cue.partIdx : 0;
    const treatmentChanged = lastTreatment !== treatment;

    function resetMotionState() {
      slots.forEach(resetSlot);
      killEntranceTl();
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

    function playEntrances() {
      if (!entranceDone[0]) {
        if (isEntrance(treatment) && partIdx < 1) armInset(treatment, slots[1]);
        entranceDone[0] = true;
      }
      if (partIdx >= 1 && !entranceDone[1] && isEntrance(treatment)) {
        runInsetEntrance(treatment, slots);
        entranceDone[1] = true;
      }
    }

    if (key !== sceneKey) {
      kill();
      sceneKey = key;
      lastTreatment = treatment;
      lastPartIdx = partIdx;
      playEntrances();
      applyFocus(treatment, slots, partIdx);
      return;
    }

    if (treatmentChanged || forceRestart) {
      resetMotionState();
      lastTreatment = treatment;
      lastPartIdx = partIdx;
      playEntrances();
      applyFocus(treatment, slots, partIdx);
      return;
    }

    lastTreatment = treatment;

    if (entranceTl && entranceTl.paused()) entranceTl.resume();

    if (!entranceDone[0]) playEntrances();

    // Same scene cue advance: never kill()/clearProps — that blanks panels before dim.
    if (partIdx > lastPartIdx) playEntrances();

    if (partIdx !== lastPartIdx) lastPartIdx = partIdx;

    applyFocus(treatment, slots, partIdx);
  }

  return {
    sync: sync,
    kill: kill,
  };
})();
