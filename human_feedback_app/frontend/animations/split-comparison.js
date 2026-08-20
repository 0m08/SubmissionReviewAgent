// split-comparison.js
// GSAP motion treatments for two_item_split_comparison layouts.
window.HFSplitComparison = (function () {
  let sceneKey = '';
  let lastPartIdx = -1;
  let lastTreatment = '';
  let entranceDone = {};
  let ambientTween = null;
  let splitTl = null;

  const TEMPLATE = 'two_item_split_comparison';
  const ENTRANCE_ON_CUE = ['slide_in', 'wipe_reveal', 'hold_and_reveal', 'center_then_split', 'slide_up'];

  function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function normalizeTreatment(treatment) {
    if (treatment === 'dim_desaturate') return 'hold_and_reveal';
    return treatment || 'hold_and_reveal';
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

  function isImageSlot(slot) {
    return !!slot.querySelector('img.hf-split-fg, img.hf-split-motion-target, img:not(.hf-hero-bg-blur)');
  }

  function resetSlot(slot) {
    const target = motionTarget(slot);
    const enter = entranceTarget(slot);
    gsap.killTweensOf([slot, target, enter]);
    slot.classList.remove('is-awaiting-visual');
    gsap.set(slot, { clearProps: 'filter,boxShadow,transform,clipPath,opacity,visibility' });
    gsap.set([target, enter], { clearProps: 'transform,filter,clipPath,opacity,visibility' });
  }

  function wipeClip(fromLeft, closed) {
    if (closed) return fromLeft ? 'inset(0% 100% 0% 0%)' : 'inset(0% 0% 0% 100%)';
    return 'inset(0% 0% 0% 0%)';
  }

  function hideUpcoming(slot) {
    gsap.set(entranceTarget(slot), { autoAlpha: 0 });
  }

  function showSpoken(slot) {
    gsap.set(entranceTarget(slot), { autoAlpha: 1 });
    gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
  }

  function runEntrance(treatment, slots, idx) {
    const slot = slots[idx];
    if (!slot) return;
    const target = entranceTarget(slot);
    const fromLeft = idx === 0;

    switch (treatment) {
      case 'slide_up':
        gsap.fromTo(
          target,
          {
            yPercent: 105,
            autoAlpha: 0,
            transformOrigin: '50% 50%',
          },
          {
            yPercent: 0,
            autoAlpha: 1,
            duration: 0.85,
            ease: 'power3.out',
            overwrite: 'auto',
          }
        );
        break;
      case 'slide_in':
        gsap.fromTo(
          target,
          {
            xPercent: fromLeft ? -105 : 105,
            autoAlpha: 0,
            transformOrigin: '50% 50%',
          },
          {
            xPercent: 0,
            autoAlpha: 1,
            duration: 0.85,
            ease: 'power3.out',
            overwrite: 'auto',
          }
        );
        break;
      case 'wipe_reveal':
        gsap.fromTo(
          slot,
          { clipPath: wipeClip(fromLeft, true) },
          {
            clipPath: wipeClip(fromLeft, false),
            duration: 0.9,
            ease: 'power2.inOut',
            overwrite: 'auto',
          }
        );
        break;
      case 'hold_and_reveal':
        if (idx === 0) {
          gsap.set(target, { autoAlpha: 1 });
        } else {
          gsap.fromTo(
            target,
            { autoAlpha: 0 },
            { autoAlpha: 1, duration: 0.55, ease: 'power2.out', overwrite: 'auto' }
          );
        }
        gsap.set(slots[0], { filter: 'brightness(1) saturate(1)' });
        break;
      case 'center_then_split':
        if (splitTl) {
          splitTl.kill();
          splitTl = null;
        }
        if (idx === 0) {
          gsap.set(slots[0], { xPercent: 50, zIndex: 3, autoAlpha: 1 });
          gsap.set(target, { autoAlpha: 1 });
          if (slots[1]) {
            slots[1].classList.add('is-awaiting-visual');
            gsap.set(slots[1], { zIndex: 1, autoAlpha: 0 });
            hideUpcoming(slots[1]);
          }
        } else {
          const right = slots[1];
          if (right) {
            right.classList.add('is-awaiting-visual');
            gsap.set(right, { zIndex: 1, autoAlpha: 0 });
            gsap.set(entranceTarget(right), { autoAlpha: 1 });
          }
          gsap.set(slots[0], { zIndex: 4, autoAlpha: 1 });
          splitTl = gsap.timeline({
            defaults: { overwrite: 'auto' },
            onComplete: function () {
              gsap.set(slots[0], { zIndex: 1 });
              if (right) {
                right.classList.remove('is-awaiting-visual');
                gsap.set(right, { zIndex: 2, autoAlpha: 1 });
              }
            },
          });
          splitTl.fromTo(
            slots[0],
            { xPercent: 50 },
            { xPercent: 0, duration: 0.78, ease: 'power2.inOut' },
            0
          );
          if (right) {
            splitTl.fromTo(
              right,
              { autoAlpha: 0 },
              { autoAlpha: 1, duration: 0.5, ease: 'power2.out' },
              0.08
            );
          }
        }
        break;
      default:
        break;
    }
  }

  function armPendingEntrances(treatment, slots) {
    const pending = slots[1];
    if (!pending) return;
    if (treatment === 'wipe_reveal') {
      gsap.set(pending, { clipPath: wipeClip(false, true) });
    } else if (treatment === 'slide_in') {
      gsap.set(entranceTarget(pending), { xPercent: 105, autoAlpha: 0 });
    } else if (treatment === 'slide_up') {
      gsap.set(entranceTarget(pending), { yPercent: 105, autoAlpha: 0 });
    } else if (treatment === 'hold_and_reveal') {
      hideUpcoming(pending);
    } else if (treatment === 'center_then_split') {
      gsap.set(slots[0], { xPercent: 50, zIndex: 3, autoAlpha: 1 });
      hideUpcoming(pending);
      gsap.set(pending, { autoAlpha: 0, zIndex: 1 });
      pending.classList.add('is-awaiting-visual');
    }
  }

  function resetOwnedSlots() {
    if (!sceneKey) return;
    const canvas = document.querySelector('[data-player-scene="' + sceneKey + '"]');
    if (!canvas) return;
    canvas.querySelectorAll('.hf-player-scene-slot').forEach(resetSlot);
  }

  function kill() {
    if (splitTl) {
      splitTl.kill();
      splitTl = null;
    }
    if (ambientTween) {
      ambientTween.kill();
      ambientTween = null;
    }
    resetOwnedSlots();
    sceneKey = '';
    lastPartIdx = -1;
    lastTreatment = '';
    entranceDone = {};
  }

  function applyFocus(treatment, slots, activeIdx, instant) {
    slots.forEach(function (slot, idx) {
      const target = motionTarget(slot);
      const tween = instant ? gsap.set : gsap.to;
      const isActive = idx === activeIdx;

      switch (treatment) {
        case 'off':
          resetSlot(slot);
          break;

        case 'hold_and_reveal':
          gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
          if (idx < activeIdx) showSpoken(slot);
          else if (idx === activeIdx && activeIdx === 0) showSpoken(slot);
          else if (idx > activeIdx) hideUpcoming(slot);
          break;

        case 'scale_emphasis':
          tween(target, Object.assign({
            scale: isActive ? 1.06 : 0.92,
            transformOrigin: '50% 50%',
            overwrite: 'auto',
          }, instant ? {} : { duration: 0.55, ease: 'power2.inOut' }));
          gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
          break;

        default:
          gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
          break;
      }
    });
  }

  function poseIdle(treatment, slots, partIdx) {
    if (treatment === 'wipe_reveal') {
      gsap.set(slots[0], { clipPath: wipeClip(true, false) });
      if (slots[1]) gsap.set(slots[1], { clipPath: wipeClip(false, true) });
    } else if (treatment === 'slide_in') {
      gsap.set(entranceTarget(slots[0]), { xPercent: 0, autoAlpha: 1 });
      if (slots[1]) gsap.set(entranceTarget(slots[1]), { xPercent: 105, autoAlpha: 0 });
    } else if (treatment === 'slide_up') {
      gsap.set(entranceTarget(slots[0]), { yPercent: 0, autoAlpha: 1 });
      if (slots[1]) gsap.set(entranceTarget(slots[1]), { yPercent: 105, autoAlpha: 0 });
    } else if (treatment === 'center_then_split') {
      gsap.set(slots[0], { xPercent: 50, zIndex: 3, autoAlpha: 1 });
      showSpoken(slots[0]);
      if (slots[1]) {
        slots[1].classList.add('is-awaiting-visual');
        hideUpcoming(slots[1]);
        gsap.set(slots[1], { autoAlpha: 0, zIndex: 1 });
      }
    } else if (treatment === 'hold_and_reveal') {
      showSpoken(slots[0]);
      if (slots[1]) hideUpcoming(slots[1]);
    } else {
      armPendingEntrances(treatment, slots);
    }

    if (treatment === 'scale_emphasis') {
      slots.forEach(function (slot) {
        const target = motionTarget(slot);
        gsap.set(slot, { filter: 'brightness(1) saturate(1)' });
        gsap.set(target, { scale: 1, transformOrigin: '50% 50%' });
        gsap.set(entranceTarget(slot), { autoAlpha: 1 });
      });
      return;
    }
    applyFocus(treatment, slots, partIdx, true);
  }

  function pauseSlotTweens(slots) {
    (slots || []).forEach(function (slot) {
      gsap.getTweensOf(slot).forEach(function (tw) { tw.pause(); });
      gsap.getTweensOf(motionTarget(slot)).forEach(function (tw) { tw.pause(); });
      gsap.getTweensOf(entranceTarget(slot)).forEach(function (tw) { tw.pause(); });
    });
    if (ambientTween) ambientTween.pause();
    if (splitTl) splitTl.pause();
  }

  function syncAmbient(treatment, slots, activeIdx, cue, duration, elapsed, playing) {
    if (ambientTween) {
      ambientTween.kill();
      ambientTween = null;
    }
  }

  function runSceneEntrances(treatment, slots) {
    if (!entranceDone[0]) {
      if (ENTRANCE_ON_CUE.indexOf(treatment) !== -1) {
        runEntrance(treatment, slots, 0);
        armPendingEntrances(treatment, slots);
      }
      entranceDone[0] = true;
    }
  }

  function sync(opts) {
    const cue = opts.cue;
    const treatment = normalizeTreatment(opts.treatment);
    const duration = opts.duration || 0;
    const elapsed = opts.elapsed || 0;
    const playing = !!opts.playing;

    if (!cue) {
      kill();
      return;
    }

    const normTemplate = (cue.sceneTemplate || '').replace(/[\s-]+/g, '_').toLowerCase();
    const isSplit = normTemplate === TEMPLATE
      || normTemplate === 'two_item_split'
      || normTemplate === 'split_comparison';

    if (!isSplit) {
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
    const treatmentChanged = lastTreatment && lastTreatment !== treatment;

    if (!playing) {
      if (key !== sceneKey || treatmentChanged) {
        if (key !== sceneKey) kill();
        else {
          slots.forEach(resetSlot);
          if (ambientTween) { ambientTween.kill(); ambientTween = null; }
          entranceDone = {};
        }
        sceneKey = key;
        lastTreatment = treatment;
        lastPartIdx = partIdx;
        poseIdle(treatment, slots, partIdx);
      } else {
        pauseSlotTweens(slots);
      }
      return;
    }

    if (key !== sceneKey) {
      kill();
      sceneKey = key;
      lastTreatment = treatment;
      runSceneEntrances(treatment, slots);
      lastPartIdx = partIdx;
      applyFocus(treatment, slots, partIdx);
      syncAmbient(treatment, slots, partIdx, cue, duration, elapsed, playing);
      return;
    }

    if (treatmentChanged) {
      slots.forEach(resetSlot);
      if (ambientTween) {
        ambientTween.kill();
        ambientTween = null;
      }
      entranceDone = {};
      lastTreatment = treatment;
      lastPartIdx = partIdx;
      runSceneEntrances(treatment, slots);
      if (partIdx > 0 && ENTRANCE_ON_CUE.indexOf(treatment) !== -1) {
        runEntrance(treatment, slots, partIdx);
        entranceDone[partIdx] = true;
      }
      applyFocus(treatment, slots, partIdx);
      syncAmbient(treatment, slots, partIdx, cue, duration, elapsed, playing);
      return;
    }

    lastTreatment = treatment;

    if (!entranceDone[0]) {
      runSceneEntrances(treatment, slots);
    }

    if (partIdx > lastPartIdx) {
      if (!entranceDone[partIdx] && ENTRANCE_ON_CUE.indexOf(treatment) !== -1) {
        runEntrance(treatment, slots, partIdx);
        entranceDone[partIdx] = true;
      }
    }

    if (partIdx !== lastPartIdx) {
      lastPartIdx = partIdx;
    }

    if (splitTl && splitTl.paused()) splitTl.play();

    applyFocus(treatment, slots, partIdx);
    syncAmbient(treatment, slots, partIdx, cue, duration, elapsed, playing);
  }

  return {
    sync: sync,
    kill: kill,
  };
})();
