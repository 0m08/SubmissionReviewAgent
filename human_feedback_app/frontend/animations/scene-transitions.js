// scene-transitions.js
// GSAP scene-to-scene transitions for the Player.
// Runs only across slide/scene boundaries — not within multi-panel cue handoffs.
window.HFSceneTransition = (function () {
  let activeTl = null;
  let veilEl = null;

  function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function currentShell() {
    return document.querySelector('.hf-player-fade-shell');
  }

  function ensureVeil(frame) {
    if (!frame) return null;
    let veil = frame.querySelector('.hf-scene-transition-veil');
    if (!veil) {
      veil = document.createElement('div');
      veil.className = 'hf-scene-transition-veil';
      frame.appendChild(veil);
    }
    veilEl = veil;
    return veil;
  }

  function resetShell(shell) {
    if (!shell) return;
    gsap.killTweensOf(shell);
    gsap.set(shell, {
      clearProps: 'transform,opacity,visibility,clipPath,filter,x,y,xPercent,yPercent',
    });
    shell.classList.remove('is-faded', 'hf-gsap-transitioning');
    shell.style.transition = '';
  }

  function holdShellHidden(shell) {
    if (!shell) return;
    shell.classList.add('is-faded', 'hf-gsap-transitioning');
    shell.style.transition = 'none';
    if (window.gsap) gsap.set(shell, { autoAlpha: 0 });
  }

  function revealShell(shell) {
    if (!shell) return;
    shell.classList.remove('is-faded');
    shell.classList.add('hf-gsap-transitioning');
    shell.style.transition = 'none';
    if (window.gsap) {
      gsap.set(shell, { autoAlpha: 1, xPercent: 0, yPercent: 0, clipPath: 'inset(0% 0% 0% 0%)' });
    }
  }

  function kill() {
    if (activeTl) {
      activeTl.kill();
      activeTl = null;
    }
    resetShell(currentShell());
    if (veilEl) {
      gsap.killTweensOf(veilEl);
      gsap.set(veilEl, { autoAlpha: 0 });
    }
  }

  function swapAndResume(tl, onSwap, prepareIn) {
    tl.pause();
    onSwap(function () {
      const liveShell = currentShell();
      if (typeof prepareIn === 'function') prepareIn(liveShell);
      tl.resume();
    });
  }

  function run(opts) {
    const treatment = opts.treatment || 'crossfade';
    let shell = opts.shell || currentShell();
    const onSwap = typeof opts.onSwap === 'function' ? opts.onSwap : function (done) { done(); };

    return new Promise(function (resolve) {
      kill();
      shell = currentShell() || shell;

      if (!shell || treatment === 'off' || treatment === 'none' || prefersReducedMotion()) {
        onSwap(function () {
          resetShell(currentShell());
          resolve();
        });
        return;
      }

      shell.classList.add('hf-gsap-transitioning');
      shell.style.transition = 'none';
      shell.classList.remove('is-faded');
      gsap.set(shell, { autoAlpha: 1, xPercent: 0, yPercent: 0, clipPath: 'inset(0% 0% 0% 0%)' });

      const frame = shell.closest('.hf-player-frame');
      const tl = gsap.timeline({
        defaults: { ease: 'power2.inOut', overwrite: 'auto' },
        onComplete: function () {
          resetShell(currentShell());
          if (veilEl) gsap.set(veilEl, { autoAlpha: 0 });
          activeTl = null;
          resolve();
        },
      });
      activeTl = tl;

      // Prefer a frame-level veil for out/swap/in. The veil is outside the React
      // remounted fade-shell, so forceUpdate cannot flash the next visual.
      function runVeilCrossfade() {
        const veil = ensureVeil(frame);
        if (!veil) {
          // Fallback: hide shell via CSS class that survives remount.
          tl.to(shell, { autoAlpha: 0, duration: 0.4, ease: 'power2.in' }, 0);
          tl.add(function () {
            holdShellHidden(currentShell() || shell);
            swapAndResume(tl, onSwap, function (live) {
              holdShellHidden(live || currentShell());
            });
          });
          tl.add(function () {
            const live = currentShell();
            revealShell(live);
            if (live) gsap.fromTo(live, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.45, ease: 'power2.out' });
          });
          return;
        }
        gsap.set(veil, { autoAlpha: 0, visibility: 'visible' });
        tl.to(veil, { autoAlpha: 1, duration: 0.38, ease: 'power2.in' }, 0);
        tl.add(function () {
          // Keep shell visually suppressed under the veil during remount.
          holdShellHidden(currentShell() || shell);
          swapAndResume(tl, onSwap, function (live) {
            revealShell(live || currentShell());
          });
        });
        tl.to(veil, { autoAlpha: 0, duration: 0.42, ease: 'power2.out' });
      }

      switch (treatment) {
        case 'fade_black':
        case 'crossfade':
        default: {
          runVeilCrossfade();
          break;
        }

        case 'push': {
          const veil = ensureVeil(frame);
          if (veil) {
            runVeilCrossfade();
            break;
          }
          tl.to(shell, {
            xPercent: -100,
            duration: 0.4,
            ease: 'power2.in',
          }, 0);
          tl.add(function () {
            holdShellHidden(currentShell() || shell);
            swapAndResume(tl, onSwap, function (live) {
              const target = live || currentShell();
              revealShell(target);
              if (target) gsap.set(target, { xPercent: 100, autoAlpha: 1 });
              shell = target || shell;
            });
          });
          tl.to(shell, {
            xPercent: 0,
            duration: 0.45,
            ease: 'power2.out',
          });
          break;
        }

        case 'wipe': {
          const veil = ensureVeil(frame);
          if (veil) {
            runVeilCrossfade();
            break;
          }
          tl.fromTo(
            shell,
            { clipPath: 'inset(0% 0% 0% 0%)' },
            { clipPath: 'inset(0% 0% 0% 100%)', duration: 0.4, ease: 'power2.in' },
            0
          );
          tl.add(function () {
            holdShellHidden(currentShell() || shell);
            swapAndResume(tl, onSwap, function (live) {
              const target = live || currentShell();
              revealShell(target);
              if (target) gsap.set(target, { clipPath: 'inset(0% 100% 0% 0%)', autoAlpha: 1 });
              shell = target || shell;
            });
          });
          tl.to(shell, {
            clipPath: 'inset(0% 0% 0% 0%)',
            duration: 0.45,
            ease: 'power2.out',
          });
          break;
        }
      }
    });
  }

  return {
    run: run,
    kill: kill,
    holdShellHidden: holdShellHidden,
    currentShell: currentShell,
  };
})();
