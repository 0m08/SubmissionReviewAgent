// scene-transitions.js
// GSAP scene-to-scene transitions for the Player.
// Runs only across slide/scene boundaries — not within multi-panel cue handoffs.
window.HFSceneTransition = (function () {
  let activeTl = null;
  let veilEl = null;

  function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
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

  function kill() {
    if (activeTl) {
      activeTl.kill();
      activeTl = null;
    }
    const shell = document.querySelector('.hf-player-fade-shell');
    resetShell(shell);
    if (veilEl) {
      gsap.killTweensOf(veilEl);
      gsap.set(veilEl, { autoAlpha: 0 });
    }
  }

  function swapAndResume(tl, onSwap, prepareIn) {
    tl.pause();
    onSwap(function () {
      if (typeof prepareIn === 'function') prepareIn();
      tl.resume();
    });
  }

  function run(opts) {
    const treatment = opts.treatment || 'crossfade';
    const shell = opts.shell;
    const onSwap = typeof opts.onSwap === 'function' ? opts.onSwap : function (done) { done(); };

    return new Promise(function (resolve) {
      kill();

      if (!shell || treatment === 'off' || treatment === 'none' || prefersReducedMotion()) {
        onSwap(function () {
          resetShell(shell);
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
          resetShell(shell);
          if (veilEl) gsap.set(veilEl, { autoAlpha: 0 });
          activeTl = null;
          resolve();
        },
      });
      activeTl = tl;

      switch (treatment) {
        case 'fade_black': {
          const veil = ensureVeil(frame);
          if (!veil) {
            tl.to(shell, { autoAlpha: 0, duration: 0.4, ease: 'power2.in' }, 0);
            tl.add(function () {
              swapAndResume(tl, onSwap, function () {
                gsap.set(shell, { autoAlpha: 0 });
              });
            });
            tl.to(shell, { autoAlpha: 1, duration: 0.45, ease: 'power2.out' });
            break;
          }
          gsap.set(veil, { autoAlpha: 0 });
          tl.to(veil, { autoAlpha: 1, duration: 0.38, ease: 'power2.in' }, 0);
          tl.add(function () {
            swapAndResume(tl, onSwap, function () {
              gsap.set(shell, { autoAlpha: 1, xPercent: 0, clipPath: 'inset(0% 0% 0% 0%)' });
            });
          });
          tl.to(veil, { autoAlpha: 0, duration: 0.42, ease: 'power2.out' });
          break;
        }

        case 'push': {
          tl.to(shell, {
            xPercent: -100,
            duration: 0.4,
            ease: 'power2.in',
          }, 0);
          tl.add(function () {
            swapAndResume(tl, onSwap, function () {
              gsap.set(shell, { xPercent: 100, autoAlpha: 1 });
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
          // Always 4 % values so GSAP can interpolate.
          tl.fromTo(
            shell,
            { clipPath: 'inset(0% 0% 0% 0%)' },
            { clipPath: 'inset(0% 0% 0% 100%)', duration: 0.4, ease: 'power2.in' },
            0
          );
          tl.add(function () {
            swapAndResume(tl, onSwap, function () {
              gsap.set(shell, { clipPath: 'inset(0% 100% 0% 0%)', autoAlpha: 1 });
            });
          });
          tl.to(shell, {
            clipPath: 'inset(0% 0% 0% 0%)',
            duration: 0.45,
            ease: 'power2.out',
          });
          break;
        }

        case 'crossfade':
        default: {
          tl.to(shell, { autoAlpha: 0, duration: 0.4, ease: 'power2.in' }, 0);
          tl.add(function () {
            swapAndResume(tl, onSwap, function () {
              gsap.set(shell, { autoAlpha: 0, xPercent: 0, clipPath: 'inset(0% 0% 0% 0%)' });
            });
          });
          tl.to(shell, { autoAlpha: 1, duration: 0.45, ease: 'power2.out' });
          break;
        }
      }
    });
  }

  return {
    run: run,
    kill: kill,
  };
})();
