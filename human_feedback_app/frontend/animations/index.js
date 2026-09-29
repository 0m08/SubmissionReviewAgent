// index.js
// Exposes the animation controller modules for the Player.
window.HFPlayerMotion = (function () {
  let active = '';

  function normalizeTemplate(template) {
    const t = String(template || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
    const aliases = {
      single_hero: 'single_visual_hero',
      hero: 'single_visual_hero',
      two_item_split: 'two_item_split_comparison',
      split_comparison: 'two_item_split_comparison',
      main_plus_inset: 'main_plus_supporting_inset',
      main_plus_supporting: 'main_plus_supporting_inset',
      main_visual_plus_inset: 'main_plus_supporting_inset',
      multi_panel: 'multi_panel_grid',
      grid: 'multi_panel_grid',
    };
    return aliases[t] || t;
  }

  function isSplitCue(cue) {
    if (!cue) return false;
    return normalizeTemplate(cue.sceneTemplate || '') === 'two_item_split_comparison';
  }

  function isInsetCue(cue) {
    if (!cue) return false;
    return normalizeTemplate(cue.sceneTemplate || '') === 'main_plus_supporting_inset';
  }

  function isGridCue(cue) {
    if (!cue) return false;
    return normalizeTemplate(cue.sceneTemplate || '') === 'multi_panel_grid';
  }

  function isSingleHeroCue(cue) {
    if (!cue) return false;
    const t = normalizeTemplate(cue.sceneTemplate || '');
    return t === 'single_visual_hero' || t === 'topic_transition';
  }

  function killModule(name) {
    if (name === 'hero' && window.HFHeroStills) window.HFHeroStills.kill();
    if (name === 'split' && window.HFSplitComparison) window.HFSplitComparison.kill();
    if (name === 'inset' && window.HFMainInset) window.HFMainInset.kill();
    if (name === 'grid' && window.HFMultiPanelGrid) window.HFMultiPanelGrid.kill();
  }

  // Only tear down the previous layout when the template kind actually changes.
  // Killing siblings on every sync clearProps opacity on live panels → blank then dim.
  function activate(name) {
    if (active && active !== name) killModule(active);
    active = name;
  }

  function sync(opts) {
    const cue = opts.cue;
    if (!cue) {
      kill();
      return;
    }
    if (isSplitCue(cue)) {
      activate('split');
      if (window.HFSplitComparison) {
        window.HFSplitComparison.sync({
          cue: cue,
          duration: opts.duration || 0,
          playing: !!opts.playing,
          treatment: opts.splitTreatment || opts.treatment || 'dim_desaturate',
          elapsed: opts.elapsed || 0,
          forceRestart: !!opts.forceRestart,
        });
      }
      return;
    }

    if (isInsetCue(cue)) {
      activate('inset');
      if (window.HFMainInset) {
        window.HFMainInset.sync({
          cue: cue,
          duration: opts.duration || 0,
          playing: !!opts.playing,
          treatment: opts.insetTreatment || opts.treatment || 'keep_main_reveal',
          elapsed: opts.elapsed || 0,
          forceRestart: !!opts.forceRestart,
        });
      }
      return;
    }

    if (isGridCue(cue)) {
      activate('grid');
      if (window.HFMultiPanelGrid) {
        window.HFMultiPanelGrid.sync({
          cue: cue,
          duration: opts.duration || 0,
          playing: !!opts.playing,
          treatment: opts.gridTreatment || opts.treatment || 'scale_emphasis',
          elapsed: opts.elapsed || 0,
          forceRestart: !!opts.forceRestart,
        });
      }
      return;
    }

    if (isSingleHeroCue(cue)) {
      activate('hero');
      if (window.HFHeroStills) {
        window.HFHeroStills.sync({
          cue: cue,
          duration: opts.duration || 0,
          playing: !!opts.playing,
          treatment: opts.treatment || 'still',
          elapsed: opts.elapsed || 0,
          forceRestart: !!opts.forceRestart,
          cueWords: opts.cueWords || null,
          cueLead: opts.cueLead || 0,
          bboxTreatment:
            opts.bboxTreatment || ((window.HFTreatmentRotator && window.HFTreatmentRotator.getBboxTreatment && cue)
              ? window.HFTreatmentRotator.getBboxTreatment(cue)
              : "bbox_draw"),
        });
      }
      return;
    }

    activate('');
  }

  function kill() {
    if (active) killModule(active);
    active = '';
  }

  return {
    sync: sync,
    syncHeroStill: sync,
    kill: kill,
  };
})();
