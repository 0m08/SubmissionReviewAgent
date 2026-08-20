// treatment-rotator.js
// Whole-course per-layout animation cycling for the Player.
// Each layout has its own ordered list and counter. Counters advance once per
// unique scene (not per part). When a list is exhausted, it wraps.
window.HFTreatmentRotator = (function () {
  const LISTS = {
    single_visual_hero: ['zoom_in', 'still', 'center_split'],
    two_item_split_comparison: [
      'hold_and_reveal',
      'scale_emphasis',
      'slide_in',
      'wipe_reveal',
      'slide_up',
      'center_then_split',
    ],
    main_plus_supporting_inset: ['keep_main_reveal', 'inset_slide'],
    multi_panel_grid: ['scale_emphasis', 'reveal', 'slide_in'],
  };

  const TOPIC_TRANSITION_LIST = [
    'fade_in_right',
    'fade_in_top_right',
    'fade_in_bottom_right',
    'fade_in_only'
  ];

  const DEFAULTS = {
    single_visual_hero: 'zoom_in',
    two_item_split_comparison: 'hold_and_reveal',
    main_plus_supporting_inset: 'keep_main_reveal',
    multi_panel_grid: 'scale_emphasis',
  };

  let byScene = Object.create(null);

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

  function isTopicTransition(cue) {
    if (!cue) return false;
    if ((cue.slideType || '').trim().toLowerCase() !== 'transition') return false;
    const t = String(cue.topic || '').trim().toLowerCase().replace(/[^a-z0-9]/g, '');
    const c = String(cue.slideChunk || '').trim().toLowerCase().replace(/[^a-z0-9]/g, '');
    const st = String(cue.slideTitle || '').trim().toLowerCase().replace(/[^a-z0-9]/g, '');
    return t && (c === t || st === t);
  }

  function sceneKey(slideIdx, sceneId) {
    return String(slideIdx) + ':' + String(sceneId);
  }

  function layoutKey(template) {
    const t = normalizeTemplate(template);
    return LISTS[t] ? t : '';
  }

  // Rebuild from player cues in play order. Unique scenes only.
  function rebuildFromCues(cues) {
    byScene = Object.create(null);
    const counters = {
      single_visual_hero: 0,
      two_item_split_comparison: 0,
      main_plus_supporting_inset: 0,
      multi_panel_grid: 0,
    };
    const seen = Object.create(null);

    const topicStartIndex = Math.floor(Math.random() * 4);
    let topicTransitionCounter = 0;

    (cues || []).forEach(function (cue) {
      if (!cue) return;
      const key = sceneKey(cue.slideIdx, cue.sceneId);
      if (seen[key]) return;
      seen[key] = true;

      if (isTopicTransition(cue)) {
        const idx = (topicStartIndex + topicTransitionCounter) % 4;
        byScene[key] = {
          layout: 'topic_transition',
          treatment: TOPIC_TRANSITION_LIST[idx],
          index: idx,
          occurrence: topicTransitionCounter,
        };
        topicTransitionCounter += 1;
        return;
      }

      const layout = layoutKey(cue.sceneTemplate || '');
      if (!layout) return;

      const animType = cue.animationType || 'none';
      if (layout === 'single_visual_hero' && animType !== 'none') {
        byScene[key] = {
          layout: layout,
          treatment: 'still',
          index: -1,
          occurrence: -1,
        };
        return;
      }

      const list = LISTS[layout];
      const idx = counters[layout] % list.length;
      byScene[key] = {
        layout: layout,
        treatment: list[idx],
        index: idx,
        occurrence: counters[layout],
      };
      counters[layout] += 1;
    });
  }

  function getEntry(cue) {
    if (!cue) return null;
    return byScene[sceneKey(cue.slideIdx, cue.sceneId)] || null;
  }

  function getTreatment(cue) {
    const entry = getEntry(cue);
    if (entry) return entry.treatment;
    const layout = layoutKey(cue && cue.sceneTemplate);
    return layout ? DEFAULTS[layout] : 'off';
  }

  function getLayout(cue) {
    const entry = getEntry(cue);
    if (entry) return entry.layout;
    return layoutKey(cue && cue.sceneTemplate);
  }

  function treatmentsFor(layout) {
    const key = layoutKey(layout);
    return key ? LISTS[key].slice() : [];
  }

  return {
    rebuildFromCues: rebuildFromCues,
    getTreatment: getTreatment,
    getLayout: getLayout,
    getEntry: getEntry,
    treatmentsFor: treatmentsFor,
    normalizeTemplate: normalizeTemplate,
    LISTS: LISTS,
  };
})();
