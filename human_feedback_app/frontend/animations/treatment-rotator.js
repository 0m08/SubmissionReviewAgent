// treatment-rotator.js
// Whole-course per-layout animation cycling for the Player.
// Allowed treatments come from config (layout_anims_* / label_anims_*).
// Each layout has its own ordered list and counter. Counters advance once per
// unique scene (not per part). When a list is exhausted, it wraps.
window.HFTreatmentRotator = (function () {
  const FALLBACK_LISTS = {
    single_visual_hero: ["zoom_in", "still", "center_split"],
    two_item_split_comparison: [
      "hold_and_reveal",
      "scale_emphasis",
      "slide_in",
      "wipe_reveal",
      "slide_up",
      "center_then_split",
    ],
    main_plus_supporting_inset: ["keep_main_reveal", "inset_slide"],
    multi_panel_grid: ["scale_emphasis", "reveal", "slide_in"],
  };

  const FALLBACK_TOPIC = [
    "fade_in_right",
    "fade_in_top_right",
    "fade_in_bottom_right",
    "fade_in_only",
  ];

  const FALLBACK_LABELS = {
    two_item_split_comparison: ["label_slide", "label_fade"],
    main_plus_supporting_inset: ["label_slide", "label_fade"],
    multi_panel_3: ["label_slide_up", "label_fade"],
    multi_panel_4: ["label_corner_slide", "label_fade"],
  };

  const FALLBACK_CALLOUT = ["callout_slide", "callout_fade"];
  const FALLBACK_BBOX = ["bbox_draw", "bbox_spotlight"];

  let byBbox = Object.create(null);

  const CONFIG_KEYS = {
    single_visual_hero: "layout_anims_hero",
    two_item_split_comparison: "layout_anims_split",
    main_plus_supporting_inset: "layout_anims_inset",
    multi_panel_grid: "layout_anims_multi_panel",
  };

  const LABEL_CONFIG_KEYS = {
    two_item_split_comparison: "label_anims_split",
    main_plus_supporting_inset: "label_anims_inset",
    multi_panel_3: "label_anims_multi_panel_3",
    multi_panel_4: "label_anims_multi_panel_4",
  };

  const CALLOUT_POSITIONS = [
    "top_left",
    "top_center",
    "top_right",
    "center_left",
    "center",
    "center_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
  ];

  let byScene = Object.create(null);
  let byCallout = Object.create(null);
  let lastCues = null;

  function themeList(key, fallback) {
    if (window.HFPlayerTheme && typeof window.HFPlayerTheme.list === "function") {
      const fromTheme = window.HFPlayerTheme.list(key, (fallback || []).join(", "));
      if (fromTheme && fromTheme.length) return fromTheme;
    }
    return (fallback || []).slice();
  }

  function resolvedLists() {
    return {
      single_visual_hero: themeList(
        CONFIG_KEYS.single_visual_hero,
        FALLBACK_LISTS.single_visual_hero
      ),
      two_item_split_comparison: themeList(
        CONFIG_KEYS.two_item_split_comparison,
        FALLBACK_LISTS.two_item_split_comparison
      ),
      main_plus_supporting_inset: themeList(
        CONFIG_KEYS.main_plus_supporting_inset,
        FALLBACK_LISTS.main_plus_supporting_inset
      ),
      multi_panel_grid: themeList(
        CONFIG_KEYS.multi_panel_grid,
        FALLBACK_LISTS.multi_panel_grid
      ),
    };
  }

  function topicList() {
    return themeList("layout_anims_topic", FALLBACK_TOPIC);
  }

  function labelListFor(layout, cue) {
    if (layout === "multi_panel_grid") {
      const slotCount = (cue && cue.partCount) || 3;
      if (slotCount === 4) {
        return themeList(LABEL_CONFIG_KEYS.multi_panel_4, FALLBACK_LABELS.multi_panel_4);
      }
      return themeList(LABEL_CONFIG_KEYS.multi_panel_3, FALLBACK_LABELS.multi_panel_3);
    }
    if (layout === "two_item_split_comparison") {
      return themeList(
        LABEL_CONFIG_KEYS.two_item_split_comparison,
        FALLBACK_LABELS.two_item_split_comparison
      );
    }
    if (layout === "main_plus_supporting_inset") {
      return themeList(
        LABEL_CONFIG_KEYS.main_plus_supporting_inset,
        FALLBACK_LABELS.main_plus_supporting_inset
      );
    }
    return ["label_fade"];
  }

  function defaultsFrom(lists) {
    return {
      single_visual_hero: lists.single_visual_hero[0] || "zoom_in",
      two_item_split_comparison: lists.two_item_split_comparison[0] || "hold_and_reveal",
      main_plus_supporting_inset: lists.main_plus_supporting_inset[0] || "keep_main_reveal",
      multi_panel_grid: lists.multi_panel_grid[0] || "scale_emphasis",
    };
  }

  function normalizeTemplate(template) {
    const t = String(template || "")
      .trim()
      .toLowerCase()
      .replace(/[\s-]+/g, "_");
    const aliases = {
      single_hero: "single_visual_hero",
      hero: "single_visual_hero",
      two_item_split: "two_item_split_comparison",
      split_comparison: "two_item_split_comparison",
      main_plus_inset: "main_plus_supporting_inset",
      main_plus_supporting: "main_plus_supporting_inset",
      main_visual_plus_inset: "main_plus_supporting_inset",
      multi_panel: "multi_panel_grid",
      grid: "multi_panel_grid",
    };
    return aliases[t] || t;
  }

  function isTopicTransition(cue) {
    if (!cue) return false;
    if ((cue.slideType || "").trim().toLowerCase() !== "transition") return false;
    const t = String(cue.topic || "")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");
    const c = String(cue.slideChunk || "")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");
    const st = String(cue.slideTitle || "")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");
    return t && (c === t || st === t);
  }

  function sceneKey(slideIdx, sceneId) {
    return String(slideIdx) + ":" + String(sceneId);
  }

  function layoutKey(template) {
    const t = normalizeTemplate(template);
    return FALLBACK_LISTS[t] ? t : "";
  }

  function normalizeCalloutPosition(position) {
    const pos = String(position || "center_left")
      .trim()
      .toLowerCase()
      .replace(/[\s-]+/g, "_");
    return CALLOUT_POSITIONS.indexOf(pos) >= 0 ? pos : "center_left";
  }

  function calloutCardKey(slideIdx, sceneId, cardIdx) {
    return String(slideIdx) + ":" + String(sceneId) + ":" + String(cardIdx);
  }

  function calloutList() {
    return themeList("callout_anims", FALLBACK_CALLOUT);
  }

  function bboxList() {
    return themeList("bbox_anims", FALLBACK_BBOX);
  }

  function rebuildBboxTreatments(cues) {
    byBbox = Object.create(null);
    const anims = bboxList();
    const animLen = Math.max(anims.length, 1);
    const seen = Object.create(null);
    let counter = 0;
    (cues || []).forEach(function (cue) {
      if (!cue || (cue.animationType || "none") !== "bbox_highlight") return;
      const key = sceneKey(cue.slideIdx, cue.sceneId);
      if (seen[key]) return;
      seen[key] = true;
      byBbox[key] = anims[counter % animLen] || "bbox_draw";
      counter += 1;
    });
  }

  // Per grid position across the course: cycle callout_anims from config.
  function rebuildCalloutEntrances(cues) {
    byCallout = Object.create(null);
    const positionCounts = Object.create(null);
    const anims = calloutList();
    const animLen = Math.max(anims.length, 1);
    CALLOUT_POSITIONS.forEach(function (pos) {
      positionCounts[pos] = 0;
    });

    (cues || []).forEach(function (cue) {
      if (!cue || (cue.animationType || "none") !== "callout_card") return;
      const cards = cue.calloutCards || [];
      cards.forEach(function (callout, cardIdx) {
        const pos = normalizeCalloutPosition(callout && callout.position);
        positionCounts[pos] += 1;
        const occurrence = positionCounts[pos];
        const idx = (occurrence - 1) % animLen;
        byCallout[calloutCardKey(cue.slideIdx, cue.sceneId, cardIdx)] = {
          entrance: anims[idx] || "callout_fade",
          position: pos,
          occurrence: occurrence,
        };
      });
    });
  }

  // Rebuild from player cues in play order. Unique scenes only.
  function rebuildFromCues(cues) {
    lastCues = cues || [];
    byScene = Object.create(null);
    rebuildCalloutEntrances(lastCues);
    rebuildBboxTreatments(lastCues);
    const lists = resolvedLists();
    const topics = topicList();
    const defaults = defaultsFrom(lists);
    const counters = {
      single_visual_hero: 0,
      two_item_split_comparison: 0,
      main_plus_supporting_inset: 0,
      multi_panel_grid: 0,
    };
    const labelCounters = {
      two_item_split_comparison: 0,
      main_plus_supporting_inset: 0,
      multi_panel_grid: 0,
    };
    const seen = Object.create(null);

    const topicLen = Math.max(topics.length, 1);
    const topicStartIndex = Math.floor(Math.random() * topicLen);
    let topicTransitionCounter = 0;

    lastCues.forEach(function (cue) {
      if (!cue) return;
      const key = sceneKey(cue.slideIdx, cue.sceneId);
      if (seen[key]) return;
      seen[key] = true;

      if (isTopicTransition(cue)) {
        const idx = (topicStartIndex + topicTransitionCounter) % topicLen;
        byScene[key] = {
          layout: "topic_transition",
          treatment: topics[idx] || FALLBACK_TOPIC[0],
          index: idx,
          occurrence: topicTransitionCounter,
        };
        topicTransitionCounter += 1;
        return;
      }

      const layout = layoutKey(cue.sceneTemplate || "");
      if (!layout) return;

      if (cue.sceneHasVideo) {
        byScene[key] = {
          layout: layout,
          treatment: "off",
          index: -1,
          occurrence: -1,
        };
        return;
      }

      const animType = cue.animationType || "none";
      if (layout === "single_visual_hero" && animType !== "none") {
        byScene[key] = {
          layout: layout,
          treatment: "still",
          index: -1,
          occurrence: -1,
        };
        return;
      }

      if (
        animType === "text_label" &&
        (layout === "two_item_split_comparison" ||
          layout === "main_plus_supporting_inset" ||
          layout === "multi_panel_grid")
      ) {
        const labelList = labelListFor(layout, cue);
        const idx = labelCounters[layout] % labelList.length;
        byScene[key] = {
          layout: layout,
          treatment: labelList[idx],
          index: idx,
          occurrence: labelCounters[layout],
        };
        labelCounters[layout] += 1;
        return;
      }

      let list = lists[layout];
      if (layout === "multi_panel_grid") {
        const slotCount = (cue && cue.partCount) || 3;
        if (slotCount === 4) {
          list = themeList("layout_anims_multi_panel_4", ["scale_emphasis", "reveal", "slide_in"]);
        } else {
          list = themeList("layout_anims_multi_panel_3", ["scale_emphasis", "reveal", "slide_in"]);
        }
      }
      if (!list || !list.length) {
        byScene[key] = {
          layout: layout,
          treatment: defaults[layout] || "off",
          index: 0,
          occurrence: counters[layout],
        };
        counters[layout] += 1;
        return;
      }
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

  function reloadLists() {
    if (lastCues) rebuildFromCues(lastCues);
  }

  function getEntry(cue) {
    if (!cue) return null;
    return byScene[sceneKey(cue.slideIdx, cue.sceneId)] || null;
  }

  function getTreatment(cue) {
    const entry = getEntry(cue);
    if (entry) return entry.treatment;
    const layout = layoutKey(cue && cue.sceneTemplate);
    if (layout && cue && cue.animationType === "text_label") {
      const labelList = labelListFor(layout, cue);
      return labelList[0] || "label_fade";
    }
    const lists = resolvedLists();
    const defaults = defaultsFrom(lists);
    return layout ? defaults[layout] : "off";
  }

  function getLayout(cue) {
    const entry = getEntry(cue);
    if (entry) return entry.layout;
    return layoutKey(cue && cue.sceneTemplate);
  }

  function treatmentsFor(layout) {
    const key = layoutKey(layout);
    if (!key) return [];
    return resolvedLists()[key].slice();
  }

  function getBboxTreatment(cue) {
    if (!cue) return "bbox_draw";
    const key = sceneKey(cue.slideIdx, cue.sceneId);
    if (byBbox[key]) return byBbox[key];
    const anims = bboxList();
    return anims[0] || "bbox_draw";
  }

  function getCalloutEntrance(cue, cardIdx) {
    if (!cue) {
      return { entrance: "callout_fade", position: "center_left", occurrence: 0 };
    }
    const key = calloutCardKey(cue.slideIdx, cue.sceneId, cardIdx);
    if (byCallout[key]) return byCallout[key];
    const pos = normalizeCalloutPosition(
      cue.calloutCards &&
        cue.calloutCards[cardIdx] &&
        cue.calloutCards[cardIdx].position
    );
    const anims = calloutList();
    return { entrance: anims[0] || "callout_fade", position: pos, occurrence: 1 };
  }

  return {
    rebuildFromCues: rebuildFromCues,
    reloadLists: reloadLists,
    getTreatment: getTreatment,
    getLayout: getLayout,
    getEntry: getEntry,
    getCalloutEntrance: getCalloutEntrance,
    getBboxTreatment: getBboxTreatment,
    bboxList: bboxList,
    calloutList: calloutList,
    normalizeCalloutPosition: normalizeCalloutPosition,
    treatmentsFor: treatmentsFor,
    normalizeTemplate: normalizeTemplate,
    get LISTS() {
      return resolvedLists();
    },
  };
})();
