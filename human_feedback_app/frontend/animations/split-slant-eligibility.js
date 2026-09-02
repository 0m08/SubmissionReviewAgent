// split-slant-eligibility.js
// Slant-split eligibility (aspect fill + seam safety) and smart label treatment assignment.
window.HFSplitSlant = (function () {
  const SLOT_W = 904;
  const SLOT_H = 748;
  const CANVAS_W = 1808;
  const CANVAS_H = 748;

  const ALL_TREATMENTS = ["label_slant_fade", "label_slide", "label_fade"];

  function themeNum(key, fallback) {
    if (window.HFPlayerTheme && typeof window.HFPlayerTheme.get === "function") {
      const raw = window.HFPlayerTheme.get(key);
      const n = parseFloat(String(raw || "").trim());
      if (isFinite(n)) return n;
    }
    return fallback;
  }

  function config() {
    return {
      minAreaFill: themeNum("split_slant_min_area_fill", 0.48),
      minDimFill: themeNum("split_slant_min_dim_fill", 0.58),
      topXPct: themeNum("split_slant_top_x_pct", 58) / 100,
      bottomXPct: themeNum("split_slant_bottom_x_pct", 42) / 100,
      seamMarginPct: themeNum("split_slant_seam_margin_pct", 8) / 100,
    };
  }

  function visibleRect(iw, ih, cw, ch) {
    if (!iw || !ih || !cw || !ch) {
      return { x: 0, y: 0, w: cw, h: ch };
    }
    const containerRatio = cw / ch;
    const imageRatio = iw / ih;
    let w;
    let h;
    let x;
    let y;
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

  function areaFill(iw, ih, cw, ch) {
    const r = visibleRect(iw, ih, cw, ch);
    return (r.w * r.h) / (cw * ch);
  }

  function dimFills(iw, ih, cw, ch) {
    const r = visibleRect(iw, ih, cw, ch);
    return {
      width: r.w / cw,
      height: r.h / ch,
      area: (r.w * r.h) / (cw * ch),
    };
  }

  function slantXGlobal(y, cfg) {
    const topX = cfg.topXPct * CANVAS_W;
    const bottomX = cfg.bottomXPct * CANVAS_W;
    return topX + (bottomX - topX) * (y / CANVAS_H);
  }

  function imageUrlFromStep(step, resolveImageUrl) {
    if (!step) return "";
    if (typeof resolveImageUrl === "function") {
      return resolveImageUrl(step) || "";
    }
    if (step.type === "video") return "";
    const raw = step.assetUrl || step.url || "";
    if (!raw) return "";
    const lower = String(raw).toLowerCase();
    if (
      lower.includes("drive.google.com") ||
      lower.includes("docs.google.com") ||
      lower.includes("googleusercontent.com")
    ) {
      return "/api/assets/image?thumb=1&url=" + encodeURIComponent(raw);
    }
    return raw;
  }

  function preloadImage(url) {
    return new Promise(function (resolve) {
      if (!url) {
        resolve(null);
        return;
      }
      const img = new Image();
      img.decoding = "async";
      img.onload = function () {
        resolve(img);
      };
      img.onerror = function () {
        resolve(null);
      };
      img.src = url;
    });
  }

  function seamSafeForSlot(slotIndex, iw, ih, cfg) {
    const r = visibleRect(iw, ih, SLOT_W, SLOT_H);
    const fills = dimFills(iw, ih, SLOT_W, SLOT_H);
    if (fills.area < cfg.minAreaFill) return false;
    if (fills.width < cfg.minDimFill || fills.height < cfg.minDimFill) return false;

    const sampleYs = [SLOT_H * 0.28, SLOT_H * 0.42, SLOT_H * 0.55, SLOT_H * 0.68];
    const margin = cfg.seamMarginPct * SLOT_W;
    const innerStart = slotIndex === 0 ? r.x + r.w * 0.12 : r.x + r.w * 0.05;
    const innerEnd = slotIndex === 0 ? r.x + r.w * 0.92 : r.x + r.w * 0.88;

    for (let i = 0; i < sampleYs.length; i++) {
      const y = sampleYs[i];
      if (y < r.y - 2 || y > r.y + r.h + 2) continue;
      const globalY = y;
      const slantGlobalX = slantXGlobal(globalY, cfg);
      const slantLocalX = slantGlobalX - slotIndex * SLOT_W;
      if (slantLocalX >= innerStart - margin && slantLocalX <= innerEnd + margin) {
        return false;
      }
    }
    return true;
  }

  function computeSlantClipPolygons(cfg) {
    const topX = cfg.topXPct * CANVAS_W;
    const bottomX = cfg.bottomXPct * CANVAS_W;
    const yCross = ((topX - SLOT_W) / (topX - bottomX)) * CANVAS_H;
    const yCrossPct = Math.max(0, Math.min(100, (yCross / SLOT_H) * 100));
    const leftBottomPct = Math.max(0, Math.min(100, (bottomX / SLOT_W) * 100));
    const rightTopPct = Math.max(0, Math.min(100, ((topX - SLOT_W) / SLOT_W) * 100));

    const left = yCrossPct > 0.5
      ? "polygon(0 0, 100% 0, 100% " + yCrossPct.toFixed(2) + "%, "
        + leftBottomPct.toFixed(2) + "% 100%, 0 100%)"
      : "polygon(0 0, 100% 0, " + leftBottomPct.toFixed(2) + "% 100%, 0 100%)";

    const right = "polygon(" + rightTopPct.toFixed(2) + "% 0, 100% 0, 100% 100%, 0 100%)";
    return { left: left, right: right, yCrossPct: yCrossPct, rightTopPct: rightTopPct };
  }

  function checkSceneEligible(step0, step1, resolveImageUrl) {
    const cfg = config();
    if (!step0 || !step1) return Promise.resolve(false);
    if (step0.type === "video" || step1.type === "video") return Promise.resolve(false);

    const url0 = imageUrlFromStep(step0, resolveImageUrl);
    const url1 = imageUrlFromStep(step1, resolveImageUrl);
    if (!url0 || !url1) return Promise.resolve(false);

    return Promise.all([preloadImage(url0), preloadImage(url1)]).then(function (imgs) {
      const img0 = imgs[0];
      const img1 = imgs[1];
      if (!img0 || !img1 || !img0.naturalWidth || !img1.naturalWidth) return false;
      if (!seamSafeForSlot(0, img0.naturalWidth, img0.naturalHeight, cfg)) return false;
      if (!seamSafeForSlot(1, img1.naturalWidth, img1.naturalHeight, cfg)) return false;
      return true;
    });
  }

  function assignTreatments(scenesWithEligibility) {
    let previousAssigned = null;
    let previousSlantSuccess = false;
    const lastUsedIndex = {
      label_slant_fade: -999,
      label_slide: -999,
      label_fade: -999,
    };

    return (scenesWithEligibility || []).map(function (scene, sceneIndex) {
      let candidates = ALL_TREATMENTS.slice();

      if (!scene.slantEligible) {
        candidates = candidates.filter(function (t) {
          return t !== "label_slant_fade";
        });
      }
      if (previousSlantSuccess) {
        candidates = candidates.filter(function (t) {
          return t !== "label_slant_fade";
        });
      }
      if (previousAssigned) {
        candidates = candidates.filter(function (t) {
          return t !== previousAssigned;
        });
      }
      if (!candidates.length) {
        candidates = ALL_TREATMENTS.filter(function (t) {
          return t !== previousAssigned;
        });
      }

      let picked;
      if (candidates.indexOf("label_slant_fade") >= 0) {
        picked = "label_slant_fade";
      } else {
        candidates.sort(function (a, b) {
          const diff = lastUsedIndex[a] - lastUsedIndex[b];
          if (diff !== 0) return diff;
          return ALL_TREATMENTS.indexOf(a) - ALL_TREATMENTS.indexOf(b);
        });
        picked = candidates[0] || "label_fade";
      }

      lastUsedIndex[picked] = sceneIndex;
      previousAssigned = picked;
      previousSlantSuccess = picked === "label_slant_fade";

      return {
        key: scene.key,
        cue: scene.cue,
        treatment: picked,
        index: sceneIndex,
        occurrence: sceneIndex,
      };
    });
  }

  function collectSplitLabelScenes(cues, layoutKeyFn, sceneKeyFn) {
    const scenes = [];
    const seen = Object.create(null);
    const stepsByScene = Object.create(null);

    (cues || []).forEach(function (cue) {
      if (!cue) return;
      if ((cue.animationType || "none") !== "text_label") return;
      const layout = layoutKeyFn(cue.sceneTemplate || "");
      if (layout !== "two_item_split_comparison") return;
      const key = sceneKeyFn(cue.slideIdx, cue.sceneId);
      if (!stepsByScene[key]) stepsByScene[key] = Object.create(null);
      if (cue.step) stepsByScene[key][cue.partIdx] = cue.step;
      if (!seen[key]) {
        seen[key] = true;
        scenes.push({ key: key, cue: cue });
      }
    });

    return scenes.map(function (scene) {
      return {
        key: scene.key,
        cue: scene.cue,
        steps: stepsByScene[scene.key] || {},
      };
    });
  }

  function assignSplitLabelScenes(cues, resolveImageUrl, layoutKeyFn, sceneKeyFn) {
    const scenes = collectSplitLabelScenes(cues, layoutKeyFn, sceneKeyFn);
    if (!scenes.length) return Promise.resolve([]);

    return Promise.all(
      scenes.map(function (scene) {
        const steps = scene.steps || {};
        return checkSceneEligible(steps[0], steps[1], resolveImageUrl).then(function (eligible) {
          return {
            key: scene.key,
            cue: scene.cue,
            slantEligible: !!eligible,
          };
        });
      })
    ).then(function (scenesWithEligibility) {
      return assignTreatments(scenesWithEligibility);
    });
  }

  return {
    config: config,
    visibleRect: visibleRect,
    areaFill: areaFill,
    computeSlantClipPolygons: computeSlantClipPolygons,
    checkSceneEligible: checkSceneEligible,
    assignTreatments: assignTreatments,
    assignSplitLabelScenes: assignSplitLabelScenes,
    SLOT_W: SLOT_W,
    SLOT_H: SLOT_H,
    CANVAS_W: CANVAS_W,
    CANVAS_H: CANVAS_H,
  };
})();
