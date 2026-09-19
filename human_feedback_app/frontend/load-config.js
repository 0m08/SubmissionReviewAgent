// Loads /api/player-theme (human_feedback_app/player_config.py) and exposes window.HFPlayerTheme.
// Currently slide title, captions, topic card, hero text_label, panel labels, and callout cards are configurable.
(function () {
  const defaults = {
    slide_title_font: "Fira Sans",
    slide_title_font_weight: "ExtraBold",
    slide_title_font_size: "20",
    slide_title_text_color: "#f35a1f",
    slide_title_bg: "rgba(255, 255, 255, 0.8)",
    slide_title_accent: "#ff5a17",
    slide_title_chip_width: "42",
    slide_title_height: "32",
    slide_title_radius: "0",
    slide_title_padding: "0 12px 0 10px",
    slide_title_shadow: "0 1px 1px rgba(0, 0, 0, 0.05)",
    slide_title_blur: "4",

    caption_font: "Open Sans",
    caption_font_weight: "Regular",
    caption_font_size: "13",
    caption_text_color: "#FFFFFF",
    caption_bg: "#31373a",
    caption_radius: "4",
    caption_padding: "10px",
    caption_max_width: "60%",
    caption_line_height: "1.5",
    caption_text_shadow: "-1px 0 #000, 0 1px #000, 1px 0 #000, 0 -1px #000",

    topic_card_font: "Public Sans",
    topic_card_font_weight: "ExtraBold",
    topic_card_font_size: "44",
    topic_card_secondary_font_size: "44",
    topic_card_text_color: "#FFFFFF",
    topic_card_secondary_text_color: "none",
    topic_card_text_align: "center",
    topic_card_text_transform: "none",
    topic_card_bg: "#000066",
    topic_card_border: "none",
    topic_card_radius: "22",
    topic_card_padding: "24px 32px",
    topic_card_max_width: "520px",
    topic_card_position_inset: "6%",
    topic_card_shadow: "0 10px 30px rgba(0, 0, 0, 0.4)",
    topic_card_line_height: "1.15",

    hero_label_font: "Public Sans",
    hero_label_font_weight: "Bold",
    hero_label_font_size: "26",
    hero_label_text_color: "#FFFFFF",
    hero_label_text_align: "center",
    hero_label_bg: "#f05725",
    hero_label_radius: "24",
    hero_label_padding: "18px 36px",
    hero_label_max_width: "90%",
    hero_label_white_space: "nowrap",
    hero_label_shadow: "0 8px 24px rgba(0, 0, 0, 0.28)",
    hero_label_line_height: "1.25",

    panel_label_font: "Public Sans",
    panel_label_font_weight: "Bold",
    panel_label_font_size: "18",
    panel_label_text_color: "#FFFFFF",
    panel_label_text_align: "center",
    panel_label_text_transform: "none",
    panel_label_bg: "#f05725",
    panel_label_radius: "10",
    panel_label_padding: "10px 22px",
    panel_label_shadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
    panel_label_white_space: "nowrap",
    panel_label_letter_spacing: "0",
    panel_label_bottom: "15",

    layout_anims_hero: "zoom_in, still, center_split",
    layout_anims_split:
      "hold_and_reveal, scale_emphasis, slide_in, wipe_reveal, slide_up, center_then_split",
    layout_anims_inset: "keep_main_reveal, inset_slide",
    layout_anims_multi_panel_3: "scale_emphasis, reveal, slide_in",
    layout_anims_multi_panel_4: "scale_emphasis, reveal, slide_in",
    layout_anims_topic:
      "fade_in_right, fade_in_top_right, fade_in_bottom_right, fade_in_only",
    label_anims_split: "label_slide, label_fade",
    label_anims_inset: "label_slide, label_fade",
    label_anims_multi_panel_3: "label_slide_up, label_fade",
    label_anims_multi_panel_4: "label_corner_slide, label_fade",

    callout_anims: "callout_slide, callout_fade",

    bbox_anims: "bbox_draw, bbox_spotlight",
    bbox_spotlight_veil_opacity: "0.75",
    bbox_spotlight_border_color: "#F9E400",
    bbox_spotlight_border_width: "3",
    bbox_spotlight_border_dash: "18 10",
    bbox_spotlight_border_march_cycle: "1.1",
    bbox_spotlight_corner_radius: "28",
    bbox_spotlight_highlight_delay: "0.28",
    bbox_draw_stroke_color: "#f05523",
    bbox_draw_stroke_width: "4.5",

    callout_card_variant: "dark",
    callout_card_bg:
      "linear-gradient(90deg, rgba(0,0,0,0.72) 0%, rgba(0,0,0,0.45) 55%, rgba(0,0,0,0.12) 100%)",
    callout_card_radius: "30",
    callout_card_border_width: "4",
    callout_card_border_gradient:
      "linear-gradient(135deg, #FF5C26 0%, #FF5C26 28%, rgba(255,92,38,0.55) 70%, rgba(255,92,38,0.4) 100%)",
    callout_card_shadow: "0 8px 24px rgba(0, 0, 0, 0.35)",
    callout_card_padding: "18px 22px 28px",
    callout_card_max_width: "42%",
    callout_card_max_width_multi_2: "34%",
    callout_card_max_width_multi_3: "32%",
    callout_card_position_inset: "6%",
    callout_card_bounds: "frame",
    callout_card_header_font: "Fira Sans",
    callout_card_header_font_weight: "Bold",
    callout_card_header_font_size: "32",
    callout_card_header_text_color: "#FF5C26",
    callout_card_header_text_transform: "uppercase",
    callout_card_header_gap: "12",
    callout_card_icon_size: "52",
    callout_card_icon_knockout_white: "true",
    callout_card_divider:
      "linear-gradient(90deg, #FF5C26 0%, rgba(255, 92, 38, 0.35) 70%, rgba(255, 92, 38, 0) 100%)",
    callout_card_divider_thickness: "2",
    callout_card_divider_width: "85%",
    callout_card_divider_margin: "6px 0 14px 0",
    callout_card_body_font: "Fira Sans",
    callout_card_body_font_weight: "Bold",
    callout_card_body_font_size: "36",
    callout_card_body_text_color: "#FFFFFF",
    callout_card_body_text_align: "left",
    callout_card_body_line_height: "1.25",
    callout_card_body_max_lines: "0",
    callout_scene_overlay:
      "linear-gradient(to left, rgba(0,0,0,0.68) 0%, rgba(0,0,0,0.31) 55%, rgba(0,0,0,0) 100%)",
  };

  const WEIGHT_MAP = {
    regular: "400",
    normal: "400",
    medium: "500",
    semibold: "600",
    "semi bold": "600",
    "semi-bold": "600",
    bold: "700",
    extrabold: "800",
    "extra bold": "800",
    "extra-bold": "800",
    black: "900",
  };

  const values = Object.assign({}, defaults);
  let scopedValues = Object.create(null);
  const loadedFonts = Object.create(null);

  const SYSTEM_FONTS = {
    arial: 1,
    helvetica: 1,
    "helvetica neue": 1,
    "times new roman": 1,
    times: 1,
    georgia: 1,
    "courier new": 1,
    courier: 1,
    "segoe ui": 1,
    "system-ui": 1,
    "sans-serif": 1,
    serif: 1,
    monospace: 1,
    "ui-sans-serif": 1,
    "ui-serif": 1,
    "ui-monospace": 1,
  };

  function get(key) {
    const k = String(key || "").toLowerCase();
    if (Object.prototype.hasOwnProperty.call(scopedValues, k) && String(scopedValues[k]).trim() !== "") {
      return String(scopedValues[k]).trim();
    }
    if (Object.prototype.hasOwnProperty.call(values, k) && String(values[k]).trim() !== "") {
      return String(values[k]).trim();
    }
    if (Object.prototype.hasOwnProperty.call(defaults, k)) return defaults[k];
    return "";
  }

  function fontStack(fontKey) {
    const name = get(fontKey) || "Fira Sans";
    return "'" + name + "', system-ui, sans-serif";
  }

  function fontWeight(weightKey) {
    let raw = get(weightKey) || "400";
    raw = String(raw).replace(/^["']|["']$/g, "").trim();
    const mapped =
      WEIGHT_MAP[raw.toLowerCase()] ||
      WEIGHT_MAP[raw.toLowerCase().replace(/[\s_-]+/g, "")];
    if (mapped) return mapped;
    if (/^\d{3}$/.test(raw)) return raw;
    return "400";
  }

  // Bare numbers → px. Pass through auto, %, multi-value, functions, etc.
  function cssValue(key, fallback) {
    const raw = get(key) || fallback || "";
    if (!raw) return fallback || "";
    if (/^(auto|none|0)$/i.test(raw)) return raw.toLowerCase() === "0" ? "0px" : raw;
    if (/[a-z%)]|\s/i.test(raw)) return raw;
    if (/^\d+(\.\d+)?$/.test(raw)) return raw + "px";
    return raw;
  }

  // Comma-separated config lists → trimmed non-empty tokens.
  function list(key, fallback) {
    const raw = get(key);
    const src = raw != null && String(raw).trim() !== "" ? raw : fallback || "";
    return String(src)
      .split(",")
      .map(function (part) {
        return part.trim();
      })
      .filter(Boolean);
  }

  function fontSizePx(sizeKey) {
    return cssValue(sizeKey, "14");
  }

  function isSystemFont(name) {
    return !!SYSTEM_FONTS[String(name || "").trim().toLowerCase()];
  }

  function ensureGoogleFontLoaded(fontName) {
    const name = String(fontName || "").trim();
    if (!name || isSystemFont(name)) return;
    const key = name.toLowerCase();
    if (loadedFonts[key]) return;
    loadedFonts[key] = true;

    const familyParam = encodeURIComponent(name).replace(/%20/g, "+");
    const href =
      "https://fonts.googleapis.com/css2?family=" +
      familyParam +
      ":ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&display=swap";

    if (document.querySelector('link[data-hf-player-font="' + key + '"]')) return;

    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    link.setAttribute("data-hf-player-font", key);
    document.head.appendChild(link);
  }

  function apply() {
    const root = document.documentElement;
    if (!root) return;

    // Apply generic CSS variables for every key in values dictionary
    const effectiveValues = Object.assign({}, values, scopedValues);
    Object.keys(effectiveValues).forEach(function (k) {
      const cssProp = "--hf-" + k.replace(/_/g, "-");
      root.style.setProperty(cssProp, effectiveValues[k]);
      if (k.indexOf("font") !== -1 && k.indexOf("weight") === -1 && k.indexOf("size") === -1) {
        ensureGoogleFontLoaded(effectiveValues[k]);
      }
    });

    ensureGoogleFontLoaded(get("slide_title_font"));
    ensureGoogleFontLoaded(get("caption_font"));
    ensureGoogleFontLoaded(get("topic_card_font"));
    ensureGoogleFontLoaded(get("hero_label_font"));
    ensureGoogleFontLoaded(get("panel_label_font"));
    ensureGoogleFontLoaded(get("callout_card_header_font"));
    ensureGoogleFontLoaded(get("callout_card_body_font"));

    const blur = cssValue("slide_title_blur", "0");
    const shadow = get("slide_title_shadow") || "none";
    const chipW = cssValue("slide_title_chip_width", "0");

    root.style.setProperty("--hf-slide-title-font", fontStack("slide_title_font"));
    root.style.setProperty("--hf-slide-title-font-weight", fontWeight("slide_title_font_weight"));
    root.style.setProperty("--hf-slide-title-font-size", cssValue("slide_title_font_size", "15"));
    root.style.setProperty("--hf-slide-title-text-color", get("slide_title_text_color"));
    root.style.setProperty("--hf-slide-title-bg", get("slide_title_bg"));
    root.style.setProperty("--hf-slide-title-accent", get("slide_title_accent"));
    root.style.setProperty("--hf-slide-title-chip-width", chipW);
    root.style.setProperty("--hf-slide-title-chip-display", chipW === "0px" || chipW === "0" ? "none" : "block");
    root.style.setProperty("--hf-slide-title-height", cssValue("slide_title_height", "auto"));
    root.style.setProperty("--hf-slide-title-radius", cssValue("slide_title_radius", "0"));
    root.style.setProperty("--hf-slide-title-padding", get("slide_title_padding") || "0 12px 0 10px");
    root.style.setProperty(
      "--hf-slide-title-shadow",
      shadow === "none" || !shadow ? "none" : (shadow.indexOf("drop-shadow") !== -1 ? shadow : "drop-shadow(" + shadow + ")")
    );
    root.style.setProperty("--hf-slide-title-blur", blur === "0px" || blur === "0" ? "0px" : blur);

    root.style.setProperty("--hf-caption-font", fontStack("caption_font"));
    root.style.setProperty("--hf-caption-font-weight", fontWeight("caption_font_weight"));
    root.style.setProperty("--hf-caption-font-size", cssValue("caption_font_size", "13"));
    root.style.setProperty("--hf-caption-text-color", get("caption_text_color"));
    root.style.setProperty("--hf-caption-bg", get("caption_bg"));
    root.style.setProperty("--hf-caption-radius", cssValue("caption_radius", "4"));
    root.style.setProperty("--hf-caption-padding", get("caption_padding") || "10px");
    root.style.setProperty("--hf-caption-max-width", get("caption_max_width") || "60%");
    root.style.setProperty("--hf-caption-line-height", get("caption_line_height") || "1.5");
    root.style.setProperty(
      "--hf-caption-text-shadow",
      get("caption_text_shadow") || "none"
    );

    root.style.setProperty("--hf-topic-card-font", fontStack("topic_card_font"));
    root.style.setProperty("--hf-topic-card-font-weight", fontWeight("topic_card_font_weight"));
    root.style.setProperty("--hf-topic-card-font-size", cssValue("topic_card_font_size", "44"));
    root.style.setProperty(
      "--hf-topic-card-secondary-font-size",
      cssValue("topic_card_secondary_font_size", "70")
    );
    root.style.setProperty("--hf-topic-card-text-color", get("topic_card_text_color"));
    root.style.setProperty(
      "--hf-topic-card-secondary-text-color",
      get("topic_card_secondary_text_color") || ""
    );
    root.style.setProperty("--hf-topic-card-text-align", get("topic_card_text_align") || "center");
    root.style.setProperty(
      "--hf-topic-card-text-transform",
      get("topic_card_text_transform") || "none"
    );
    root.style.setProperty("--hf-topic-card-bg", get("topic_card_bg"));
    root.style.setProperty("--hf-topic-card-border", get("topic_card_border") || "none");
    root.style.setProperty("--hf-topic-card-radius", cssValue("topic_card_radius", "22"));
    root.style.setProperty("--hf-topic-card-padding", get("topic_card_padding") || "24px 32px");
    root.style.setProperty("--hf-topic-card-max-width", get("topic_card_max_width") || "520px");
    root.style.setProperty("--hf-topic-card-shadow", get("topic_card_shadow") || "none");
    root.style.setProperty("--hf-topic-card-line-height", get("topic_card_line_height") || "1.15");

    root.style.setProperty("--hf-hero-label-font", fontStack("hero_label_font"));
    root.style.setProperty("--hf-hero-label-font-weight", fontWeight("hero_label_font_weight"));
    root.style.setProperty("--hf-hero-label-font-size", cssValue("hero_label_font_size", "26"));
    root.style.setProperty("--hf-hero-label-text-color", get("hero_label_text_color"));
    root.style.setProperty("--hf-hero-label-text-align", get("hero_label_text_align") || "center");
    root.style.setProperty("--hf-hero-label-bg", get("hero_label_bg"));
    root.style.setProperty("--hf-hero-label-radius", cssValue("hero_label_radius", "24"));
    root.style.setProperty("--hf-hero-label-padding", get("hero_label_padding") || "18px 36px");
    root.style.setProperty("--hf-hero-label-max-width", get("hero_label_max_width") || "90%");
    root.style.setProperty("--hf-hero-label-white-space", get("hero_label_white_space") || "nowrap");
    root.style.setProperty("--hf-hero-label-shadow", get("hero_label_shadow") || "none");
    root.style.setProperty("--hf-hero-label-line-height", get("hero_label_line_height") || "1.25");

    root.style.setProperty("--hf-panel-label-font", fontStack("panel_label_font"));
    root.style.setProperty("--hf-panel-label-font-weight", fontWeight("panel_label_font_weight"));
    root.style.setProperty("--hf-panel-label-font-size", cssValue("panel_label_font_size", "18"));
    root.style.setProperty("--hf-panel-label-text-color", get("panel_label_text_color"));
    root.style.setProperty("--hf-panel-label-text-align", get("panel_label_text_align") || "center");
    root.style.setProperty(
      "--hf-panel-label-text-transform",
      get("panel_label_text_transform") || "none"
    );
    root.style.setProperty("--hf-panel-label-bg", get("panel_label_bg"));
    root.style.setProperty("--hf-panel-label-radius", cssValue("panel_label_radius", "10"));
    root.style.setProperty("--hf-panel-label-padding", get("panel_label_padding") || "10px 22px");
    root.style.setProperty("--hf-panel-label-shadow", get("panel_label_shadow") || "none");
    root.style.setProperty("--hf-panel-label-white-space", get("panel_label_white_space") || "nowrap");
    root.style.setProperty(
      "--hf-panel-label-letter-spacing",
      get("panel_label_letter_spacing") || "0"
    );
    root.style.setProperty("--hf-panel-label-bottom", cssValue("panel_label_bottom", "15"));

    root.style.setProperty("--hf-callout-card-bg", get("callout_card_bg"));
    root.style.setProperty("--hf-callout-card-radius", cssValue("callout_card_radius", "30"));
    root.style.setProperty(
      "--hf-callout-card-border-gradient",
      get("callout_card_border_gradient") || "linear-gradient(90deg, #FF5C26 21%, rgba(255, 92, 38, 0.28) 73%)"
    );
    root.style.setProperty("--hf-callout-card-shadow", get("callout_card_shadow") || "none");
    root.style.setProperty("--hf-callout-card-padding", get("callout_card_padding") || "20px 24px");
    root.style.setProperty("--hf-callout-card-max-width", get("callout_card_max_width") || "42%");
    root.style.setProperty(
      "--hf-callout-card-header-font",
      fontStack("callout_card_header_font")
    );
    root.style.setProperty(
      "--hf-callout-card-header-font-weight",
      fontWeight("callout_card_header_font_weight")
    );
    root.style.setProperty(
      "--hf-callout-card-header-font-size",
      cssValue("callout_card_header_font_size", "32")
    );
    root.style.setProperty("--hf-callout-card-header-text-color", get("callout_card_header_text_color"));
    root.style.setProperty(
      "--hf-callout-card-body-font",
      fontStack("callout_card_body_font")
    );
    root.style.setProperty(
      "--hf-callout-card-body-font-weight",
      fontWeight("callout_card_body_font_weight")
    );
    root.style.setProperty(
      "--hf-callout-card-body-font-size",
      cssValue("callout_card_body_font_size", "36")
    );
    root.style.setProperty("--hf-callout-card-body-text-color", get("callout_card_body_text_color"));
    root.style.setProperty("--hf-callout-scene-overlay", get("callout_scene_overlay") || "none");

    root.style.setProperty("--hf-player-chrome-font", fontStack("player_chrome_font"));
    root.style.setProperty("--hf-player-nav-font-weight", fontWeight("player_nav_font_weight"));
    root.style.setProperty("--hf-player-nav-font-size", cssValue("player_nav_font_size", "12"));
    root.style.setProperty("--hf-player-nav-text-color", get("player_nav_text_color"));
    root.style.setProperty(
      "--hf-player-speed-title-font-weight",
      fontWeight("player_speed_title_font_weight")
    );
    root.style.setProperty(
      "--hf-player-speed-title-font-size",
      cssValue("player_speed_title_font_size", "16")
    );
    root.style.setProperty(
      "--hf-player-speed-title-text-color",
      get("player_speed_title_text_color")
    );
    root.style.setProperty(
      "--hf-player-speed-option-font-weight",
      fontWeight("player_speed_option_font_weight")
    );
    root.style.setProperty(
      "--hf-player-speed-option-font-size",
      cssValue("player_speed_option_font_size", "15")
    );
    root.style.setProperty(
      "--hf-player-speed-option-text-color",
      get("player_speed_option_text_color")
    );
  }

  function mergeConfig(obj) {
    if (!obj || typeof obj !== "object") return;
    Object.keys(obj).forEach(function (key) {
      const k = String(key).toLowerCase();
      const val = obj[key];
      if (val != null && String(val).trim() !== "") {
        values[k] = String(val).trim();
      }
    });
  }

  var stylesSchema = null;

  fetch("/api/styles-schema?v=1")
    .then(function (res) {
      if (!res.ok) throw new Error("styles schema load failed");
      return res.json();
    })
    .then(function (schema) {
      stylesSchema = schema;
      window.HFStylesSchema = schema;
      if (window.HFPlayerTheme && typeof window.HFPlayerTheme.onSchemaLoaded === "function") {
        window.HFPlayerTheme.onSchemaLoaded(schema);
      }
    })
    .catch(function () {});

  fetch("/api/player-theme?v=1")
    .then(function (res) {
      if (!res.ok) throw new Error("player theme load failed");
      return res.json();
    })
    .then(function (data) {
      mergeConfig(data);
      apply();
      if (window.HFTreatmentRotator && typeof window.HFTreatmentRotator.reloadLists === "function") {
        window.HFTreatmentRotator.reloadLists();
      }
    })
    .catch(function () {
      apply();
    });

  apply();

  window.HFPlayerTheme = {
    get: get,
    list: list,
    fontStack: fontStack,
    fontWeight: fontWeight,
    fontSizePx: fontSizePx,
    apply: apply,
    ensureGoogleFontLoaded: ensureGoogleFontLoaded,
    getAll: function () {
      return Object.assign({}, values);
    },
    getBase: function (k) {
      var key = String(k || "").toLowerCase();
      if (Object.prototype.hasOwnProperty.call(values, key) && String(values[key]).trim() !== "") {
        return String(values[key]).trim();
      }
      return Object.prototype.hasOwnProperty.call(defaults, key) ? defaults[key] : "";
    },
    getScopedOverrides: function () {
      return Object.assign({}, scopedValues);
    },
    setScopedOverrides: function (obj) {
      var next = Object.create(null);
      if (obj && typeof obj === "object") {
        Object.keys(obj).forEach(function (key) {
          var value = obj[key];
          if (value != null && String(value).trim() !== "") {
            next[String(key).toLowerCase()] = String(value).trim();
          }
        });
      }
      scopedValues = next;
      apply();
    },
    clearScopedOverrides: function () {
      scopedValues = Object.create(null);
      apply();
    },
    getDefaults: function () {
      return Object.assign({}, defaults);
    },
    getSchema: function () {
      return stylesSchema;
    },
    set: function (k, v) {
      if (k == null) return;
      var key = String(k).toLowerCase();
      var strVal = v != null ? String(v) : "";
      values[key] = strVal;
      var root = document.documentElement;
      if (root) {
        root.style.setProperty("--hf-" + key.replace(/_/g, "-"), get(key));
      }
      if (key.indexOf("font") !== -1 && key.indexOf("weight") === -1 && key.indexOf("size") === -1) {
        ensureGoogleFontLoaded(strVal);
      }
    },
    merge: mergeConfig,
  };
})();
