// overlay-timing.js
// Resolve trigger_phrase excerpts to seconds using TTS word marks.
window.HFOverlayTiming = (function () {
  function normalizeToken(s) {
    return String(s || '')
      .toLowerCase()
      .replace(/&/g, 'and')
      .replace(/[^a-z0-9']/g, '');
  }

  function tokenizeVoiceover(text) {
    const rawTokens = String(text || '').trim().split(/\s+/).filter(Boolean);
    if (!rawTokens.length) return [];
    const tokens = [];
    rawTokens.forEach(function (token) {
      if (/[\-\u2014\u2013\/]/.test(token)) {
        token.split(/([\-\u2014\u2013\/]+)/).forEach(function (sub) {
          if (!sub) return;
          if (/^[\-\u2014\u2013\/]+$/.test(sub)) {
            if (tokens.length) tokens[tokens.length - 1] += sub;
            else tokens.push(sub);
          } else {
            tokens.push(sub);
          }
        });
      } else {
        tokens.push(token);
      }
    });
    return tokens;
  }

  function tokenizePhrase(phrase) {
    return tokenizeVoiceover(phrase);
  }

  function alignMarksWithVoiceover(marks, voiceover) {
    if (!marks || !marks.length) return [];
    if (!voiceover) return marks.slice();
    const origTokens = tokenizeVoiceover(voiceover);
    if (!origTokens.length) return marks.slice();

    const origNormalized = origTokens.map(normalizeToken);
    const spokenNormalized = marks.map(function (m) {
      return normalizeToken(m && m.text);
    });

    let origIdx = 0;
    const aligned = [];
    for (let i = 0; i < marks.length; i++) {
      const spokenNorm = spokenNormalized[i];
      if (!spokenNorm) {
        aligned.push(marks[i]);
        continue;
      }
      let foundIdx = -1;
      for (let j = 0; j < 5; j++) {
        const checkIdx = origIdx + j;
        if (checkIdx >= origTokens.length) break;
        const origNorm = origNormalized[checkIdx];
        if (origNorm && origNorm === spokenNorm) {
          foundIdx = checkIdx;
          break;
        }
      }
      if (foundIdx !== -1) {
        aligned.push(Object.assign({}, marks[i], { text: origTokens[foundIdx] }));
        origIdx = foundIdx + 1;
      } else {
        aligned.push(marks[i]);
      }
    }
    return aligned;
  }

  function findPhraseStartSeconds(phrase, marks, voiceover) {
    const phraseTokens = tokenizePhrase(phrase);
    if (!phraseTokens.length) return null;

    const aligned = alignMarksWithVoiceover(marks, voiceover);
    const wordTokens = aligned
      .map(function (m) {
        return {
          text: String(m && m.text || '').trim(),
          offset: Number(m && m.offset) || 0,
        };
      })
      .filter(function (m) { return m.text; });

    if (!wordTokens.length) return null;

    const phraseNorm = phraseTokens.map(normalizeToken);
    const n = phraseNorm.length;

    for (let i = 0; i <= wordTokens.length - n; i++) {
      let matched = true;
      for (let j = 0; j < n; j++) {
        if (normalizeToken(wordTokens[i + j].text) !== phraseNorm[j]) {
          matched = false;
          break;
        }
      }
      if (matched) return wordTokens[i].offset;
    }
    return null;
  }

  function estimateFromTextPosition(phrase, voiceover, duration) {
    const vo = String(voiceover || '').trim();
    const needle = String(phrase || '').trim();
    if (!vo || !needle || !(duration > 0)) return null;
    const lowerVo = vo.toLowerCase();
    const lowerNeedle = needle.toLowerCase();
    const idx = lowerVo.indexOf(lowerNeedle);
    if (idx < 0) return null;
    const ratio = idx / Math.max(1, lowerVo.length);
    return Math.max(0, ratio * duration);
  }

  function resolveSeconds(phrase, opts, fallbackSeconds) {
    const fallback = Number.isFinite(fallbackSeconds) ? fallbackSeconds : 0.15;
    const raw = String(phrase || '').trim();
    if (!raw || raw.toUpperCase() === 'N/A') return Math.max(0, fallback);

    const marks = opts && opts.cueWords;
    const voiceover = (opts && opts.voiceover) || '';
    const duration = opts && opts.duration;
    const lead = opts && opts.cueLead ? Math.max(0, Number(opts.cueLead) || 0) : 0;

    let seconds = findPhraseStartSeconds(raw, marks, voiceover);
    if (seconds == null) {
      seconds = estimateFromTextPosition(raw, voiceover, duration);
    }
    if (seconds == null) return Math.max(0, fallback);
    return Math.max(0, seconds - lead);
  }

  return {
    resolveSeconds: resolveSeconds,
    findPhraseStartSeconds: findPhraseStartSeconds,
  };
})();
