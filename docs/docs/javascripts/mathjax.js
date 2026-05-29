// MathJax configuration for pymdownx.arithmatex (generic mode).
// Inline math is delimited by \( ... \) and display math by \[ ... \],
// which arithmatex emits; we also accept $...$ / $$...$$ for convenience.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Re-typeset on instant navigation (Material's SPA-style page loads).
document$.subscribe(() => {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});
