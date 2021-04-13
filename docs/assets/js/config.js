window.addEventListener('load', function () {
  var p = localStorage.getItem("data-md-color-primary");
  if (p) {
    document.body.setAttribute('data-md-color-primary', p);
  }
  var a = localStorage.getItem('data-md-color-accent');
  if (a) {
    document.body.setAttribute('data-md-color-accent', a);
  }
  var s = localStorage.getItem('data-md-color-scheme');
  if (s) {
    document.body.setAttribute('data-md-color-scheme', s)
  }
}, false);

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {
      '[+]': ['ams', 'boldsymbol']
    },
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  loader: {
    load: ['[tex]/ams', '[tex]/boldsymbol']
  }
};

hljs.highlightAll()
