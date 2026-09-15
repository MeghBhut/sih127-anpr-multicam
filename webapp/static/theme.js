/* Cyanotype theme toggle — vanilla, drop-in.
   The <head> snippet handles the pre-paint read; this does click-to-toggle,
   persistence and the icon swap. */

(function () {
  var STORAGE_KEY = 'cyanotype-theme';

  var SUN = 'M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10Zm0-5v3m0 14v3M2 12h3m14 0h3' +
            'M4.2 4.2l2.1 2.1m11.4 11.4l2.1 2.1M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1';
  var MOON = 'M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36A5.39 5.39 0 0 1 12 3Z';

  function getStoredTheme() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return null; }
  }

  function setStoredTheme(theme) {
    try { localStorage.setItem(STORAGE_KEY, theme); } catch (e) { /* ignore */ }
  }

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark'
      ? 'dark' : 'light';
  }

  function applyTheme(theme) {
    if (theme === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    updateToggleIcons(theme);
  }

  function updateToggleIcons(theme) {
    var icon = document.getElementById('themeIcon');
    if (icon) {
      icon.setAttribute('d', theme === 'dark' ? SUN : MOON);
      // the sun is strokes, the moon is a filled shape
      icon.setAttribute('fill', theme === 'dark' ? 'none' : 'currentColor');
      icon.setAttribute('stroke', theme === 'dark' ? 'currentColor' : 'none');
      icon.setAttribute('stroke-width', '2');
      icon.setAttribute('stroke-linecap', 'round');
    }
    document.querySelectorAll('[data-theme-toggle]').forEach(function (btn) {
      btn.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
      btn.setAttribute('aria-label',
        theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
    });
  }

  function toggleTheme() {
    var next = currentTheme() === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    setStoredTheme(next);
    // the editor paints from CSS tokens, so it has to be redrawn
    if (window.ANPR && window.ANPR.redraw) window.ANPR.redraw();
  }

  document.addEventListener('DOMContentLoaded', function () {
    updateToggleIcons(currentTheme());
    document.querySelectorAll('[data-theme-toggle]').forEach(function (btn) {
      btn.addEventListener('click', toggleTheme);
    });
  });

  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)')
      .addEventListener('change', function (e) {
        if (!getStoredTheme()) applyTheme(e.matches ? 'dark' : 'light');
      });
  }
})();
