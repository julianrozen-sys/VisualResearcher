// Copy-timestamp buttons (CLAUDE.md §15). Delegated from the document so it
// keeps working for cards HTMX swaps in after page load.
document.addEventListener('click', function (event) {
  var button = event.target.closest('.copy-timestamp');
  if (!button) return;
  event.preventDefault();
  var value = button.getAttribute('data-copy') || '';
  var restore = button.textContent;
  function done(ok) {
    button.textContent = ok ? 'copied' : 'copy failed';
    setTimeout(function () { button.textContent = restore; }, 1200);
  }
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(value).then(function () { done(true); }, function () { done(false); });
  } else {
    // 127.0.0.1 over plain http is a secure context in practice, but older
    // browsers disagree; fall back rather than silently doing nothing.
    var field = document.createElement('textarea');
    field.value = value;
    document.body.appendChild(field);
    field.select();
    try { done(document.execCommand('copy')); } catch (e) { done(false); }
    document.body.removeChild(field);
  }
});
