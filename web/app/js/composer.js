/* The message box: an autosizing textarea, Send, and Stop while a response streams.
 *
 * Keys: Enter sends, Shift+Enter adds a new line, Ctrl/Cmd+Enter sends. (Esc stops; that lives in
 * layout.js because it works from anywhere on the page.) While an IME composition is open
 * (Korean, Japanese, Chinese input) Enter belongs to the IME: `isComposing`, or keyCode 229 on
 * engines that report the keydown before compositionend, never sends.
 */

const MAX_HEIGHT = 240;

export function isImeEvent(event) {
  return Boolean(event.isComposing) || event.keyCode === 229;
}

export function initComposer({ form, input, sendButton, stopButton, onSend, onStop }) {
  let streaming = false;

  function resize() {
    input.style.height = 'auto';
    const wanted = input.scrollHeight;
    input.style.height = `${Math.min(wanted, MAX_HEIGHT)}px`;
    input.style.overflowY = wanted > MAX_HEIGHT ? 'auto' : 'hidden';
  }

  function refresh() {
    sendButton.disabled = streaming || input.value.trim() === '';
  }

  function submit() {
    if (streaming || input.value.trim() === '') return;
    onSend(input.value);
  }

  input.addEventListener('input', () => {
    resize();
    refresh();
  });

  input.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' || isImeEvent(event)) return;
    const withCommand = event.ctrlKey || event.metaKey;
    if (event.shiftKey && !withCommand) return; // a new line
    if (event.altKey) return;
    event.preventDefault();
    submit();
  });

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    submit();
  });

  stopButton.addEventListener('click', () => onStop());
  window.addEventListener('resize', resize);

  return {
    setStreaming(on) {
      const stopHadFocus = document.activeElement === stopButton;
      streaming = Boolean(on);
      sendButton.hidden = streaming;
      stopButton.hidden = !streaming;
      refresh();
      if (!streaming && stopHadFocus) input.focus();
    },
    /* Put `text` in the box (a suggested prompt) without sending it. */
    fill(text) {
      input.value = text;
      resize();
      refresh();
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
    },
    clear() {
      input.value = '';
      resize();
      refresh();
    },
    focus() {
      input.focus();
    },
    get value() {
      return input.value;
    },
  };
}
