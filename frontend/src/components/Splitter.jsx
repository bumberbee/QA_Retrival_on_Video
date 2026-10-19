import { useCallback, useRef, useState } from "react";

// A panel size in px, kept per browser like VS Code keeps its layout.
export function useStoredSize(key, initial, min, max) {
  const clamp = useCallback((value) => Math.min(max, Math.max(min, value)), [min, max]);
  const [size, setSize] = useState(() => {
    try {
      const saved = Number(localStorage.getItem(`size:${key}`));
      return saved ? clamp(saved) : initial;
    } catch {
      return initial;
    }
  });
  const update = useCallback(
    (value) => {
      const next = clamp(value);
      setSize(next);
      try {
        localStorage.setItem(`size:${key}`, String(Math.round(next)));
      } catch {
        // private mode: the size just is not remembered
      }
    },
    [key, clamp],
  );
  return [size, update, () => update(initial)];
}

// Drag handle between two panels. direction "x" resizes a width, "y" a height;
// invert = the panel grows when dragging left/up (it sits after the handle).
export default function Splitter({ direction = "x", size, onResize, onReset, invert = false, label }) {
  const start = useRef(null);
  const [dragging, setDragging] = useState(false);

  function onPointerDown(event) {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    start.current = { at: direction === "x" ? event.clientX : event.clientY, size };
    setDragging(true);
  }
  function onPointerMove(event) {
    if (!start.current) return;
    const delta = (direction === "x" ? event.clientX : event.clientY) - start.current.at;
    onResize(start.current.size + (invert ? -delta : delta));
  }
  function onPointerUp() {
    start.current = null;
    setDragging(false);
  }
  function onKeyDown(event) {
    const step = event.shiftKey ? 40 : 12;
    const keys = direction === "x" ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const grow = (event.key === keys[1]) !== invert;
    onResize(size + (grow ? step : -step));
  }

  return (
    <div
      className={`splitter splitter-${direction} ${dragging ? "dragging" : ""}`}
      role="separator"
      aria-orientation={direction === "x" ? "vertical" : "horizontal"}
      aria-label={label}
      aria-valuenow={Math.round(size)}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDoubleClick={onReset}
      onKeyDown={onKeyDown}
      title={label}
    />
  );
}
