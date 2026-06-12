import { useEffect, useState } from "react";
import type { Deck } from "./model/deck";
import { SlideCanvas } from "./preview/SlideCanvas";
import deckJson from "../deck.json";

const deck = deckJson as Deck;

export default function App() {
  const [i, setI] = useState(0);
  const slides = deck.slides;
  const clamp = (n: number) => Math.max(0, Math.min(slides.length - 1, n));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === "PageDown") setI((p) => clamp(p + 1));
      if (e.key === "ArrowLeft" || e.key === "PageUp") setI((p) => clamp(p - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [slides.length]);

  return (
    <div className="app">
      <aside className="rail">
        {slides.map((s, idx) => (
          <button
            key={s.id}
            className={"thumb" + (idx === i ? " active" : "")}
            onClick={() => setI(idx)}
          >
            <span className="thumb-n">{idx + 1}</span>
            <span className="thumb-label">{labelOf(s)}</span>
          </button>
        ))}
      </aside>

      <main className="main">
        <SlideCanvas slide={slides[i]} footerText={deck.meta.footer} />
        <div className="hud">
          <button onClick={() => setI((p) => clamp(p - 1))} disabled={i === 0}>
            ‹
          </button>
          <span>
            {i + 1} / {slides.length} · {deck.slides[i].layout}
          </span>
          <button onClick={() => setI((p) => clamp(p + 1))} disabled={i === slides.length - 1}>
            ›
          </button>
        </div>
      </main>
    </div>
  );
}

function labelOf(s: Deck["slides"][number]): string {
  return s.title;
}
