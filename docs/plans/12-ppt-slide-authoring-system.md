# Plan: React Slide-Deck Authoring System with Live Preview + PPTX Export

## Context
Author presentation decks from a design-system spec (`ppt/design/DESIGN.md`), preview them
live in a localhost React app, edit conversationally, then export a real `.pptx` that
matches the preview. Working dir `ppt/` is greenfield (only `design/DESIGN.md` exists).
The `pptx` skill creates decks via **pptxgenjs** (JS, native editable shapes) and renders
to images via LibreOffice for QA. Both the preview and the exporter are JavaScript, so
they share one deck model, one theme, and one geometry engine.

Resolved decisions: edit loop = Claude edits JSON files + Vite HMR; export =
shared-geometry pptxgenjs (native editable shapes); scope = MVP layout families first.

## Core architecture: one source of truth, one geometry engine

```
design/DESIGN.md ──(compile, Claude one-time)──► theme.json   (machine tokens)
                                                      │
deck.json (content SSOT) ──────────────┬─────────────┘
                                        ▼
                          src/layout/resolve.ts  (pure TS)
                          slide+theme ► [positioned elements in INCHES]
                                        │
                 ┌──────────────────────┴───────────────────────┐
                 ▼                                               ▼
   React preview (CSS, inches×96 px)              export-pptx (pptxgenjs, inches)
   localhost, HMR, scale-to-fit                   deck.pptx (editable shapes)
```

**Fidelity guarantee:** geometry is computed once, in inches, by a framework-agnostic
resolver. The preview converts inches→px (1 in = 96 px, canvas 1280×720 = 13.333×7.5 in,
16:9). The exporter passes inches straight to pptxgenjs. Neither renderer invents layout;
both consume the same resolved element list. This is what makes "preview matches export"
true by construction.

## Data model (the two JSON files = SSOT)

`ppt/src/model/deck.ts` (types) + `ppt/deck.json` (data):
```ts
type Deck = { meta: { title: string; footer: string }; slides: Slide[] };
type Slide =
  | { id; layout: "cover";       kicker?; title; citation?; authors?: string[]; seal?: string }
  | { id; layout: "body";        kicker?; title; bullets: Bullet[]; note?: string }
  | { id; layout: "comparison";  kicker?; title; cards: Card[]; note?: string }   // 2-3 cards
  | { id; layout: "table";       kicker?; title; verdict?; columns: string[]; rows: Row[]; highlightRow?: number }
  | { id; layout: "closing";     title; subtitle? };
type Bullet = { text: string; level: 0|1; emphasis?: "green"|"red"|"bold" };
type Card   = { header: string; bullets: Bullet[] };
```
Edits are applied to **this data**, never to rendered DOM.

`ppt/theme.json` (compiled from DESIGN.md; types in `ppt/src/theme/theme.ts`):
```jsonc
{
  "canvas": { "w": 13.333, "h": 7.5 },
  "colors": { "bg":"#F7F9FC","textPrimary":"#1F2A44","textSecondary":"#5B6677",
              "kicker":"#9AA3AF","brand":"#2B4FA0","brandBright":"#3B6FD4",
              "green":"#1F9D55","red":"#D64545","amber":"#F2A93B","amberFill":"#FFE08A",
              "cardHeader":"#2C3E5C","panel":"#EAF0FB","border":"#D9DEE6" },
  "fonts": { "title":"Poppins", "body":"Inter" },          // DESIGN.md: pick ONE title font
  "type":  { "kicker":15, "title":34, "body":17, "caption":10.5 },  // pt
  "margin": { "x":0.9, "top":0.7, "bottom":0.5 },          // inches
  "layout": { "cardGap":0.3, "cardRadius":0.12, "footerH":0.4 }
}
```

## DESIGN.md → theme.json compile step
`design/DESIGN.md` is human prose (approximate hexes, rules, anti-patterns). It is NOT
parsed at runtime. Instead Claude compiles it once into `theme.json` (a deterministic
token map), pulling the Color System, Typography, Grid/Spacing sections. Re-run the
compile whenever DESIGN.md changes. Encode key DESIGN.md rules as theme constraints:
one title font, kicker gray, footer centered every slide, no accent line under titles
(also a pptx-skill rule), semantic green/red.

## Shared layout resolver (the critical file)
`ppt/src/layout/resolve.ts` — pure TS, no DOM and no Node APIs, importable by both Vite
and the export script. Exposes `resolveSlide(slide, theme): Element[]` where
```ts
type Element =
  | { kind:"text"; x;y;w;h; text; font; size; color; bold?; align?; valign? }
  | { kind:"rect"; x;y;w;h; fill?; line?; radius? }
  | { kind:"image"; x;y;w;h; path }
  | { kind:"table"; x;y;w;h; columns; rows; headerFill; highlightRow? }
  | { kind:"line"; x;y;w;h; color }   // (avoid title underlines per DESIGN.md)
```
Per-family resolvers in `ppt/src/layout/families/{cover,body,comparison,table,closing}.ts`.
Each lays out header zone (kicker+title top-left), body zone, and the centered footer
band, honoring `theme.margin`. All coordinates in inches.

## React preview app
- Stack: Vite + React + TypeScript. `npm create vite@latest . -- --template react-ts`.
- Files: `ppt/index.html`, `ppt/src/main.tsx`, `ppt/src/App.tsx`,
  `ppt/src/preview/SlideCanvas.tsx`, `ppt/src/preview/Element.tsx`.
- `App.tsx`: one route; loads `deck.json` + `theme.json`; slide nav (←/→ keys + thumbnail
  rail); renders current slide.
- `SlideCanvas.tsx`: fixed 1280×720 stage; `transform: scale()` to fit viewport
  (accurate-to-print scaling); paints `theme.colors.bg`.
- `Element.tsx`: maps each resolved `Element` to an absolutely-positioned div
  (`left/top/width/height = inches×96 px`); text/rect/image/table/line renderers.
- Web fonts (Poppins/Inter) loaded so preview type matches.

## Edit loop (Claude edits JSON + HMR)
1. User requests a change in chat ("make slide 3 a comparison with two cards", "swap
   accent to brandBright on the cover").
2. Claude edits `deck.json` (content) or `theme.json` (style) directly.
3. Vite HMR re-renders the localhost preview instantly (Vite watches JSON imports; if
   needed, import JSON through a tiny module so HMR fires). No full reload, no DOM edits.

## PPTX export (shared-geometry pptxgenjs)
- `ppt/export/export-pptx.mts`, run with `tsx` (`npm run export`).
- Imports `deck.json`, `theme.json`, and the SAME `resolveSlide`. For each slide:
  `pptx.defineLayout` 13.333×7.5; add one `addSlide`; map each resolved `Element` →
  `slide.addText / addShape(rect) / addImage / addTable / addShape(line)` using its
  inch coords directly. Fills/fonts/sizes pulled from the element (already theme-derived).
- Output `ppt/out/deck.pptx`. Font faces set to `theme.fonts`; note fonts must be
  installed for faithful LibreOffice render (fallback documented).
- QA via pptx skill: `soffice --convert-to pdf` then `pdftoppm -jpeg -r 150`, then a
  visual subagent compares slide images against preview screenshots (the skill's QA loop).

## Critical files to create
- `ppt/package.json`, `ppt/vite.config.ts`, `ppt/tsconfig.json`, `ppt/index.html`
- `ppt/deck.json`, `ppt/theme.json`
- `ppt/src/model/deck.ts`, `ppt/src/theme/theme.ts`
- `ppt/src/layout/resolve.ts` + `ppt/src/layout/families/*.ts`  ← geometry SSOT
- `ppt/src/preview/{SlideCanvas,Element}.tsx`, `ppt/src/App.tsx`, `ppt/src/main.tsx`
- `ppt/export/export-pptx.mts`
- `ppt/package.json` scripts: `dev` (vite), `export` (tsx export/export-pptx.mts)

## MVP layout families (this build)
cover, body (kicker+title+bullets+optional note band), comparison (2-3 navy-header cards),
table (navy header row, color-coded cells, highlighted reference row), closing. Defer:
section-divider, insight, chart, process/timeline (add as new family resolvers later;
architecture already supports them).

## Verification (end to end)
1. `npm install && npm run dev` → open localhost; confirm 16:9 canvas, scale-to-fit,
   ←/→ navigation, KAIST colors/fonts from `theme.json`, centered footer on every slide.
2. Build a 5-slide `deck.json` (one per MVP family) seeded from the real KAIST deck
   content; visually compare to `design/DESIGN.md` rules (kicker top-left, navy titles,
   semantic green/red, no title underline).
3. Edit-loop test: change a bullet and an accent color in JSON → preview HMR-updates with
   no reload.
4. `npm run export` → `out/deck.pptx`. Open + run pptx-skill QA: convert to images,
   subagent visual inspection; diff against preview screenshots. Confirm geometry, colors,
   table, and footer match. Fix resolver (single source) if drift appears, re-export.
5. Confirm shapes are native/editable in the pptx (text boxes, not flat images).
