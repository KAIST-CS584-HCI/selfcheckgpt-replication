// Renders one resolved Element as absolutely-positioned CSS. Inches -> px at 96/in,
// points -> px at 96/72. This is the ONLY place geometry becomes pixels in the preview.

import type { CSSProperties } from "react";
import type { Element, Para, VAlign } from "../layout/element";
import { PX_PER_IN } from "../theme/theme";

const PT_PX = 96 / 72;
const inPx = (v: number) => v * PX_PER_IN;

const justify: Record<VAlign, CSSProperties["justifyContent"]> = {
  top: "flex-start",
  middle: "center",
  bottom: "flex-end",
};

function box(e: { x: number; y: number; w: number; h: number }): CSSProperties {
  return {
    position: "absolute",
    left: inPx(e.x),
    top: inPx(e.y),
    width: inPx(e.w),
    height: inPx(e.h),
  };
}

function Paragraph({
  p,
  defFont,
  defSize,
  defColor,
  lineHeightPt,
}: {
  p: Para;
  defFont: string;
  defSize: number;
  defColor: string;
  lineHeightPt?: number;
}) {
  const size = (p.size ?? defSize) * PT_PX;
  const style: CSSProperties = {
    margin: 0,
    marginBottom: p.spaceAfterPt ? p.spaceAfterPt * PT_PX : 0,
    fontFamily: `'${p.font ?? defFont}', sans-serif`,
    fontSize: size,
    lineHeight: lineHeightPt ? `${lineHeightPt * PT_PX}px` : 1.3,
    color: p.color ?? defColor,
    fontWeight: p.bold ? 700 : 400,
    textAlign: p.align ?? "left",
    paddingLeft: p.bullet ? 16 + (p.indentLevel ?? 0) * 18 : 0,
    textIndent: p.bullet ? -16 : 0,
  };
  const runs = p.runs.map((r, i) => (
    <span
      key={i}
      style={{
        fontWeight: r.bold ? 700 : undefined,
        fontStyle: r.italic ? "italic" : undefined,
        color: r.color,
      }}
    >
      {r.text}
    </span>
  ));
  return (
    <p style={style}>
      {p.bullet ? <span style={{ color: defColor }}>•&nbsp;</span> : null}
      {runs}
    </p>
  );
}

export function ElementView({ e }: { e: Element }) {
  if (e.kind === "rect") {
    return (
      <div
        style={{
          ...box(e),
          background: e.fill ?? "transparent",
          border: e.line ? `${(e.lineWidthPt ?? 1) * PT_PX}px solid ${e.line}` : undefined,
          borderRadius: e.radius ? inPx(e.radius) : undefined,
          boxSizing: "border-box",
        }}
      />
    );
  }

  if (e.kind === "image") {
    return <img src={e.path} style={{ ...box(e), objectFit: "contain" }} alt="" />;
  }

  if (e.kind === "table") {
    const fs = e.size * PT_PX;
    return (
      <table
        style={{
          ...box(e),
          borderCollapse: "collapse",
          fontFamily: `'${e.font}', sans-serif`,
          fontSize: fs,
          tableLayout: "fixed",
        }}
      >
        <thead>
          <tr>
            {e.columns.map((c, i) => (
              <th
                key={i}
                style={{
                  background: e.headerFill,
                  color: e.headerColor,
                  textAlign: i === 0 ? "left" : "center",
                  padding: "6px 8px",
                  fontWeight: 600,
                  border: `1px solid ${e.borderColor}`,
                }}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {e.rows.map((row, ri) => (
            <tr key={ri} style={{ background: ri === e.highlightRow ? e.highlightFill : undefined }}>
              {row.map((cell, ci) => (
                <td
                  key={ci}
                  style={{
                    color: cell.color ?? e.textColor,
                    fontWeight: cell.color ? 700 : ci === 0 ? 600 : 400,
                    fontStyle: ri === e.highlightRow ? "italic" : undefined,
                    textAlign: ci === 0 ? "left" : "center",
                    padding: "5px 8px",
                    border: `1px solid ${e.borderColor}`,
                  }}
                >
                  {cell.text}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  // text
  return (
    <div
      style={{
        ...box(e),
        display: "flex",
        flexDirection: "column",
        justifyContent: justify[e.valign ?? "top"],
        overflow: "hidden",
      }}
    >
      {e.paragraphs.map((p, i) => (
        <Paragraph
          key={i}
          p={p}
          defFont={e.font}
          defSize={e.size}
          defColor={e.color}
          lineHeightPt={e.lineHeightPt}
        />
      ))}
    </div>
  );
}
