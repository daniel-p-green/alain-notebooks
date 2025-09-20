# ALAIN Brand Tokens (Tailwind + CSS)

This repo uses a minimal, readable set of brand tokens mapped to Tailwind and CSS variables. Keep usage consistent for contrast and accessibility.

## Tailwind utilities
- Colors: `alain-blue` `alain-yellow` `alain-stroke` `alain-navy` `alain-card` `alain-bg` `alain-text`
- Dark (optional): `alain-dark-bg` `alain-dark-low`
- Radius: `rounded-alain-lg` (12px)
- Shadow: `shadow-alain-sm` (0 1px 2px rgba(15,23,42,0.06))

Example:

```tsx
<button className="rounded-alain-lg bg-alain-blue text-white shadow-alain-sm">Primary</button>
```

## CSS variables
Defined in `app/globals.css` for global usage and non-Tailwind surfaces.

```css
:root {
  --alain-blue: #0058A3;
  --alain-yellow: #FFDA1A;
  --alain-stroke: #004580;
  --alain-navy: #1E3A8A;
  --alain-card: #F7F7F6;
  --alain-bg: #FFFFFF;
  --alain-text: #111827;
  --alain-dark-bg: #0B1220;
  --alain-dark-low: #1E3A8A;
}
```

Fonts are loaded via `next/font` in `app/layout.tsx`. Headings use Montserrat; body uses Inter.

## Accessibility
- Maintain 4.5:1 contrast for body text.
- Use `#FFDA1A` (yellow) as accent/badges only; avoid long body text on yellow.
- Blue on white and white on blue pass AA/AAA.

