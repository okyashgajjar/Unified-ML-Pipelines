---
name: Analytical Logic
colors:
  surface: '#0b1326'
  surface-dim: '#0b1326'
  surface-bright: '#31394d'
  surface-container-lowest: '#060e20'
  surface-container-low: '#131b2e'
  surface-container: '#171f33'
  surface-container-high: '#222a3d'
  surface-container-highest: '#2d3449'
  on-surface: '#dae2fd'
  on-surface-variant: '#c2c6d6'
  inverse-surface: '#dae2fd'
  inverse-on-surface: '#283044'
  outline: '#8c909f'
  outline-variant: '#424754'
  surface-tint: '#adc6ff'
  primary: '#adc6ff'
  on-primary: '#002e6a'
  primary-container: '#4d8eff'
  on-primary-container: '#00285d'
  inverse-primary: '#005ac2'
  secondary: '#4fdbc8'
  on-secondary: '#003731'
  secondary-container: '#04b4a2'
  on-secondary-container: '#003f38'
  tertiary: '#ffb95f'
  on-tertiary: '#472a00'
  tertiary-container: '#ca8100'
  on-tertiary-container: '#3e2400'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#adc6ff'
  on-primary-fixed: '#001a42'
  on-primary-fixed-variant: '#004395'
  secondary-fixed: '#71f8e4'
  secondary-fixed-dim: '#4fdbc8'
  on-secondary-fixed: '#00201c'
  on-secondary-fixed-variant: '#005048'
  tertiary-fixed: '#ffddb8'
  tertiary-fixed-dim: '#ffb95f'
  on-tertiary-fixed: '#2a1700'
  on-tertiary-fixed-variant: '#653e00'
  background: '#0b1326'
  on-background: '#dae2fd'
  surface-variant: '#2d3449'
typography:
  display:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '700'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  h1:
    fontFamily: Inter
    fontSize: 30px
    fontWeight: '600'
    lineHeight: '1.3'
  h2:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.4'
  h3:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: '1.4'
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
  label-caps:
    fontFamily: Space Grotesk
    fontSize: 12px
    fontWeight: '600'
    lineHeight: '1'
    letterSpacing: 0.05em
  mono-data:
    fontFamily: Space Grotesk
    fontSize: 14px
    fontWeight: '500'
    lineHeight: '1.5'
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  gutter: 24px
  margin: 40px
---

## Brand & Style

This design system is engineered for high-density data environments, prioritizing cognitive clarity and professional rigor. The brand personality is authoritative yet accessible, designed to evoke a sense of precision, intelligence, and stability. 

The aesthetic follows a **Modern Corporate** style with a focus on functional minimalism. It avoids decorative distractions, utilizing a dark-mode-first approach to reduce eye strain during prolonged sessions of data analysis. Depth is established through subtle layering and structural borders rather than expressive gradients, ensuring that the machine learning insights remains the focal point of the experience.

## Colors

The palette is anchored in deep slate and charcoal tones to provide a sophisticated, low-fatigue backdrop. 
- **Primary (Vibrant Blue):** Reserved for high-priority actions, primary buttons, and active states. It signals intent and navigation.
- **Secondary (Teal):** Used for data visualization highlights, success states, and secondary telemetry. It provides a technical, "fluorescent" contrast against the dark background.
- **Surface Tones:** A layered approach using Slate 900 for the base and Slate 800 for cards and containers, ensuring clear separation of information modules.
- **Data Accents:** Tertiary amber is used sparingly for warnings or model anomalies, ensuring high visibility without overwhelming the primary teal/blue narrative.

## Typography

This design system utilizes **Inter** as the primary workhorse for its exceptional legibility and neutral character in technical interfaces. **Space Grotesk** is introduced as a secondary font for labels, data points, and metadata to provide a subtle "technical/scientific" edge without sacrificing readability.

Hierarchy is strictly enforced through weight and scale. Headlines use tighter letter spacing for a compact, professional look, while labels utilize uppercase styling and increased tracking to differentiate themselves from narrative body text.

## Layout & Spacing

The layout utilizes a **12-column fluid grid** system designed for high-resolution dashboard monitors. A base 4px spacing unit (the "qubit") ensures mathematical consistency across all margins and paddings.

- **Dashboard Layout:** Standard 240px fixed left navigation with a fluid content area.
- **Module Spacing:** Cards and widgets are separated by a 24px (lg) gutter to allow data-heavy visualizations room to "breathe."
- **Internal Padding:** Dashboard cards utilize 24px padding for headlines and 16px for internal content to maintain a tight, information-dense profile without feeling cluttered.

## Elevation & Depth

Depth is communicated through **Tonal Layering** and **Subtle Outlines** rather than heavy shadows, maintaining a clean, modern aesthetic.

1.  **Level 0 (Base):** Deep Slate/Charcoal (#0F172A). Used for the global background.
2.  **Level 1 (Surface):** Slate 800 (#1E293B). Used for primary dashboard widgets and cards.
3.  **Borders:** All surface elements utilize a 1px solid border (#334155). This provides crisp definition between modules.
4.  **Shadows:** When an element is focused or "picked up," a subtle ambient shadow (0px 4px 20px rgba(0,0,0,0.4)) is applied to provide a lift effect. 
5.  **Interactive States:** Primary buttons and active navigational elements use a soft outer glow in the primary blue color (30% opacity) to signify focus.

## Shapes

The design system adopts a **Soft** shape language. This provides a professional balance between the starkness of sharp corners and the playfulness of fully rounded UI.

- **Standard Elements:** 0.25rem (4px) radius for inputs, small buttons, and chips.
- **Containers:** 0.5rem (8px) radius for dashboard cards and modal windows.
- **Selection Indicators:** Vertical bars used in navigation utilize a subtle rounding on the inner corners to soften the "active" indicator.

## Components

### Buttons & Controls
- **Primary Button:** Solid vibrant blue with white text. High contrast, 4px corner radius.
- **Ghost/Outline Button:** Subtle slate border with teal text for secondary actions.
- **Input Fields:** Darker slate background with a 1px border that transitions to blue on focus. Label text always uses the `label-caps` typography style.

### Data Visualization Components
- **Dashboard Cards:** Slate 800 background, 1px border, 8px radius. Titles are set in `h3` Inter.
- **Status Chips:** Small, condensed pills using low-opacity backgrounds (e.g., 10% Teal) with high-saturation text for status indicators (e.g., "Model Training").
- **Data Tables:** Borderless rows with 1px slate horizontal dividers. Numeric data utilizes the `mono-data` font for tabular alignment.

### ML Specific Components
- **Code Snippets:** Deep charcoal blocks with syntax highlighting consistent with the primary/secondary palette.
- **Metric Tiles:** Large-format typography for KPIs (e.g., Accuracy, Loss) using Space Grotesk for the numerical values to emphasize the technical nature of the data.
- **Progress Steppers:** Vertical, thin lines using teal for completed states and slate for pending steps.