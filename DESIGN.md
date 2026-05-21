---
name: VaccineNLP Design System
version: 1.0.0
tokens:
  colors:
    dark:
      background: "#0a192f"
      panel: "#172a45"
      primary: "#ccd6f6"
      secondary: "#8892b0"
      accent: "#64ffda"
      accent-hover: "#41d6b4"
    light:
      background: "#f8f9fa"
      panel: "#ffffff"
      primary: "#212529"
      secondary: "#6c757d"
      accent: "#007bff"
      accent-hover: "#0056b3"
  typography:
    fonts:
      sans: "Inter, system-ui, -apple-system, sans-serif"
    sizes:
      h1: "2.5rem"
      h2: "1.75rem"
      body: "1rem"
      caption: "0.85rem"
  spacing:
    xs: "4px"
    sm: "8px"
    md: "16px"
    lg: "24px"
    xl: "32px"
---

# VaccineNLP Design System

## Overview
This design system provides a premium, responsive, and highly accessible user interface for the VaccineNLP-Thesis project. It defines cohesive color schemes for both dark and light modes, modern typography, spacing constants, and specific UI component behaviors to ensure a seamless experience across all applications.

## Colors

### Dark Mode (Default)
Our primary interface is a high-tech dark slate mode, providing maximum focus for data visualization and XAI model predictions.
- **Background**: `#0a192f` (Deep obsidian blue)
- **Panel/Card**: `#172a45` (Navy slate for content boxes)
- **Primary Text**: `#ccd6f6` (Bright silver for readability)
- **Secondary Text**: `#8892b0` (Muted grey for descriptions and metadata)
- **Accent**: `#64ffda` (Teal/Emerald green for active states, key buttons, and highlights)

### Light Mode
A secondary clean light mode for traditional academic reporting.
- **Background**: `#f8f9fa` (Crisp light grey)
- **Panel/Card**: `#ffffff` (Pure white cards)
- **Primary Text**: `#212529` (Dark charcoal)
- **Secondary Text**: `#6c757d` (Muted slate)
- **Accent**: `#007bff` (Vibrant blue for primary controls)

## Typography
We use **Inter** as our primary font family, loaded directly from Google Fonts, falling back to clean system sans-serif font families.
- **Headers (H1)**: 2.5rem, Semibold / Bold
- **Section Headers (H2)**: 1.75rem, Semibold
- **Body Text**: 1.0rem, Regular
- **Captions & Small Text**: 0.85rem, Muted

## Layout
The responsive grid is built on a standard 12-column layout:
- **Max Width**: 1200px
- **Breakpoints**:
  - Mobile: `< 768px`
  - Tablet: `768px - 1024px`
  - Desktop: `> 1024px`
- **Gutter**: 1.0rem (`16px`)

## Elevation & Shadows
- **Card Shadow**: `0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)`
- **Hover Shadow**: `0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)`

## Shapes
We use modern, smooth, rounded shapes to make elements feel premium and friendly:
- **Button Border Radius**: 8px (`0.5rem`)
- **Card Border Radius**: 12px (`0.75rem`)
- **Visual Controls**: 6px (`0.375rem`)

## Components

### 1. Interactive Buttons
Buttons must transition smoothly on hover and focus states with micro-animations:
- **Primary Action**: Employs the `accent` color as background with custom hover scaling (1.02x).
- **Secondary Action**: Muted outline buttons that light up on hover.

### 2. Tabs
Clean underline tabs rather than blocky borders:
- Active state uses `accent` color underline with a transition timing of `0.2s ease`.

## Do's and Don'ts

### Do
- Use HSL-curated Hues to ensure harmonic UI components.
- Ensure text elements have at least WCAG AA contrast ratios (4.5:1).
- Utilize Streamlit custom containers and toast elements for subtle notifications.

### Don't
- Use browser-default plain red/blue/green colors.
- Mix dark-mode grey text directly on dark-blue backgrounds without checking contrast.
- Use heavy, blocky borders that degrade the premium feeling.
