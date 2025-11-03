# Constraint Ratio Experiment Visualization Plan

## Overview
Visualize results from 8 constraint ratio experiments testing the effect of increasing constrained manifold sampling on different encoder-generator combinations.

## Experimental Setup
- **X-axis**: Constraint ratio (0.0 → 1.0, every 0.1 increment) - proportion of negatives sampled from constrained manifold
- **Y-axis**: Target metrics (linear performance, permutation performance, angle error, loss)
- **Grouping**: By generative process ($g$) with comparison of encoder types
- **Two lines per subplot**: MLP encoder vs Corresponding Inverse Process encoder

## Visualization Structure

### Main Body Figures

#### Figure 1: Linear Performance - 2×2 Grid
```
┌─────────────┬─────────────┐
│   Identity  │   Linear    │
│     (g)     │     (g)     │
├─────────────┼─────────────┤
│   Spiral    │   Patches   │
│     (g)     │     (g)     │
└─────────────┴─────────────┘
```

#### Figure 2: Permutation Performance - 2×2 Grid
```
┌─────────────┬─────────────┐
│   Identity  │   Linear    │
│     (g)     │     (g)     │
├─────────────┼─────────────┤
│   Spiral    │   Patches   │
│     (g)     │     (g)     │
└─────────────┴─────────────┘
```

#### Figure 3: Angle Error - 2×2 Grid
```
┌─────────────┬─────────────┐
│   Identity  │   Linear    │
│     (g)     │     (g)     │
├─────────────┼─────────────┤
│   Spiral    │   Patches   │
│     (g)     │     (g)     │
└─────────────┴─────────────┘
```

#### Figure 4: Loss - 2×2 Grid
```
┌─────────────┬─────────────┐
│   Identity  │   Linear    │
│     (g)     │     (g)     │
├─────────────┼─────────────┤
│   Spiral    │   Patches   │
│     (g)     │     (g)     │
└─────────────┴─────────────┘
```

### Within Each Subplot
- **Two lines**: 
  - Line 1: MLP Encoder ($f$ = MLP)
  - Line 2: Corresponding Inverse Process Encoder ($f$ = Inverse)
- **X-axis**: Constraint ratio (0.0 → 1.0, marked every 0.1)
- **Y-axis**: Respective metric for that figure
- **Colors**: Blue for MLP, Red for Inverse Process (consistent across all figures)
- **Markers**: Different shapes for visual distinction
- **No error bars**: Clean line plots for clarity
- **Mathematical notation**: Use $g$ and $f$ in labels

### Missing Data Handling
- **MLP experiments**: Temporarily excluded from plotting
- **Missing experiment**: Show placeholder subplot or skip if no corresponding inverse process exists

### Design Features
- **Clear separation**: Each figure focuses on one encoder type
- **Easy comparison**: Same layout for direct visual comparison between figures
- **Thesis-friendly**: Can be placed on consecutive pages or side-by-side
- **Less cognitive load**: Reader processes 4 metrics at a time, not 8
- **Flexible placement**: Can separate with explanatory text between figures

## Appendix Tables

### Table Structure
- **Table A1**: Detailed numerical results for MLP encoders (mean ± std for all constraint ratios)
- **Table A2**: Detailed numerical results for Inverse Process encoders
- **Table A3**: Statistical significance tests (optional)
- **Table A4**: Best performing configurations summary

### Table Format
- LaTeX format for thesis inclusion
- Mathematical notation: $d_{\mathrm{fixed}}$, Process ($g$), etc.
- Mean ± std format for all metrics
- Consistent with previous table styles used in thesis

## Benefits of This Structure
✅ **Clean main narrative**: Figures tell the story visually  
✅ **Complete documentation**: Tables preserve all numerical details  
✅ **Reader choice**: Quick visual understanding vs. detailed analysis  
✅ **Space efficiency**: Main text stays focused and readable  
✅ **Reproducibility**: Full numerical data available for verification  

## Implementation Notes
- Use dual 2×2 grid visualization script
- Generate LaTeX tables for appendix
- Consistent color schemes across figures
- High-resolution output for thesis inclusion
- Error bars showing ±1 standard deviation
- Mathematical notation throughout

## Figure Captions
- **Figure 1**: "Effect of constraint sampling on MLP encoders across different generative processes"
- **Figure 2**: "Effect of constraint sampling on inverse process encoders across different generative processes"

This follows best academic practice: **figures for insight, tables for precision**.