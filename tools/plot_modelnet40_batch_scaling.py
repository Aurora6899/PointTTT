#!/usr/bin/env python3
"""Create the paper-ready Figure A from the batch-scaling CSV."""

import argparse
import csv
from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input', type=Path,
      default=Path('logs/modelnet40/batch_scaling/batch_scaling_summary.csv'))
  parser.add_argument(
      '--output', type=Path,
      default=Path('logs/modelnet40/batch_scaling/figure_a_batch_scaling'))
  return parser.parse_args()


def column(rows, key):
  return np.asarray([float(row[key]) for row in rows])


def main():
  args = parse_args()
  with args.input.open(newline='') as stream:
    rows = list(csv.DictReader(stream))

  batch = column(rows, 'batch_size').astype(int)
  on = column(rows, 'batch_latency_on_mean_ms')
  on_std = column(rows, 'batch_latency_on_std_ms')
  off = column(rows, 'batch_latency_off_mean_ms')
  off_std = column(rows, 'batch_latency_off_std_ms')
  overhead = column(rows, 'update_overhead_mean_ms')
  overhead_std = column(rows, 'update_overhead_std_ms')
  per_on = column(rows, 'per_sample_on_mean_ms')
  per_on_std = column(rows, 'per_sample_on_std_ms')
  per_off = column(rows, 'per_sample_off_mean_ms')
  per_off_std = column(rows, 'per_sample_off_std_ms')
  per_overhead = column(rows, 'per_sample_update_overhead_ms')
  per_overhead_std = overhead_std / batch

  times_font_paths = (
      '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',
      '/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf',
      '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf',
      '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf',
  )
  for font_path in times_font_paths:
    font_manager.fontManager.addfont(font_path)

  plt.rcParams.update({
      'font.family': 'Times New Roman',
      'font.size': 93.6,
      'axes.labelsize': 93.6,
      'axes.titlesize': 93.6,
      'legend.fontsize': 93.6,
      'pdf.fonttype': 42,
      'ps.fonttype': 42,
  })
  colors = {'on': '#E57373', 'off': '#0072B2', 'delta': '#009E73'}
  on_edge = '#B84A4A'
  fig, ax = plt.subplots(figsize=(48.0, 45.0))
  group_centers = np.arange(len(batch)) * 2.65
  offsets = (-1.12, -0.72, -0.32, 0.08, 0.48, 0.88)
  bar_height = 0.27

  for index, center in enumerate(group_centers):
    if index % 2:
      ax.axhspan(center - 1.22, center + 1.22,
                 color='#F4F5F6', zorder=0)

  series = (
      (on, on_std, offsets[0], colors['on'], False),
      (per_on, per_on_std, offsets[1], colors['on'], True),
      (off, off_std, offsets[2], colors['off'], False),
      (per_off, per_off_std, offsets[3], colors['off'], True),
      (overhead, overhead_std, offsets[4], colors['delta'], False),
      (per_overhead, per_overhead_std, offsets[5], colors['delta'], True),
  )
  for latency, std, offset, color, per_sample in series:
    y_values = group_centers + offset
    edge_color = on_edge if color == colors['on'] else color
    ax.barh(
        y_values, latency, height=bar_height,
        facecolor='white' if per_sample else color,
        edgecolor=edge_color, hatch='////' if per_sample else None,
        linewidth=1.0, zorder=3)
    ax.errorbar(
        latency, y_values, xerr=std, fmt='none', ecolor='#333333',
        elinewidth=0.8, capsize=2.0, zorder=4)
    for y_value, latency_value, std_value in zip(y_values, latency, std):
      ax.annotate(
          f'{latency_value:.2f}',
          xy=(latency_value + std_value, y_value), xytext=(4, 0),
          textcoords='offset points', va='center', ha='left',
          color=edge_color, fontsize=86.58, fontweight='bold', zorder=5)

  ax.set_yticks(group_centers, batch, fontweight='normal')
  ax.invert_yaxis()
  ax.set_xlabel('Latency (ms)')
  ax.set_ylabel('Batch size')
  max_latency = np.max(on + on_std)
  ax.set_xlim(0, max_latency * 1.23)
  ax.grid(axis='x', linestyle=(0, (1.5, 2.5)), linewidth=0.7,
          color='#C9CED3', alpha=0.8, zorder=1)
  ax.axvline(0, color='black', linewidth=5.0, zorder=4)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(True)
  ax.spines['left'].set_color('black')
  ax.spines['bottom'].set_color('black')
  ax.spines['left'].set_linewidth(5.0)
  ax.spines['bottom'].set_linewidth(5.0)
  ax.tick_params(axis='both', which='major', color='black',
                 labelcolor='black', width=5.0, length=14)
  ax.tick_params(axis='y', pad=20)

  metric_handles = [
      Patch(facecolor=colors['on'], edgecolor=on_edge,
            label='TTT update on'),
      Patch(facecolor=colors['off'], edgecolor=colors['off'],
            label='TTT update off'),
      Patch(facecolor=colors['delta'], edgecolor=colors['delta'],
            label='Update overhead'),
  ]
  style_handles = [
      Patch(facecolor='#777777', edgecolor='#777777', label='Per batch'),
      Patch(facecolor='white', edgecolor='#777777', hatch='////',
            label='Per sample'),
  ]
  fig.legend(handles=metric_handles, frameon=False, loc='upper center',
             bbox_to_anchor=(0.5, 0.99), ncol=3, columnspacing=2.0,
             handlelength=1.8)
  fig.legend(handles=style_handles, frameon=False, loc='upper center',
             bbox_to_anchor=(0.5, 0.925), ncol=2, columnspacing=2.2,
             handlelength=1.8)

  fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  for suffix in ('.pdf', '.png'):
    fig.savefig(args.output.with_suffix(suffix), dpi=200,
                bbox_inches='tight')
  plt.close(fig)


if __name__ == '__main__':
  main()
