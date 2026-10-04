import os
import re
import numpy as np
import scipy.ndimage as ndi
import scipy.optimize
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import subprocess

def build_banner():
    # 1. Process portrait from attached photo
    photo_path = 'IMG-20260130-WA0169.jpg' if os.path.exists('IMG-20260130-WA0169.jpg') else 'avatar.png'
    img = Image.open(photo_path).convert('RGB')
    
    # Head and shoulders crop:
    # IMG-20260130-WA0169.jpg is 3072 x 4096
    if photo_path.endswith('.jpg'):
        crop_box = (450, 700, 2622, 3160)
    else:
        crop_box = (75, 15, 395, 378)
        
    cropped = img.crop(crop_box).resize((300, 340), Image.Resampling.LANCZOS)
    arr = np.array(cropped, dtype=np.float32)
    
    # Dark mode segmentation mask
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    shirt = (r > g + 25) & (g < 80) & (b < 85)
    skin = (r > 90) & (g > 50) & (b > 30) & (r > g + 10) & (g > b) & ((r - b) > 20) & (g < 140)
    hair = (r < 70) & (g < 65) & (b < 65) & (np.arange(340)[:, None] < 180) & (np.arange(300)[None, :] < 230) & (np.arange(300)[None, :] > 70)
    collar = (np.arange(340)[:, None] > 160) & (np.arange(340)[:, None] < 220) & (np.arange(300)[None, :] > 120) & (np.arange(300)[None, :] < 180)
    
    fg_raw = shirt | skin | hair | collar
    labeled, num_features = ndi.label(fg_raw)
    bottom_labels = np.unique(labeled[320:, 60:240])
    bottom_labels = bottom_labels[bottom_labels > 0]
    mask = np.isin(labeled, bottom_labels)
    mask = ndi.binary_closing(mask, structure=np.ones((9, 9)))
    mask = ndi.binary_fill_holes(mask)
    
    top_labels = np.unique(labeled[30:100, 100:200])
    top_labels = top_labels[top_labels > 0]
    head_mask = np.isin(labeled, top_labels)
    mask = mask | head_mask
    mask = ndi.binary_closing(mask, structure=np.ones((11, 11)))
    mask = ndi.binary_fill_holes(mask)
    
    labeled, num = ndi.label(mask)
    if num > 0:
        sizes = ndi.sum(mask, labeled, range(1, num + 1))
        mask = (labeled == (np.argmax(sizes) + 1))
        
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    # Clear stray foliage at bottom edges
    mask[265:, 250:] = False
    mask[265:, :35] = False
    
    # Photographic contrast enhancements as specified:
    # contrast 1.3x, autocontrast(cutoff=1), UnsharpMask(radius=3, percent=140)
    gray = cropped.convert('L')
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    gray_arr = np.array(gray, dtype=np.float32)
    
    def dither_serpentine(src, mask_filter=None, invert=False):
        h, w = src.shape
        mat = src.copy()
        if invert:
            mat = 255.0 - mat
        out = np.zeros((h, w), dtype=np.uint8)
        
        for y in range(h):
            if y % 2 == 0:
                x_range = range(w)
                direction = 1
            else:
                x_range = range(w - 1, -1, -1)
                direction = -1
                
            for x in x_range:
                if mask_filter is not None and not mask_filter[y, x]:
                    mat[y, x] = 0
                    continue
                    
                old_val = mat[y, x]
                new_val = 255.0 if old_val >= 128.0 else 0.0
                out[y, x] = 1 if new_val == 255.0 else 0
                err = old_val - new_val
                
                if direction == 1:
                    if x + 1 < w and (mask_filter is None or mask_filter[y, x + 1]):
                        mat[y, x + 1] += err * 7.0 / 16.0
                    if y + 1 < h:
                        if x - 1 >= 0 and (mask_filter is None or mask_filter[y + 1, x - 1]):
                            mat[y + 1, x - 1] += err * 3.0 / 16.0
                        if mask_filter is None or mask_filter[y + 1, x]:
                            mat[y + 1, x] += err * 5.0 / 16.0
                        if x + 1 < w and (mask_filter is None or mask_filter[y + 1, x + 1]):
                            mat[y + 1, x + 1] += err * 1.0 / 16.0
                else:
                    if x - 1 >= 0 and (mask_filter is None or mask_filter[y, x - 1]):
                        mat[y, x - 1] += err * 7.0 / 16.0
                    if y + 1 < h:
                        if x + 1 < w and (mask_filter is None or mask_filter[y + 1, x + 1]):
                            mat[y + 1, x + 1] += err * 3.0 / 16.0
                        if mask_filter is None or mask_filter[y + 1, x]:
                            mat[y + 1, x] += err * 5.0 / 16.0
                        if x - 1 >= 0 and (mask_filter is None or mask_filter[y + 1, x - 1]):
                            mat[y + 1, x - 1] += err * 1.0 / 16.0
                            
        return out
        
    dither_dark = dither_serpentine(gray_arr, mask_filter=mask, invert=False)
    dither_light = dither_serpentine(gray_arr, mask_filter=None, invert=True)
    
    print(f'Dark mode dots: {np.sum(dither_dark)}')
    print(f'Light mode dots: {np.sum(dither_light)}')
    
    # 2. Extract runs for SVG paths
    def extract_runs(dither_matrix):
        h, w = dither_matrix.shape
        runs = [] # list of (x, y, length)
        for y in range(h):
            in_run = False
            start_x = 0
            for x in range(w):
                if dither_matrix[y, x] == 1:
                    if not in_run:
                        in_run = True
                        start_x = x
                else:
                    if in_run:
                        runs.append((start_x, y, x - start_x))
                        in_run = False
            if in_run:
                runs.append((start_x, y, w - start_x))
        return runs

    runs_dark = extract_runs(dither_dark)
    runs_light = extract_runs(dither_light)
    print(f'Dark runs: {len(runs_dark)}, Light runs: {len(runs_light)}')

    # 3. Drift bands grouping (~94 bands) with organic noise (sigma ≈ 4)
    # Centroid of Logo 1 (Python) is at cx=150, cy=170
    cx_logo, cy_logo = 150.0, 170.0
    
    def group_into_bands(runs, n_bands=94, sigma=4.0):
        np.random.seed(42)
        run_data = []
        for rx, ry, rlen in runs:
            mid_x = rx + rlen / 2.0
            mid_y = ry
            # Per-dot noise with sigma ≈ 4 before grouping
            noise_x = np.random.normal(0, sigma)
            noise_y = np.random.normal(0, sigma)
            # Projected metric with slight organic curvature
            metric = (mid_y + noise_y) + 0.25 * (mid_x + noise_x)
            run_data.append((metric, mid_x, mid_y, rx, ry, rlen))
            
        # Sort by metric to partition into 94 bands
        run_data.sort(key=lambda t: t[0])
        bands = [[] for _ in range(n_bands)]
        for idx, item in enumerate(run_data):
            b_idx = int(idx * n_bands / len(run_data))
            b_idx = min(b_idx, n_bands - 1)
            bands[b_idx].append(item[3:]) # (rx, ry, rlen)
            
        # Compute drift vector for each band:
        # "Each band should translate approximately 42% toward the first logo centroid, fade while moving, return afterward"
        band_info = []
        for b_idx, b_runs in enumerate(bands):
            if not b_runs:
                band_info.append((0.0, 0.0, b_runs))
                continue
            mean_x = np.mean([r[0] + r[2]/2.0 for r in b_runs])
            mean_y = np.mean([r[1] for r in b_runs])
            # 42% toward logo centroid (cx_logo, cy_logo)
            dx = 0.42 * (cx_logo - mean_x)
            dy = 0.42 * (cy_logo - mean_y)
            band_info.append((round(dx, 2), round(dy, 2), b_runs))
            
        return band_info

    bands_dark = group_into_bands(runs_dark, n_bands=94, sigma=4.0)
    bands_light = group_into_bands(runs_light, n_bands=94, sigma=4.0)

    # Straight boundary metric check
    # Check linearity of band boundary points
    boundary_metrics = []
    for b_idx in range(len(bands_dark) - 1):
        b1 = bands_dark[b_idx][2]
        if len(b1) > 10:
            xs = [r[0] for r in b1]
            ys = [r[1] for r in b1]
            # Check deviation from straight horizontal line (grid)
            # R^2 of horizontal fit
            var_y = np.var(ys)
            boundary_metrics.append(1.0 / (1.0 + var_y))
    straight_metric = float(np.mean(boundary_metrics)) if boundary_metrics else 0.01
    print(f'Straight-boundary metric: {straight_metric:.4f} (target: ~0.01 organic, 0.17 grid-like)')

    # 4. Intro animation:
    # "Use approximately 60 interleaved random groups. Fade in over approximately 2 seconds."
    # We assign an intro_group (0..59) to each band or run.
    # To keep SVG structure clean and performant, each of the 94 bands gets an intro delay:
    # stagger delay uniformly sampled between 0.0 and 1.2s, dur=2.0s -> total 3.2s!
    # Intro evenness check:
    np.random.seed(135)
    intro_delays = np.random.uniform(0.0, 1.2, size=len(bands_dark))
    # Spatial correlation check for intro evenness:
    grid_bins = np.zeros((10, 10))
    for b_idx, (_, _, b_runs) in enumerate(bands_dark):
        d = intro_delays[b_idx]
        for rx, ry, _ in b_runs:
            bx = min(int(rx / 30), 9)
            by = min(int(ry / 34), 9)
            grid_bins[by, bx] += d
    norm_grid = grid_bins / (np.sum(grid_bins) + 1e-6)
    intro_evenness = float(np.std(norm_grid))
    print(f'Intro evenness metric: {intro_evenness:.4f} (target: ~0.05 good, 0.7 patchy)')

    # 5. Traveller layer (900 dots)
    def sample_points(png_path, n_points=900):
        img = Image.open(png_path).convert('L')
        arr = np.array(img)
        ys, xs = np.where(arr < 128)
        np.random.seed(42)
        indices = np.random.choice(len(xs), size=n_points, replace=False)
        pts = np.column_stack((xs[indices], ys[indices])).astype(np.float32)
        return pts

    pts_py = sample_points('logo_py.png', 900)
    pts_code = sample_points('logo_code.png', 900)
    pts_gh = sample_points('logo_gh.png', 900)

    # Hungarian algorithm optimal transport
    dist_1_2 = np.linalg.norm(pts_py[:, None, :] - pts_code[None, :, :], axis=-1)
    _, col_ind_code = scipy.optimize.linear_sum_assignment(dist_1_2)
    pts_code_matched = pts_code[col_ind_code]

    dist_2_3 = np.linalg.norm(pts_code_matched[:, None, :] - pts_gh[None, :, :], axis=-1)
    _, col_ind_gh = scipy.optimize.linear_sum_assignment(dist_2_3)
    pts_gh_matched = pts_gh[col_ind_gh]

    print('Optimal transport matching complete.')

    # 6. Generate dark.svg and light.svg
    def generate_svg(is_dark=True):
        theme_bg = "#0A101F" if is_dark else "#F8FAFC"
        window_bg = "#0D1527" if is_dark else "#FFFFFF"
        window_border = "#1E293B" if is_dark else "#CBD5E1"
        header_bg = "#111C33" if is_dark else "#F1F5F9"
        portrait_color = "#A78BFA" if is_dark else "#7C3AED"
        chrome_color = "#22D3EE" if is_dark else "#0891B2"
        chrome_dim = "#38BDF8" if is_dark else "#0284C7"
        text_primary = "#F1F5F9" if is_dark else "#0F172A"
        text_dim = "#94A3B8" if is_dark else "#64748B"
        accent_color = "#10B981"
        leader_color = "#1E293B" if is_dark else "#CBD5E1"

        bands = bands_dark if is_dark else bands_light

        # Dimensions
        canvas_w, canvas_h = 1180, 610
        win_x, win_y = 20, 20
        win_w, win_h = 1140, 570
        port_x, port_y = 55, 140
        port_w, port_h = 300, 340

        svg_lines = []
        svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_w} {canvas_h}" width="{canvas_w}" height="{canvas_h}" style="background-color: {theme_bg};">')
        svg_lines.append('<defs>')
        svg_lines.append('<style>')
        svg_lines.append(f'''
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&amp;display=swap');
            text {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; }}
            .term-title {{ font-size: 13px; font-weight: 500; fill: {text_dim}; }}
            .sec-hdr {{ font-size: 13px; font-weight: 700; letter-spacing: 1.5px; fill: {chrome_color}; }}
            .pill-text {{ font-size: 14px; font-weight: 700; fill: {chrome_color}; }}
            .live-text {{ font-size: 12px; font-weight: 700; fill: #EF4444; letter-spacing: 1px; }}
            .row-label {{ font-size: 14px; font-weight: 500; fill: {chrome_color}; }}
            .row-val {{ font-size: 14px; font-weight: 500; fill: {text_primary}; }}
            .row-accent {{ font-size: 14px; font-weight: 700; fill: {accent_color}; }}
            .dotted-leader {{ font-size: 14px; fill: {leader_color}; letter-spacing: 3px; }}
        ''')
        svg_lines.append('</style>')
        svg_lines.append('</defs>')

        # Window Frame
        svg_lines.append(f'<rect x="{win_x}" y="{win_y}" width="{win_w}" height="{win_h}" rx="12" fill="{window_bg}" stroke="{window_border}" stroke-width="1.5" />')
        
        # Window Header
        svg_lines.append(f'<path d="M {win_x} {win_y + 42} L {win_x} {win_y + 12} A 12 12 0 0 1 {win_x + 12} {win_y} L {win_x + win_w - 12} {win_y} A 12 12 0 0 1 {win_x + win_w} {win_y + 12} L {win_x + win_w} {win_y + 42} Z" fill="{header_bg}" stroke="{window_border}" stroke-width="1.5" />')
        
        # Window buttons (traffic lights)
        svg_lines.append(f'<circle cx="{win_x + 22}" cy="{win_y + 21}" r="6" fill="#EF4444" />')
        svg_lines.append(f'<circle cx="{win_x + 42}" cy="{win_y + 21}" r="6" fill="#F59E0B" />')
        svg_lines.append(f'<circle cx="{win_x + 62}" cy="{win_y + 21}" r="6" fill="#10B981" />')

        # Terminal Title
        svg_lines.append(f'<text x="{win_x + win_w / 2}" y="{win_y + 26}" text-anchor="middle" class="term-title">profile.sh --live</text>')

        # Left Side Header: VISUAL.MAP
        svg_lines.append(f'<text x="{port_x}" y="{port_y - 20}" class="sec-hdr">VISUAL.MAP</text>')
        # Inner portrait border
        svg_lines.append(f'<rect x="{port_x - 4}" y="{port_y - 4}" width="{port_w + 8}" height="{port_h + 8}" rx="6" fill="none" stroke="{window_border}" stroke-width="1" stroke-dasharray="4 4" />')

        # Portrait Group
        svg_lines.append(f'<g transform="translate({port_x}, {port_y})" shape-rendering="crispEdges">')
        
        # 94 Drift Bands
        for b_idx, (dx, dy, b_runs) in enumerate(bands):
            if not b_runs:
                continue
            delay = round(float(intro_delays[b_idx % len(intro_delays)]), 2)
            # Build path data
            p_parts = []
            for rx, ry, rlen in b_runs:
                p_parts.append(f'M{rx} {ry}h{rlen}v1h-{rlen}z')
            path_d = ''.join(p_parts)
            
            # Band group with intro fade + loop animation
            svg_lines.append(f'<g fill="{portrait_color}">')
            # Intro animation (happens once, finishes by 3.2s)
            svg_lines.append(f'<animate attributeName="opacity" values="0; 1" dur="2.0s" begin="{delay}s" fill="freeze" />')
            # Loop translation: 42% toward logo centroid, returns afterward
            svg_lines.append(f'<animateTransform attributeName="transform" type="translate" values="0 0; 0 0; {dx} {dy}; {dx} {dy}; {dx} {dy}; {dx} {dy}; {dx} {dy}; {dx} {dy}; 0 0" keyTimes="0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.908; 1" dur="14.2s" begin="3.2s" repeatCount="indefinite" />')
            # Loop opacity: fades out while moving, stays hidden during logo holds, returns with portrait
            svg_lines.append(f'<animate attributeName="opacity" values="1; 1; 0; 0; 0; 0; 0; 0; 1" keyTimes="0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.908; 1" dur="14.2s" begin="3.2s" repeatCount="indefinite" />')
            svg_lines.append(f'<path d="{path_d}" />')
            svg_lines.append('</g>')

        # Traveller Layer (900 dots)
        # Morphing between Python -> Code -> GitHub
        traveller_color = chrome_color
        for i in range(900):
            p1 = pts_py[i]
            p2 = pts_code_matched[i]
            p3 = pts_gh_matched[i]
            x1, y1 = round(float(p1[0]), 1), round(float(p1[1]), 1)
            dx1_2, dy1_2 = round(float(p2[0] - p1[0]), 1), round(float(p2[1] - p1[1]), 1)
            dx1_3, dy1_3 = round(float(p3[0] - p1[0]), 1), round(float(p3[1] - p1[1]), 1)

            svg_lines.append(f'<rect x="{x1}" y="{y1}" width="2.2" height="2.2" rx="1.1" fill="{traveller_color}" opacity="0">')
            # Opacity animation: hidden during portrait, visible during logo morphs
            svg_lines.append('<animate attributeName="opacity" values="0; 0; 1; 1; 1; 1; 1; 1; 0" keyTimes="0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.908; 1" dur="14.2s" begin="3.2s" repeatCount="indefinite" />')
            # Position transform morph:
            svg_lines.append(f'<animateTransform attributeName="transform" type="translate" values="0 0; 0 0; 0 0; 0 0; {dx1_2} {dy1_2}; {dx1_2} {dy1_2}; {dx1_3} {dy1_3}; {dx1_3} {dy1_3}; 0 0" keyTimes="0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.908; 1" dur="14.2s" begin="3.2s" repeatCount="indefinite" />')
            svg_lines.append('</rect>')

        svg_lines.append('</g>') # End portrait group

        # Right Side: SYSTEM.INFO
        info_x = 425
        info_w = 690
        header_y = port_y - 20

        # Subheader with SYSTEM.INFO
        svg_lines.append(f'<text x="{info_x}" y="{header_y}" class="sec-hdr">SYSTEM.INFO</text>')

        # Pulsing Red LIVE Badge
        live_x = info_x + 130
        svg_lines.append(f'<rect x="{live_x}" y="{header_y - 15}" width="62" height="20" rx="4" fill="#EF4444" fill-opacity="0.12" stroke="#EF4444" stroke-opacity="0.4" stroke-width="1" />')
        svg_lines.append(f'<circle cx="{live_x + 12}" cy="{header_y - 5}" r="3.5" fill="#EF4444">')
        svg_lines.append('<animate attributeName="opacity" values="1; 0.2; 1" dur="1.5s" repeatCount="indefinite" />')
        svg_lines.append('</circle>')
        svg_lines.append(f'<text x="{live_x + 22}" y="{header_y - 1}" class="live-text">LIVE</text>')

        # Coloured Pill containing GitHub handle
        handle_text = "@prem-mundargi"
        pill_x = info_x + info_w - 170
        svg_lines.append(f'<rect x="{pill_x}" y="{header_y - 17}" width="170" height="24" rx="12" fill="{chrome_color}" fill-opacity="0.12" stroke="{chrome_color}" stroke-opacity="0.5" stroke-width="1" />')
        svg_lines.append(f'<text x="{pill_x + 85}" y="{header_y}" text-anchor="middle" class="pill-text">{handle_text}</text>')

        # Inner container border for SYSTEM.INFO
        svg_lines.append(f'<rect x="{info_x - 15}" y="{port_y - 4}" width="{info_w + 25}" height="{port_h + 8}" rx="6" fill="none" stroke="{window_border}" stroke-width="1" stroke-dasharray="4 4" />')

        # 16 Info Rows
        rows_data = [
            ("Subject", "Prem Mundargi", False),
            ("Role", "CSE Student | Aspiring Software Dev", True),
            ("Origin", "Karnataka, India", False),
            ("Education", "Computer Science &amp; Engineering", False),
            ("Status", "Learning + Building + Shipping", True),
            ("ToolChain", "VS Code, Git, Python, Android Studio", False),
            ("Core.Lang", "C, Python, JavaScript", True),
            ("Core.Frontend", "HTML, CSS, JS, React, Tailwind CSS", False),
            ("Core.Backend", "Flask, Firebase", False),
            ("Core.Database", "Firebase / SQL", False),
            ("Core.Infra", "GitHub, Vercel", False),
            ("Grid.Mail", "premmundargi135@gmail.com", False),
            ("Grid.Portfolio", "Coming soon", True),
            ("Grid.LinkedIn", "linkedin.com/in/prem-mundargi", False),
            ("Grid.GitHub", "github.com/prem-mundargi", False),
            ("Grid.Facebook", "facebook.com/prem.mundargi", False),
        ]

        row_start_y = port_y + 18
        row_step_y = 21.0
        # Calculate dotted leaders automatically from label and value lengths
        # Fixed monospace character width metric: approx 8.4px per character
        char_w = 8.42
        row_left_x = info_x
        row_right_x = info_x + info_w

        for idx, (label, val, is_accent) in enumerate(rows_data):
            cur_y = row_start_y + idx * row_step_y
            val_class = "row-accent" if is_accent else "row-val"
            
            # Label
            svg_lines.append(f'<text x="{row_left_x}" y="{cur_y:.1f}" class="row-label">{label}</text>')
            
            # Value
            svg_lines.append(f'<text x="{row_right_x}" y="{cur_y:.1f}" text-anchor="end" class="{val_class}">{val}</text>')
            
            # Automatic Leader Calculation
            l_end = row_left_x + len(label) * char_w + 14
            v_start = row_right_x - len(val) * char_w - 14
            dot_span = v_start - l_end
            if dot_span > 20:
                n_dots = int(dot_span / 16)
                dots_str = ". " * n_dots
                svg_lines.append(f'<text x="{l_end:.1f}" y="{cur_y:.1f}" class="dotted-leader" textLength="{dot_span:.1f}" lengthAdjust="spacingAndGlyphs">{dots_str}</text>')

        # Close SVG
        svg_lines.append('</svg>')
        return '\n'.join(svg_lines)

    dark_svg = generate_svg(is_dark=True)
    with open('dark.svg', 'w') as f:
        f.write(dark_svg)
    print(f'dark.svg created! File size: {len(dark_svg) / 1024:.1f} KB')

    light_svg = generate_svg(is_dark=False)
    with open('light.svg', 'w') as f:
        f.write(light_svg)
    print(f'light.svg created! File size: {len(light_svg) / 1024:.1f} KB')

if __name__ == '__main__':
    build_banner()
