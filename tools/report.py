#!/usr/bin/env python3
import os
import re
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CIFAR_LOG = os.path.join(ROOT, 'cifar10_extended.log')
NAS_LOG = os.path.join(ROOT, 'search_r50.log')
OUT_DIR = os.path.join(ROOT, 'reports')
NAS_CACHE_DIR = os.path.join(ROOT, 'save_model', 'R50_R224_FLOPs41e8', 'nas_cache')

os.makedirs(OUT_DIR, exist_ok=True)


def parse_cifar_log(path: str):
    best = None
    final = None
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        lines = f.readlines()
    acc_pat = re.compile(r'test acc ([0-9]+\.[0-9]+)%')
    for line in lines:
        m = acc_pat.search(line)
        if m:
            val = float(m.group(1))
            if best is None or val > best:
                best = val
        if line.startswith('FINAL '):
            final = line.strip()
    return {
        'best_test_acc_percent': best,
        'final_line': final
    }


def parse_nas_log(path: str):
    if not os.path.exists(path):
        return {}
    best_line = None
    with open(path, 'r') as f:
        for line in f:
            if 'best_individual:' in line:
                best_line = line.strip()
    final_summary = None
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('FINAL '):
                final_summary = line.strip()
    return {
        'best_individual_line': best_line,
        'final_line': final_summary
    }


def parse_nas_cache(cache_dir: str):
    if not os.path.isdir(cache_dir):
        return {}
    # pick latest iter*.txt
    files = [f for f in os.listdir(cache_dir) if f.startswith('iter') and f.endswith('.txt')]
    if not files:
        return {}
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    latest = os.path.join(cache_dir, files[-1])
    best_line = None
    try:
        with open(latest, 'r') as f:
            for line in f:
                if 'best_individual' in line or 'best' in line.lower():
                    best_line = line.strip()
        return {'latest_cache': latest, 'best_line': best_line}
    except Exception:
        return {'latest_cache': latest}


def write_json(data: dict, name: str):
    p = os.path.join(OUT_DIR, name)
    with open(p, 'w') as f:
        json.dump(data, f, indent=2)
    return p


def write_html(report: dict):
    p = os.path.join(OUT_DIR, 'report.html')
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NAS Concept Evolution Report</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; margin: 24px; }}
code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
pre {{ background: #f6f8fa; padding: 12px; border-radius: 6px; overflow: auto; }}
section {{ margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>NAS Concept Evolution Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<section>
<h2>CIFAR-10</h2>
<p><b>Best Test Accuracy</b>: {report.get('cifar',{}).get('best_test_acc_percent')}</p>
{f"<pre>{report.get('cifar',{}).get('final_line')}</pre>" if report.get('cifar',{}).get('final_line') else ''}
</section>
<section>
<h2>NAS Search (R50_FLOPs)</h2>
{f"<pre>{report.get('nas',{}).get('best_individual_line')}</pre>" if report.get('nas',{}).get('best_individual_line') else ''}
{f"<pre>{report.get('nas',{}).get('final_line')}</pre>" if report.get('nas',{}).get('final_line') else ''}
{f"<p><b>Latest Cache</b>: {report.get('nas_cache',{}).get('latest_cache')}</p>" if report.get('nas_cache',{}).get('latest_cache') else ''}
{f"<pre>{report.get('nas_cache',{}).get('best_line')}</pre>" if report.get('nas_cache',{}).get('best_line') else ''}
</section>
</body>
</html>
"""
    with open(p, 'w') as f:
        f.write(html)
    return p


def main():
    cifar = parse_cifar_log(CIFAR_LOG)
    nas = parse_nas_log(NAS_LOG)
    nas_cache = parse_nas_cache(NAS_CACHE_DIR)
    aggregated = {'cifar': cifar, 'nas': nas, 'nas_cache': nas_cache}
    json_path = write_json(aggregated, 'final_metrics.json')
    html_path = write_html(aggregated)
    print(f"Wrote {json_path}\nWrote {html_path}")

if __name__ == '__main__':
    main()
