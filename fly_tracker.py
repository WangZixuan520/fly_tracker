#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
果蝇攀爬行为追踪分析系统 v20-modified
========================================
基于v20，修改：
1. 攀爬成功率判定线从1/2改为2/3
2. 速度计算从平均值改为中位数（消除异常值影响）
3. 输出JSON + CSV + JPG诊断图 + Excel汇总
"""

import cv2
import numpy as np
import pandas as pd
import json
import re
import os
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

TUBE_CONFIGS = {
    "5mm": {
        "tubes": 19, "pattern": "1010101010101010101",
        "exp_indices": [0,2,4,6,8,10,12,14,16,18],
        "big_tube_od_mm": 7, "small_tube_od_mm": 7,
        "total_width_mm": 142, "tube_height_mm": 200,
    },
    "10mm": {
        "tubes": 13, "pattern": "1010101010101",
        "exp_indices": [0,2,4,6,8,10,12],
        "big_tube_od_mm": 14, "small_tube_od_mm": 9,
        "total_width_mm": 134, "tube_height_mm": 200,
    },
    "20mm": {
        "tubes": 9, "pattern": "101010101",
        "exp_indices": [0,2,4,6,8],
        "big_tube_od_mm": 24, "small_tube_od_mm": 5,
        "total_width_mm": 137, "tube_height_mm": 200,
    },
    "40mm": {
        "tubes": 5, "pattern": "10101",
        "exp_indices": [0,2,4],
        "big_tube_od_mm": 50, "small_tube_od_mm": 5,
        "shade_tube_od_mm": 7,
        "total_width_mm": 176, "tube_height_mm": 200,
    },
}

CALIBRATION_TIME = 10.0
REACH = 2.0 / 3.0  # 成功率判定线：2/3管高


def parse_filename(filename):
    name = Path(filename).stem.lower()
    m = re.match(r'(5mm|10mm|20mm|40mm)[-_]', name)
    if not m:
        return None, None
    g = 'F' if '-f-' in name else 'M' if '-m-' in name else '?'
    return m.group(1), g


def detect_walls_by_projection(gray, n_tubes, cfg):
    """v20原始管壁检测"""
    h, w = gray.shape
    y1, y2 = int(h*0.30), int(h*0.70)
    roi = gray[y1:y2, :]
    clahe = cv2.createCLAHE(2.0, (8, 8))
    roi_enh = clahe.apply(roi)
    grad_x = cv2.Sobel(roi_enh, cv2.CV_64F, 1, 0, ksize=3)
    grad_abs = np.abs(grad_x)
    projection = grad_abs.sum(axis=0)
    from scipy.ndimage import gaussian_filter1d
    smooth = gaussian_filter1d(projection, sigma=3)
    from scipy.signal import find_peaks
    peaks, props = find_peaks(smooth, distance=20, prominence=smooth.max()*0.05)
    n_walls = n_tubes + 1
    px_per_mm = w / cfg['total_width_mm'] * 0.8
    big_px = int(cfg['big_tube_od_mm'] * px_per_mm)
    small_px = int(cfg['small_tube_od_mm'] * px_per_mm)
    template_gaps = []
    for i in range(n_tubes):
        is_big = cfg['pattern'][i] == '1'
        tube_w = big_px if is_big else small_px
        template_gaps.append(tube_w)
    best_score = float('inf')
    best_walls = None
    if len(peaks) >= n_walls:
        for start in range(len(peaks) - n_walls + 1):
            for step in range(1, min(5, len(peaks) - start - n_walls + 2)):
                selected = peaks[start:start + n_walls * step:step]
                if len(selected) < n_walls:
                    continue
                selected = selected[:n_walls]
                actual_gaps = np.diff(selected)
                score = sum((ag - tg) ** 2 for ag, tg in zip(actual_gaps, template_gaps))
                if score < best_score:
                    best_score = score
                    best_walls = selected.tolist()
    if best_walls is None or best_score > (big_px ** 2) * n_tubes * 0.5:
        log.info(f"峰值匹配不佳(得分={best_score:.0f})，退回到物理估算")
        total_px = sum(template_gaps)
        start_x = (w - total_px) // 2
        best_walls = [start_x]
        for gap in template_gaps:
            best_walls.append(best_walls[-1] + gap)
    return best_walls, big_px, small_px


def divide_tubes(wall_x_list, n_tubes, pattern):
    """划分管内区域"""
    tubes = []
    for i in range(min(n_tubes, len(wall_x_list)-1)):
        cx = (wall_x_list[i] + wall_x_list[i+1]) // 2
        tw = wall_x_list[i+1] - wall_x_list[i]
        is_exp = pattern[i] == '1' if i < len(pattern) else False
        tubes.append((i+1, int(cx), int(tw), is_exp))
    return tubes


def extract_roi(gray, cx, y_top, y_bottom, tube_width):
    h, w = gray.shape
    half_w = max(tube_width // 2, 5)
    x1 = max(0, cx - half_w)
    x2 = min(w, cx + half_w)
    return gray[y_top:y_bottom, x1:x2], x1, y_top


def find_fly(roi, prev_y=None):
    """v20原始果蝇检测"""
    if roi.size == 0:
        return None
    h, w = roi.shape
    clahe = cv2.createCLAHE(3.0, (4, min(16, w)))
    en = clahe.apply(roi)
    blur = cv2.GaussianBlur(en, (3, 3), 0)
    med = np.median(blur)
    th = max(40, int(med - 30))
    _, bi = cv2.threshold(blur, th, 255, cv2.THRESH_BINARY_INV)
    k = np.ones((2, 2), np.uint8)
    bi = cv2.morphologyEx(bi, cv2.MORPH_OPEN, k, iterations=1)
    cs, _ = cv2.findContours(bi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cs:
        a = cv2.contourArea(c)
        if 3 < a < 200:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                if 2 < cy < h-2:
                    cands.append((cx, cy, a))
    if not cands:
        return None
    if prev_y and prev_y > 0:
        b = min(cands, key=lambda c: abs(c[1]-prev_y))
        if abs(b[1]-prev_y) < h*0.4:
            return b[:2]
    return max(cands, key=lambda c: c[2])[:2]


def calc_metrics(pos, y_tgt, yb, yt, fps, px2mm=1.0):
    """计算指标 - 速度改用中位数"""
    m = {
        'avg_spd': 0.0, 'avg_mm': 0.0, 'max_spd': 0.0, 'max_mm': 0.0,
        'med_spd': 0.0, 'med_mm': 0.0,  # 中位数速度
        'tot_px': 0.0, 'tot_mm': 0.0, 'net_px': 0.0, 'net_mm': 0.0,
        'eff': 0.0, 'falls': 0, 'pauses': 0, 'revs': 0,
        't_bot': 0.0, 't_mid': 0.0, 't_top': 0.0,
        'max_y': yb, 'max_mm_pos': 0.0, 'activity': 0.0,
        'reach': False, 'reach_t': None
    }
    if len(pos) < 2:
        return m
    ys = np.array([p[3] for p in pos])
    ts = np.array([p[1] for p in pos])
    fs = np.array([p[0] for p in pos])
    
    # 速度计算
    dys = -np.diff(ys)
    dts = np.diff(ts)
    sp = np.abs(dys) / np.where(dts > 0, dts, 0.001)
    
    # 原始平均（保留参考）
    m['avg_spd'] = round(float(np.mean(sp)), 2)
    m['avg_mm'] = round(float(np.mean(sp)) * px2mm, 2)
    
    # ========== 关键修复：中位数速度 ==========
    m['med_spd'] = round(float(np.median(sp)), 2)
    m['med_mm'] = round(float(np.median(sp)) * px2mm, 2)
    
    # 最大速度
    m['max_spd'] = round(float(np.max(sp)), 2)
    m['max_mm'] = round(float(np.max(sp)) * px2mm, 2)
    
    # 总距离/净位移
    m['tot_px'] = round(float(np.sum(np.abs(dys))), 1)
    m['tot_mm'] = round(float(np.sum(np.abs(dys))) * px2mm, 1)
    m['net_px'] = round(float(ys[0]-ys[-1]), 1)
    m['net_mm'] = round(float(ys[0]-ys[-1]) * px2mm, 1)
    
    # 效率
    tu = float(np.sum(dys[dys > 0]))
    if tu > 0:
        m['eff'] = round(max(0, ys[0]-ys[-1]) / tu, 3)
    
    # 区域时间
    bt, tt = yt+(yb-yt)*0.7, yt+(yb-yt)*0.3
    m['t_bot'] = round(float(np.sum(ys >= bt))/fps, 1)
    m['t_top'] = round(float(np.sum(ys <= tt))/fps, 1)
    m['t_mid'] = round(float(np.sum((ys > tt) & (ys < bt)))/fps, 1)
    
    # 最高位置
    m['max_y'] = int(ys.min())
    m['max_mm_pos'] = round((yb - ys.min()) * px2mm, 1)
    
    # 活跃比
    ac = sp > 2.0
    if len(ac) > 0:
        m['activity'] = round(float(np.sum(ac)/len(ac)), 3)
    
    # 转向
    rev, ld = 0, 0
    for i in range(len(dys)):
        if abs(dys[i]) < 5:
            continue
        d = 1 if dys[i] > 0 else -1
        if ld != 0 and d != ld:
            rev += 1
        ld = d
    m['revs'] = rev
    
    # 滑落
    fc, lff = 0, -8
    for i in range(1, len(ys)):
        dy, dt = ys[i]-ys[i-1], ts[i]-ts[i-1]
        if dt <= 0:
            continue
        if dy > 30 and abs(dy)/dt > 50 and fs[i]-lff >= 8:
            fc += 1
            lff = fs[i]
    m['falls'] = fc
    
    # 停顿
    pc, ip, ps, lpe = 0, False, 0, -15
    for i in range(1, len(ys)):
        dt = ts[i]-ts[i-1]
        if dt <= 0:
            continue
        s = abs(ys[i]-ys[i-1])/dt
        if s < 3.0:
            if not ip:
                ip = True
                ps = fs[i]
        else:
            if ip:
                if fs[i]-ps >= 20 and ps-lpe >= 15:
                    pc += 1
                    lpe = fs[i]
                ip = False
    if ip and fs[-1]-ps >= 20 and ps-lpe >= 15:
        pc += 1
    m['pauses'] = pc
    
    return m


def process(vp, out):
    """处理单个视频"""
    vp = Path(vp)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dia, gen = parse_filename(vp.name)
    if not dia or dia not in TUBE_CONFIGS:
        log.error(f"不支持: {dia}")
        return None
    cfg = TUBE_CONFIGS[dia]
    n, pat = cfg['tubes'], cfg['pattern']
    
    log.info(f"\n{'='*60}")
    log.info(f"处理: {vp.name}")
    log.info(f"管径: {dia}, 性别: {gen}, 管数: {n}")
    
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        log.error("打不开")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"视频: {w}x{h}, {fps:.0f}fps, {nf}帧")
    
    # 标定帧（保存彩色版用于诊断图）
    cf = min(int(fps * CALIBRATION_TIME), nf - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cf)
    ret, fr = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, fr = cap.read()
        if not ret:
            cap.release()
            return None
    
    fr_color = fr.copy()  # 保存彩色帧用于诊断图
    g0 = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    
    # 管壁检测
    wall_x, big_px, small_px = detect_walls_by_projection(g0, n, cfg)
    tubes = divide_tubes(wall_x, n, pat)
    exp = [t for t in tubes if t[3]]
    
    # px2mm
    if len(wall_x) >= 2:
        px_width = wall_x[-1] - wall_x[0]
    else:
        px_width = w
    px2mm = cfg['total_width_mm'] / max(px_width, 1)
    log.info(f"标定: 1px={px2mm:.4f}mm")
    
    yt, yb = int(h*0.08), int(h*0.95)
    ytg = int(yb - (yb - yt) * REACH)
    log.info(f"追踪: y=[{yt}-{yb}], 判定线(2/3): y={ytg}")
    
    # 追踪
    trk = {}
    for tid, cx, tw, _ in exp:
        trk[tid] = {'pos': [], 'prev_y': None, 'reach': False, 'reach_t': None}
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fi in range(nf):
        if fi % 500 == 0:
            log.info(f"  帧 {fi}/{nf}")
        ret, fr = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        tsec = fi/fps
        for tid, cx, tw, _ in exp:
            roi, rx, ry = extract_roi(gray, cx, yt, yb, tw)
            tr = trk[tid]
            r = find_fly(roi, tr['prev_y'])
            if r:
                lx, ly = r
                gx, gy = rx + lx, ry + ly
                tr['prev_y'] = ly
                tr['pos'].append((fi, tsec, int(gx), int(gy)))
                if not tr['reach'] and gy <= ytg and len(tr['pos']) > 10:
                    tr['reach'] = True
                    tr['reach_t'] = tsec
    cap.release()
    
    # 计算指标
    res = {}
    for tid, tr in trk.items():
        pos = tr['pos']
        m = calc_metrics(pos, ytg, yb, yt, fps, px2mm)
        m['tube'] = tid
        m['reach'] = tr['reach']
        m['reach_t'] = tr['reach_t']
        m['n_frames'] = len(pos)
        res[tid] = m
    
    # 保存JSON + CSV + JPG（使用彩色帧作为诊断图背景）
    save_results(vp.stem, res, out, trk, yt, yb, tubes, fr_color, px2mm)
    
    active = sum(1 for r in res.values() if r['n_frames'] > 0)
    reach = sum(1 for r in res.values() if r['reach'])
    log.info(f"完成: {len(exp)}支实验管, {active}支有轨迹, {reach}支到达2/3高度")
    return res


def save_results(vid, res, out, trk, yt, yb, tubes, color_frame, px2mm):
    """保存JSON + CSV + JPG诊断图（使用彩色帧作为背景）"""
    out = Path(out)
    
    try:
        # JSON
        with open(out/f"{vid}_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        
        # CSV轨迹
        with open(out/f"{vid}_trajectories.csv", 'w') as f:
            f.write('tube,frame,time,x,y\n')
            for tid, tr in trk.items():
                for p in tr['pos']:
                    f.write(f'{tid},{p[0]},{p[1]:.3f},{p[2]},{p[3]}\n')
        
        # JPG诊断图（使用彩色帧作为背景）- 用PIL保存避免中文路径问题
        vis = color_frame.copy()
        for tid, cx, tw, ie in tubes:
            c = (0, 255, 0) if ie else (128, 128, 128)
            x1, x2 = cx - tw//2, cx + tw//2
            cv2.line(vis, (x1, yt), (x1, yb), c, 2 if ie else 1)
            cv2.line(vis, (x2, yt), (x2, yb), c, 2 if ie else 1)
            cv2.putText(vis, f"T{tid}{'*' if ie else ''}", (cx-15, yt+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2 if ie else 1)
        ytg = int(yb - (yb - yt) * REACH)
        cv2.line(vis, (0, ytg), (vis.shape[1], ytg), (0, 0, 255), 2)
        cv2.putText(vis, f"1px={px2mm:.3f}mm  REACH=2/3", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 用绝对路径+PIL保存，避免中文路径问题
        jpg_path = (out / f"{vid}_tubes.jpg").resolve()
        try:
            from PIL import Image
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            Image.fromarray(vis_rgb).save(str(jpg_path), quality=95)
            log.info(f"  诊断图: {jpg_path}")
        except Exception as e:
            log.error(f"  诊断图保存失败: {e}")
            # 备用：cv2直接写
            try:
                cv2.imwrite(str(jpg_path), vis)
                log.info(f"  诊断图(cv2): {jpg_path}")
            except Exception as e2:
                log.error(f"  cv2也失败: {e2}")
    except Exception as e:
        log.error(f"  保存结果时出错: {e}")
        import traceback
        traceback.print_exc()


def batch(inp, out, filter_dias=None):
    """批量处理"""
    inp = Path(inp)
    out = Path(out)
    vids = []
    for e in ['.mp4', '.avi', '.mov', '.mkv']:
        vids.extend(inp.rglob(f'*{e}'))
    vids = sorted(vids)
    if filter_dias:
        filter_dias = [d.lower() for d in filter_dias]
        vids = [v for v in vids if parse_filename(v.name)[0] in filter_dias]
    log.info(f"\n发现 {len(vids)} 个视频")
    if not vids:
        return
    
    all_res = {}
    for v in vids:
        try:
            r = process(str(v), out)
            if r:
                all_res[v.name] = r
        except Exception as e:
            log.error(f"失败 {v.name}: {e}")
    
    if not all_res:
        return
    
    # Excel汇总 - 速度用中位数
    rows = []
    for vn, tubes in all_res.items():
        dia, gen = parse_filename(vn)
        parts = Path(vn).stem.split('-')
        day = parts[2] if len(parts) > 2 else ''
        for tid, d in tubes.items():
            rows.append({
                '视频': vn,
                '管径': dia,
                '性别': gen,
                '天数': day,
                '管号': f'T{tid}',
                '速度_中位数_mm/s': d['med_mm'],
                '速度_平均_mm/s': d['avg_mm'],
                '最大速度_mm/s': d['max_mm'],
                '总距离_mm': d['tot_mm'],
                '净位移_mm': d['net_mm'],
                '效率': d['eff'],
                '到达_2/3高度': '是' if d['reach'] else '否',
                '到达时间_s': d.get('reach_t'),
                '最高位置_mm': d['max_mm_pos'],
                '滑落次数': d['falls'],
                '停顿次数': d['pauses'],
                '转向次数': d['revs'],
                '底部时间_s': d['t_bot'],
                '顶部时间_s': d['t_top'],
                '活跃比': d['activity'],
                '追踪帧数': d['n_frames']
            })
    
    df = pd.DataFrame(rows)
    xp = out / '实验数据汇总.xlsx'
    with pd.ExcelWriter(xp, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='原始数据', index=False)
        if not df.empty:
            # 按管径统计
            s = df.groupby('管径').agg({
                '到达_2/3高度': lambda x: (x=='是').sum(),
                '管号': 'count',
                '速度_中位数_mm/s': 'median',
                '滑落次数': 'median',
                '最高位置_mm': 'median'
            }).rename(columns={'管号': '样本数', '到达_2/3高度': '到达数'})
            s['到达率_%'] = (s['到达数'] / s['样本数'] * 100).round(1)
            s.to_excel(w, sheet_name='统计')
            
            # 按视频统计
            vs = df.groupby('视频').agg({
                '到达_2/3高度': lambda x: (x=='是').sum(),
                '管号': 'count',
                '速度_中位数_mm/s': 'median',
                '最高位置_mm': 'median'
            }).rename(columns={'管号': '实验管数', '到达_2/3高度': '到达数'})
            vs['到达率_%'] = (vs['到达数'] / vs['实验管数'] * 100).round(1)
            vs.to_excel(w, sheet_name='视频统计')
    
    log.info(f"\n{'='*60}")
    log.info(f"Excel: {xp}")
    log.info(f"样本: {len(df)}条")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='果蝇攀爬行为追踪分析 v20-mod')
    p.add_argument("--dir", required=True, help="视频文件夹路径")
    p.add_argument("--output", default="./output", help="输出目录")
    p.add_argument("--filter", nargs='+', choices=['5mm','10mm','20mm','40mm'],
                   help="仅处理指定管径组")
    a = p.parse_args()
    batch(a.dir, a.output, filter_dias=a.filter)
