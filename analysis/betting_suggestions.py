#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""具体的な買い目提案を生成するモジュール (Tiered Strategy)"""

from typing import Dict, List, Any, Tuple
import itertools


def calculate_rider_strength(rider: Dict[str, Any], index: int) -> float:
    """選手の強さを評価（スコア化）"""
    score = 100.0  # Base score

    # 得点による評価
    avg_score = rider.get('avg_score')
    if avg_score:
        score += (avg_score - 100) * 2  # 得点差を2倍で加算

    # 階級による評価
    grade = rider.get('grade', '').upper()
    grade_bonus = {
        'SS': 20,
        'S1': 15,
        'S2': 10,
        'A1': 5,
        'A2': 2,
        'A3': 0,
        'L1': 10,
    }.get(grade, 0)
    score += grade_bonus

    # 脚質による評価（逃げは有利）
    style = rider.get('style', '')
    if '逃' in style:
        score += 5
    elif '両' in style:
        score += 3
        
    # B回数（バック回数）による評価
    back_count = rider.get('back_count', 0)
    if back_count:
        try:
            bc = float(back_count)
            if bc >= 20:
                score += 5
            elif bc >= 10:
                score += 3
            elif bc >= 5:
                score += 1
        except (ValueError, TypeError):
            pass
    
    # 戦術履歴による評価
    nige_count = rider.get('nige_count', 0)
    if nige_count:
        try:
            nc = float(nige_count)
            if nc >= 10: score += 4
            elif nc >= 5: score += 2
        except (ValueError, TypeError): pass
    
    makuri_count = rider.get('makuri_count', 0)
    if makuri_count:
        try:
            mc = float(makuri_count)
            if mc >= 10: score += 6
            elif mc >= 5: score += 3
        except (ValueError, TypeError): pass
    
    sasi_count = rider.get('sasi_count', 0)
    if sasi_count:
        try:
            sc = float(sasi_count)
            if sc >= 10: score += 5
            elif sc >= 5: score += 2
        except (ValueError, TypeError): pass

    # Recent Win Rate
    recent_win_rate = rider.get('recent_win_rate', 0.0)
    if recent_win_rate:
        try:
            wr = float(recent_win_rate)
            if wr >= 0.3: score += 5
            elif wr >= 0.1: score += 2
        except (ValueError, TypeError): pass

    # Gear Ratio (Higher gear = more power/makuri potential?)
    gear = rider.get('gear_ratio', 0.0)
    if gear:
        try:
            g = float(gear)
            if g >= 3.92: score += 2 # Slight bonus for heavy gear
        except (ValueError, TypeError): pass

    # H/S Count (Active racer)
    hs = rider.get('hs_count', 0) # Could be string "H:1 S:2" or float
    # If float/int
    if isinstance(hs, (int, float)) and hs > 0:
         score += 2
         if hs >= 5: score += 3
    
    return score

def calculate_rider_strength_v2(rider: Dict[str, Any], index: int, track_name: str = None) -> float:
    score = calculate_rider_strength(rider, index)
    
    # Home Bank Bonus
    home_bank = rider.get('home_bank')
    # If home_bank is explicitly 1 (int/str), it means "Yes" (already checked by caller)
    # If it's a string name, check if it matches track_name
    is_home = False
    if str(home_bank) == "1":
        is_home = True
    elif isinstance(home_bank, str) and track_name and home_bank in track_name:
        is_home = True
    elif isinstance(home_bank, str) and track_name and track_name in home_bank:
        is_home = True
        
    if is_home:
        score += 5.0
        
    return score


def rank_riders(riders: List[Dict[str, Any]], track_name: str = None) -> List[Tuple[int, float, Dict[str, Any]]]:
    """選手を強さ順にランク付け"""
    ranked = []
    for i, rider in enumerate(riders):
        strength = calculate_rider_strength_v2(rider, i, track_name)
        car_no = i + 1  # 車番は1から
        ranked.append((car_no, strength, rider))

    # 強さ順にソート
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def generate_tiered_suggestions(
    race_info: Dict[str, Any],
    roughness_score: float,
    confidence: str
) -> Dict[str, Any]:
    """
    荒れ度スコアに基づき、松・竹・梅の3パターンの買い目を提案する。
    """
    if not isinstance(roughness_score, (int, float)):
        try:
            roughness_score = float(roughness_score)
        except:
            roughness_score = 50.0

    riders = race_info.get('riders', [])
    track_name = race_info.get('track', '')
    
    if len(riders) < 3:
        return {'error': '選手が3名未満のため買い目を生成できません'}

    # 選手をランク付け
    ranked = rank_riders(riders, track_name)
    
    # ランク順の車番リスト
    r_order = [r[0] for r in ranked]
    
    # 上位選手
    top1 = r_order[0]
    top2 = r_order[1]
    top3 = r_order[:3]
    top4 = r_order[:4]
    top5 = r_order[:5]
    top6 = r_order[:6]

    suggestions = {
        "low_cost": [],   # 梅: 少額 (5-10点)
        "mid_cost": [],   # 竹: 中額 (10-30点)
        "high_cost": [],  # 松: 高額 (30-60+点)
        "high_cost_reduced": [] # 松・絞り
    }
    
    strategies = {
        "low_cost": "",
        "mid_cost": "",
        "high_cost": "",
        "high_cost_reduced": ""
    }

    # === ロジック分岐 ===
    
    # 1. 鉄板レース (Score 0-20)
    if roughness_score <= 20:
        # 梅: ワイド1点 (本命-対抗)
        strategies["low_cost"] = "ワイド1点勝負"
        suggestions["low_cost"].append({
            "combination": f"{top1}={top2}", "type": "ワイド", "points": 1
        })
        
        # 竹: 3連単 上位3名BOX (6点)
        strategies["mid_cost"] = "上位3名BOX"
        for p in itertools.permutations(top3, 3):
            suggestions["mid_cost"].append({
                "combination": f"{p[0]}-{p[1]}-{p[2]}", "type": "3連単BOX", "points": 1
            })
            
        # 松: 3連単 1着固定流し (12点)
        strategies["high_cost"] = "本命軸・相手4名"
        others = r_order[1:5] # 2-5位
        for s, t in itertools.permutations(others, 2):
            suggestions["high_cost"].append({
                "combination": f"{top1}-{s}-{t}", "type": "3連単流し", "points": 1
            })

    # 2. 堅い〜標準 (Score 20-60)
    elif roughness_score <= 60:
        # 梅: 2車単 上位3名BOX (6点)
        strategies["low_cost"] = "2車単 上位BOX"
        for p in itertools.permutations(top3, 2):
            suggestions["low_cost"].append({
                "combination": f"{p[0]}-{p[1]}", "type": "2車単BOX", "points": 1
            })
            
        # 竹: 3連単 フォーメーション (12点)
        # 1着: 1,2位 -> 2着: 1,2,3位 -> 3着: 1,2,3,4位
        strategies["mid_cost"] = "本命・対抗フォーメーション"
        w_list = r_order[:2] # 1st and 2nd riders
        s_list = top3
        t_list = top4
        for w in w_list:
            for s in s_list:
                if w == s: continue
                for t in t_list:
                    if t == w or t == s: continue
                    suggestions["mid_cost"].append({
                        "combination": f"{w}-{s}-{t}", "type": "フォーメーション", "points": 1
                    })
                    
        # 松: 3連単 上位4名BOX (24点)
        strategies["high_cost"] = "上位4名BOX"
        for p in itertools.permutations(top4, 3):
            suggestions["high_cost"].append({
                "combination": f"{p[0]}-{p[1]}-{p[2]}", "type": "3連単BOX", "points": 1
            })

    # 3. 波乱含み (Score 60-80)
    elif roughness_score <= 80:
        # 梅: ワイドBOX 上位4名 (6点)
        strategies["low_cost"] = "ワイドBOX"
        for p in itertools.combinations(top4, 2):
            suggestions["low_cost"].append({
                "combination": f"{p[0]}={p[1]}", "type": "ワイドBOX", "points": 1
            })
            
        # 竹: 2車単 上位5名BOX (20点)
        strategies["mid_cost"] = "2車単 上位5名BOX"
        for p in itertools.permutations(top5, 2):
            suggestions["mid_cost"].append({
                "combination": f"{p[0]}-{p[1]}", "type": "2車単BOX", "points": 1
            })
            
        # 松: 穴軸マルチ (60点) - 4番人気を軸に手広く
        strategies["high_cost"] = "穴軸マルチ (高配当)"
        axis = top4[3] # Rank 4
        partners = top3 + top6[4:6] # 1,2,3,5,6
        for p1, p2 in itertools.permutations(partners, 2):
            suggestions["high_cost"].append({"combination": f"{axis}-{p1}-{p2}", "type": "穴軸マルチ", "points": 1})
            suggestions["high_cost"].append({"combination": f"{p1}-{axis}-{p2}", "type": "穴軸マルチ", "points": 1})
            suggestions["high_cost"].append({"combination": f"{p1}-{p2}-{axis}", "type": "穴軸マルチ", "points": 1})

        # 松・絞り: 穴軸マルチ・絞り (18点)
        strategies["high_cost_reduced"] = "穴軸マルチ・絞り"
        partners_reduced = top3
        for p1, p2 in itertools.permutations(partners_reduced, 2):
            suggestions["high_cost_reduced"].append({"combination": f"{axis}-{p1}-{p2}", "type": "穴軸マルチ絞", "points": 1})
            suggestions["high_cost_reduced"].append({"combination": f"{p1}-{axis}-{p2}", "type": "穴軸マルチ絞", "points": 1})
            suggestions["high_cost_reduced"].append({"combination": f"{p1}-{p2}-{axis}", "type": "穴軸マルチ絞", "points": 1})

    # 4. 激荒れ (Score 80-100)
    else:
        # 梅: ワイドBOX 上位5名 (10点)
        strategies["low_cost"] = "ワイドBOX広め"
        for p in itertools.combinations(top5, 2):
            suggestions["low_cost"].append({
                "combination": f"{p[0]}={p[1]}", "type": "ワイドBOX", "points": 1
            })
            
        # 竹: 3連複BOX 上位6名 (20点)
        strategies["mid_cost"] = "3連複 上位6名BOX"
        for p in itertools.combinations(top6, 3):
            suggestions["mid_cost"].append({
                "combination": f"{p[0]}={p[1]}={p[2]}", "type": "3連複BOX", "points": 1
            })
            
        # 松: 大穴BOX (60点)
        strategies["high_cost"] = "大穴BOX (超高配当)"
        target_indices = [2, 3, 4, 5, 6]
        chaos_members = []
        for idx in target_indices:
            if idx < len(r_order):
                chaos_members.append(r_order[idx])
        
        if len(chaos_members) >= 3:
            for p in itertools.permutations(chaos_members, 3):
                suggestions["high_cost"].append({
                    "combination": f"{p[0]}-{p[1]}-{p[2]}", "type": "大穴BOX", "points": 1
                })
        
        # 松・絞り: 大穴BOX・絞り (24点)
        strategies["high_cost_reduced"] = "大穴BOX・絞り"
        target_indices_reduced = [2, 3, 4, 5]
        chaos_members_reduced = []
        for idx in target_indices_reduced:
            if idx < len(r_order):
                chaos_members_reduced.append(r_order[idx])
                
        if len(chaos_members_reduced) >= 3:
            for p in itertools.permutations(chaos_members_reduced, 3):
                suggestions["high_cost_reduced"].append({
                    "combination": f"{p[0]}-{p[1]}-{p[2]}", "type": "大穴BOX絞", "points": 1
                })

    # 選手情報を追加
    rider_info = []
    for car_no, strength, rider in ranked:
        rider_info.append({
            'car_no': car_no,
            'name': rider.get('name', ''),
            'strength': strength,
            'grade': rider.get('grade', ''),
            'style': rider.get('style', ''),
            'avg_score': rider.get('avg_score')
        })

    return {
        'roughness_score': roughness_score,
        'confidence': confidence,
        'suggestions': suggestions,
        'strategies': strategies,
        'rider_ranking': rider_info,
    }


def format_betting_suggestions(suggestions_data: Dict[str, Any]) -> str:
    """買い目提案を見やすくフォーマット"""

    if 'error' in suggestions_data:
        return f"エラー: {suggestions_data['error']}"

    output = []
    output.append("=" * 70)
    output.append(f"💰 買い目提案 (荒れ度: {suggestions_data['roughness_score']:.1f})")
    output.append("=" * 70)

    # 選手ランキング
    output.append("【AI評価ランキング】")
    for i, rider in enumerate(suggestions_data['rider_ranking'][:6], 1):
        score_str = f"{rider['avg_score']:.1f}" if rider['avg_score'] else '-'
        output.append(
            f"{i}位: {rider['car_no']}番 {rider['name']} "
            f"({rider['grade']}/{rider['style']}/得点:{score_str}) "
            f"評価:{rider['strength']:.1f}"
        )
    output.append("-" * 70)

    # 松竹梅の提案
    tiers = [
        ("梅 (少額・手堅く)", "low_cost"),
        ("竹 (中額・バランス)", "mid_cost"),
        ("松 (高額・高配当)", "high_cost"),
        ("松・絞り (高配当・厳選)", "high_cost_reduced"),
    ]

    for label, key in tiers:
        sug_list = suggestions_data['suggestions'].get(key, [])
        strategy_name = suggestions_data['strategies'].get(key, "")
        
        if not sug_list and not strategy_name:
            continue
            
        points = len(sug_list)
        cost = points * 100
        
        output.append(f"■ {label}: {strategy_name}")
        output.append(f"   点数: {points}点 (¥{cost:,})")
        
        # 買い目をコンパクトに表示 (最初の5つ + 残り)
        if points > 0:
            preview = [s['combination'] for s in sug_list[:5]]
            preview_str = ", ".join(preview)
            if points > 5:
                preview_str += f" ...他{points-5}点"
            output.append(f"   買い目: {preview_str}")
        else:
            output.append("   (提案なし)")
        output.append("")

    output.append("=" * 70)
    return "\n".join(output)
