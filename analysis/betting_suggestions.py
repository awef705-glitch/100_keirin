#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""具体的な買い目提案を生成するモジュール"""

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
    # 積極的な選手は展開を作れるので加点
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
    
    # 戦術履歴による評価（新規追加）
    # 逃げ回数：積極的な展開を作れる
    nige_count = rider.get('nige_count', 0)
    if nige_count:
        try:
            nc = float(nige_count)
            if nc >= 10:
                score += 4
            elif nc >= 5:
                score += 2
        except (ValueError, TypeError):
            pass
    
    # 捲り回数：強力な決め手
    makuri_count = rider.get('makuri_count', 0)
    if makuri_count:
        try:
            mc = float(makuri_count)
            if mc >= 10:
                score += 6  # 捲りは強力
            elif mc >= 5:
                score += 3
        except (ValueError, TypeError):
            pass
    
    # 差し回数：安定した決め手
    sasi_count = rider.get('sasi_count', 0)
    if sasi_count:
        try:
            sc = float(sasi_count)
            if sc >= 10:
                score += 5
            elif sc >= 5:
                score += 2
        except (ValueError, TypeError):
            pass

    return score


def rank_riders(riders: List[Dict[str, Any]]) -> List[Tuple[int, float, Dict[str, Any]]]:
    """選手を強さ順にランク付け"""
    ranked = []
    for i, rider in enumerate(riders):
        strength = calculate_rider_strength(rider, i)
        car_no = i + 1  # 車番は1から
        ranked.append((car_no, strength, rider))

    # 強さ順にソート
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def generate_betting_suggestions(
    race_info: Dict[str, Any],
    probability: float,
    confidence: str
) -> Dict[str, Any]:
    """具体的な買い目を生成"""

    riders = race_info.get('riders', [])
    if len(riders) < 3:
        return {
            'error': '選手が3名未満のため買い目を生成できません'
        }

    # 選手をランク付け
    ranked = rank_riders(riders)

    # 上位3名
    top3 = [r[0] for r in ranked[:3]]
    # 上位5名
    top5 = [r[0] for r in ranked[:min(5, len(ranked))]]
    # 中位（4-6位）
    mid = [r[0] for r in ranked[3:min(6, len(ranked))]]

    suggestions = []
    strategy = ""

    # 新戦略：1着予測精度（50%）を活かす
    # 予測上位を1着に固定し、2-3着を手広く流す
    
    if probability >= 0.5:  # 高確率レース
        strategy = "勝者固定・大流し"
        
        # 上位7名まで使用
        top7 = [r[0] for r in ranked[:min(7, len(ranked))]]
        
        # パターン1: 1着本命（rank 1）固定、2-3着フルカバー（最大42点）
        winner = top3[0]
        others = top7[1:]
        for second, third in itertools.permutations(others, 2):
            suggestions.append({
                'combination': f'{winner}-{second}-{third}',
                'type': '勝者軸・全流し',
                'points': 1
            })
        
        # パターン2: 1着2番手固定、2-3着流し（バックアップ、最大20点）
        if len(top3) >= 2:
            winner2 = top3[1]
            for second, third in itertools.permutations(top5[1:], 2):
                if second != winner2:
                    suggestions.append({
                        'combination': f'{winner2}-{second}-{third}',
                        'type': '2番手軸',
                        'points': 1
                    })

    elif probability >= 0.3:  # 中確率
        strategy = "勝者固定・手堅く流し"
        
        # パターン1: 1着本命固定、2-3着上位5名で流し（最大20点）
        winner = top3[0]
        for second, third in itertools.permutations(top5[1:], 2):
            suggestions.append({
                'combination': f'{winner}-{second}-{third}',
                'type': '勝者軸',
                'points': 2
            })
        
        # パターン2: 上位3名ボックス（保険、6点）
        for combo in itertools.permutations(top3, 3):
            suggestions.append({
                'combination': f'{combo[0]}-{combo[1]}-{combo[2]}',
                'type': '上位BOX',
                'points': 1
            })

    else:  # 低確率（混戦・荒れ予想）
        strategy = "穴狙い・広角流し"
        
        # 上位7名まで使用
        top7 = [r[0] for r in ranked[:min(7, len(ranked))]]
        
        # パターン1: 1着（評価1位）から手広く流す（最大30点）
        winner = top3[0]
        others = top7[1:]
        for second, third in itertools.permutations(others, 2):
            suggestions.append({
                'combination': f'{winner}-{second}-{third}',
                'type': '軸1頭流し',
                'points': 1
            })
        
        # パターン2: 上位4名ボックス（24点）- 混戦用
        # top3 + 4th ranked rider
        box_members = top3 + [ranked[3][0]] if len(ranked) > 3 else top3
        for combo in itertools.permutations(box_members, 3):
            suggestions.append({
                'combination': f'{combo[0]}-{combo[1]}-{combo[2]}',
                'type': '上位BOX',
                'points': 1
            })

    # 重複削除
    seen = set()
    final_suggestions = []
    for s in suggestions:
        combo = s['combination']
        if combo not in seen:
            seen.add(combo)
            final_suggestions.append(s)

    # 点数順にソート
    final_suggestions.sort(key=lambda x: x['points'], reverse=True)

    # 確率に応じて買い目数を調整
    # 低確率（荒れそう）な場合こそ、点数を増やして網を広げる
    # 的中率向上のため、全体的に買い目数を大幅に増加
    if probability >= 0.5:
        max_suggestions = 60  # 超高確率: 60点（フルカバー）
    elif probability >= 0.3:
        max_suggestions = 48  # 中穴: 48点（広めカバー）
    else:
        max_suggestions = 54  # 大穴: 54点（超広角流し）
    
    final_suggestions = final_suggestions[:max_suggestions]
    total_points = sum(s['points'] for s in final_suggestions)

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
        'strategy': strategy,
        'probability': probability,
        'confidence': confidence,
        'suggestions': final_suggestions,
        'total_points': total_points,
        'rider_ranking': rider_info,
        'summary': f'{strategy}で{total_points}点（{len(final_suggestions)}通り）を推奨'
    }


def format_betting_suggestions(suggestions_data: Dict[str, Any]) -> str:
    """買い目提案を見やすくフォーマット"""

    if 'error' in suggestions_data:
        return f"エラー: {suggestions_data['error']}"

    output = []
    output.append("=" * 70)
    output.append("💰 具体的な買い目提案")
    output.append("=" * 70)
    output.append(f"戦略: {suggestions_data['strategy']}")
    output.append(f"荒れる確率: {suggestions_data['probability']:.1%}")
    output.append(f"信頼度: {suggestions_data['confidence']}")
    output.append(f"\n{suggestions_data['summary']}")
    output.append("")

    # 選手ランキング
    output.append("【選手評価ランキング】")
    for i, rider in enumerate(suggestions_data['rider_ranking'][:6], 1):
        score_str = f"{rider['avg_score']:.1f}" if rider['avg_score'] else '-'
        output.append(
            f"{i}位: {rider['car_no']}番 {rider['name']} "
            f"({rider['grade']}/{rider['style']}/得点:{score_str}) "
            f"評価:{rider['strength']:.1f}"
        )
    output.append("")

    # 買い目リスト
    output.append("【推奨買い目】")
    for i, sug in enumerate(suggestions_data['suggestions'], 1):
        output.append(
            f"{i:2d}. {sug['combination']:10s}  "
            f"{sug['points']}点  ({sug['type']})"
        )

    output.append("")
    output.append(f"合計: {suggestions_data['total_points']}点 × 100円 = {suggestions_data['total_points'] * 100}円")
    output.append("=" * 70)

    return "\n".join(output)
