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

    # 確率に応じて買い目を変える
    if probability >= 0.7:  # 高確率で荒れる
        strategy = "穴狙い戦略"

        # パターン1: 中穴を絡める
        if len(mid) >= 1:
            for third in mid[:2]:
                suggestions.append({
                    'combination': f'{top3[0]}-{top3[1]}-{third}',
                    'type': '本命軸で中穴を3着に',
                    'points': 1
                })
                suggestions.append({
                    'combination': f'{top3[0]}-{third}-{top3[1]}',
                    'type': '本命1着、穴2着',
                    'points': 1
                })

        # パターン2: 上位で流す
        for combo in itertools.permutations(top3, 3):
            suggestions.append({
                'combination': f'{combo[0]}-{combo[1]}-{combo[2]}',
                'type': '上位3名のボックス',
                'points': 1
            })

        # パターン3: 大穴狙い
        if len(ranked) >= 7:
            dark_horses = [r[0] for r in ranked[5:min(8, len(ranked))]]
            for dark in dark_horses[:2]:
                suggestions.append({
                    'combination': f'{top3[0]}-{dark}-{top3[1]}',
                    'type': '大穴を2着に',
                    'points': 1
                })

    elif probability >= 0.5:  # 中確率
        strategy = "堅め軸穴流し"

        # パターン1: 本命-2,3着流し
        for second, third in itertools.permutations(top5[1:], 2):
            suggestions.append({
                'combination': f'{top3[0]}-{second}-{third}',
                'type': '本命1着固定',
                'points': 2
            })

        # パターン2: 上位2名軸
        for first, second in [(top3[0], top3[1]), (top3[1], top3[0])]:
            for third in top5[2:]:
                suggestions.append({
                    'combination': f'{first}-{second}-{third}',
                    'type': '上位2名軸',
                    'points': 1
                })

    else:  # 低確率（堅い展開）
        strategy = "堅め本命勝負"

        # パターン1: 上位3名のボックス（重点）
        for combo in itertools.permutations(top3, 3):
            suggestions.append({
                'combination': f'{combo[0]}-{combo[1]}-{combo[2]}',
                'type': '上位3名ボックス',
                'points': 3
            })

        # パターン2: 本命1着固定
        for second, third in itertools.permutations(top3[1:], 2):
            suggestions.append({
                'combination': f'{top3[0]}-{second}-{third}',
                'type': '本命1着固定',
                'points': 2
            })

    # 重複を削除して点数を合計
    unique_suggestions = {}
    for sug in suggestions:
        combo = sug['combination']
        if combo in unique_suggestions:
            unique_suggestions[combo]['points'] += sug['points']
        else:
            unique_suggestions[combo] = sug

    # 点数順にソート
    final_suggestions = sorted(
        unique_suggestions.values(),
        key=lambda x: x['points'],
        reverse=True
    )

    # 上位10-15点に絞る
    final_suggestions = final_suggestions[:15]
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
