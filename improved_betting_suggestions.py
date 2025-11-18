#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改善版買い目提案ロジック
機械学習モデルの確率と選手情報を組み合わせて、期待値の高い買い目を提案
"""

from typing import Dict, List, Any, Tuple
import itertools


def rank_riders_by_strength(riders: List[Dict[str, Any]]) -> List[Tuple[int, float, Dict]]:
    """
    選手を強さ順にランク付け
    競走得点を最重視し、級班・脚質でボーナス
    """
    ranked = []

    for i, rider in enumerate(riders):
        score = rider.get('avg_score', 0) or 0
        grade = rider.get('grade', '').upper()
        style = rider.get('style', '')

        # Base strength = 競走得点
        strength = float(score)

        # Grade bonus
        grade_bonus = {
            'SS': 10,
            'S1': 7,
            'S2': 4,
            'A1': 2,
            'A2': 1,
            'A3': 0,
            'L1': 5,
        }.get(grade, 0)
        strength += grade_bonus

        # Style bonus (逃げ is slightly advantageous)
        if '逃' in style:
            strength += 2
        elif '両' in style:
            strength += 1

        car_no = i + 1
        ranked.append((car_no, strength, rider))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def calculate_upset_potential(probability: float, ranked: List[Tuple[int, float, Dict]]) -> str:
    """
    荒れる可能性に応じた戦略を決定
    """
    if probability >= 0.6:
        return "穴狙い（大荒れ期待）"
    elif probability >= 0.45:
        return "中穴狙い（バランス型）"
    elif probability >= 0.3:
        return "堅め軸穴流し"
    else:
        return "本命勝負"


def generate_trifecta_combinations(
    probability: float,
    ranked: List[Tuple[int, float, Dict]],
    budget: int = 2000
) -> List[Dict[str, Any]]:
    """
    三連単の買い目を生成（期待値重視）
    """
    if len(ranked) < 3:
        return []

    top3 = [r[0] for r in ranked[:3]]
    top5 = [r[0] for r in ranked[:min(5, len(ranked))]]
    mid = [r[0] for r in ranked[3:min(7, len(ranked))]]

    suggestions = []

    if probability >= 0.6:
        # 大荒れ期待 - 穴を絡める
        # パターン1: 中穴を3着に
        if mid:
            for third in mid[:2]:
                suggestions.append({
                    'combination': f'{top3[0]}-{top3[1]}-{third}',
                    'type': '本命軸・穴3着',
                    'points': 2,
                    'expected_payout': '中',
                })
                suggestions.append({
                    'combination': f'{top3[0]}-{third}-{top3[1]}',
                    'type': '本命1着・穴2着',
                    'points': 2,
                    'expected_payout': '中',
                })

        # パターン2: 上位3名のボックス
        for combo in itertools.permutations(top3, 3):
            suggestions.append({
                'combination': f'{combo[0]}-{combo[1]}-{combo[2]}',
                'type': '上位3名ボックス',
                'points': 1,
                'expected_payout': '低',
            })

        # パターン3: 大穴（本命固定、穴を2-3着）
        if len(mid) >= 2:
            for second, third in itertools.permutations(mid[:3], 2):
                suggestions.append({
                    'combination': f'{top3[0]}-{second}-{third}',
                    'type': '大穴狙い',
                    'points': 1,
                    'expected_payout': '高',
                })

    elif probability >= 0.45:
        # 中穴狙い - バランス型
        # パターン1: 本命1着固定、2-3着流し
        for second, third in itertools.permutations(top5[1:], 2):
            suggestions.append({
                'combination': f'{top3[0]}-{second}-{third}',
                'type': '本命1着固定',
                'points': 2,
                'expected_payout': '中',
            })

        # パターン2: 上位2名軸
        for first, second in [(top3[0], top3[1]), (top3[1], top3[0])]:
            for third in top5[2:]:
                suggestions.append({
                    'combination': f'{first}-{second}-{third}',
                    'type': '上位2名軸',
                    'points': 1,
                    'expected_payout': '低',
                })

    elif probability >= 0.3:
        # 堅め軸穴流し
        # パターン1: 本命1着固定
        for second, third in itertools.permutations(top3[1:], 2):
            suggestions.append({
                'combination': f'{top3[0]}-{second}-{third}',
                'type': '本命1着固定',
                'points': 3,
                'expected_payout': '低',
            })

        # パターン2: 上位3名ボックス
        for combo in itertools.permutations(top3, 3):
            suggestions.append({
                'combination': f'{combo[0]}-{combo[1]}-{combo[2]}',
                'type': '上位3名ボックス',
                'points': 2,
                'expected_payout': '低',
            })

    else:
        # 本命勝負 - 堅いレース
        # 上位3名のボックス（重点）
        for combo in itertools.permutations(top3, 3):
            suggestions.append({
                'combination': f'{combo[0]}-{combo[1]}-{combo[2]}',
                'type': '上位3名ボックス',
                'points': 3,
                'expected_payout': '低',
            })

        # 本命1着固定（保険）
        for second, third in itertools.permutations(top3[1:], 2):
            suggestions.append({
                'combination': f'{top3[0]}-{second}-{third}',
                'type': '本命1着固定',
                'points': 2,
                'expected_payout': '低',
            })

    # 重複削除と点数集計
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

    # 予算に応じて絞り込み（1点=100円想定）
    max_points = budget // 100
    cumulative = 0
    filtered = []
    for sug in final_suggestions:
        if cumulative + sug['points'] <= max_points:
            filtered.append(sug)
            cumulative += sug['points']

    return filtered[:20]  # Max 20 combinations


def generate_trio_combinations(
    probability: float,
    ranked: List[Tuple[int, float, Dict]],
) -> List[Dict[str, Any]]:
    """
    三連複の買い目を生成（的中率重視）
    """
    if len(ranked) < 3:
        return []

    top3 = [r[0] for r in ranked[:3]]
    top5 = [r[0] for r in ranked[:min(5, len(ranked))]]

    suggestions = []

    # 三連複は順不同なので組み合わせ数が少ない
    if probability >= 0.5:
        # 中穴を絡める
        for combo in itertools.combinations(top5, 3):
            suggestions.append({
                'combination': f'{combo[0]}-{combo[1]}-{combo[2]}',
                'type': '上位5名から3連複',
                'points': 1,
            })
    else:
        # 堅め
        suggestions.append({
            'combination': f'{top3[0]}-{top3[1]}-{top3[2]}',
            'type': '上位3名の3連複',
            'points': 3,
        })

        # 上位4名から
        if len(ranked) >= 4:
            top4 = [r[0] for r in ranked[:4]]
            for combo in itertools.combinations(top4, 3):
                if combo != tuple(top3):
                    suggestions.append({
                        'combination': f'{combo[0]}-{combo[1]}-{combo[2]}',
                        'type': '上位4名から3連複',
                        'points': 1,
                    })

    return suggestions[:10]  # Max 10


def generate_betting_suggestions(
    race_info: Dict[str, Any],
    probability: float,
    confidence: str,
    budget: int = 2000
) -> Dict[str, Any]:
    """
    機械学習の確率を使った総合的な買い目提案
    """
    riders = race_info.get('riders', [])

    if len(riders) < 3:
        return {
            'error': '選手が3名未満のため買い目を生成できません'
        }

    # 選手をランク付け
    ranked = rank_riders_by_strength(riders)

    # 戦略決定
    strategy = calculate_upset_potential(probability, ranked)

    # 三連単の買い目
    trifecta = generate_trifecta_combinations(probability, ranked, budget)
    trifecta_points = sum(s['points'] for s in trifecta)

    # 三連複の買い目（オプション）
    trio = generate_trio_combinations(probability, ranked)
    trio_points = sum(s['points'] for s in trio)

    # 選手ランキング情報
    rider_ranking = []
    for car_no, strength, rider in ranked:
        rider_ranking.append({
            'car_no': car_no,
            'name': rider.get('name', ''),
            'strength': strength,
            'grade': rider.get('grade', ''),
            'style': rider.get('style', ''),
            'avg_score': rider.get('avg_score'),
        })

    return {
        'strategy': strategy,
        'probability': probability,
        'confidence': confidence,
        'trifecta': {
            'suggestions': trifecta,
            'total_points': trifecta_points,
            'total_cost': trifecta_points * 100,
        },
        'trio': {
            'suggestions': trio,
            'total_points': trio_points,
            'total_cost': trio_points * 100,
        },
        'rider_ranking': rider_ranking,
        'summary': f'{strategy}・三連単{trifecta_points}点（{trifecta_points * 100}円）',
        'recommendation': _get_recommendation(probability, trifecta_points, trio_points),
    }


def _get_recommendation(probability: float, trifecta_points: int, trio_points: int) -> str:
    """推奨アクション"""
    if probability >= 0.6:
        return f"大荒れ期待！三連単で穴狙いがおすすめ。予算: {trifecta_points * 100}円"
    elif probability >= 0.45:
        return f"中穴狙いでバランス良く。三連単{trifecta_points}点または三連複{trio_points}点"
    elif probability >= 0.3:
        return f"堅めのレース。本命軸で手堅く。予算: {trifecta_points * 100}円"
    else:
        return f"本命勝負推奨。上位3名のボックスで堅実に。"


if __name__ == "__main__":
    # Test
    import json

    test_race = {
        'riders': [
            {'name': '選手A', 'prefecture': '東京', 'grade': 'S1', 'style': '逃げ', 'avg_score': 115.2},
            {'name': '選手B', 'prefecture': '神奈川', 'grade': 'S1', 'style': '追込', 'avg_score': 113.5},
            {'name': '選手C', 'prefecture': '埼玉', 'grade': 'S2', 'style': '逃げ', 'avg_score': 110.8},
            {'name': '選手D', 'prefecture': '千葉', 'grade': 'S2', 'style': '追込', 'avg_score': 109.2},
            {'name': '選手E', 'prefecture': '大阪', 'grade': 'A1', 'style': '両', 'avg_score': 108.5},
            {'name': '選手F', 'prefecture': '福岡', 'grade': 'A1', 'style': '追込', 'avg_score': 107.1},
            {'name': '選手G', 'prefecture': '愛知', 'grade': 'A2', 'style': '逃げ', 'avg_score': 105.8},
        ]
    }

    result = generate_betting_suggestions(test_race, probability=0.55, confidence='medium', budget=2000)
    print(json.dumps(result, ensure_ascii=False, indent=2))
