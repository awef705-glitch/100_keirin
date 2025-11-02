#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FastAPI web app for the pre-race high-payout predictor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from analysis import prerace_model


app = FastAPI(title="競輪 高配当予測ツール")

TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>競輪 高配当予測</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f7f7fb;
      --card: #ffffff;
      --primary: #2b59c3;
      --accent: #f97316;
      --text: #1f2933;
      --muted: #6b7280;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .page {
      max-width: 960px;
      margin: 0 auto;
      padding: 24px 16px 72px;
    }
    header {
      text-align: center;
      margin-bottom: 24px;
    }
    header h1 {
      font-size: 1.8rem;
      margin: 0;
    }
    header p {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 0.95rem;
      line-height: 1.6;
    }
    form {
      background: var(--card);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 12px 30px rgba(43,89,195,0.08);
    }
    .section-title {
      font-weight: 600;
      font-size: 1.05rem;
      margin: 24px 0 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .section-title span {
      display: inline-flex;
      width: 26px;
      height: 26px;
      border-radius: 50%;
      background: rgba(43,89,195,0.15);
      color: var(--primary);
      justify-content: center;
      align-items: center;
      font-size: 0.85rem;
    }
    label {
      display: block;
      font-size: 0.9rem;
      font-weight: 600;
      margin-bottom: 6px;
    }
    input[type="text"],
    input[type="date"],
    input[type="number"],
    select,
    textarea {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid #dfe3f0;
      border-radius: 12px;
      font-size: 0.95rem;
      background: #fff;
      color: var(--text);
    }
    textarea {
      min-height: 96px;
      resize: vertical;
    }
    input[type="checkbox"] {
      transform: scale(1.15);
      margin-right: 8px;
    }
    .grid {
      display: grid;
      gap: 16px;
    }
    .grid.two {
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    }
    .checkbox-group {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 8px;
    }
    .checkbox-item {
      display: flex;
      align-items: center;
      background: rgba(43,89,195,0.08);
      padding: 8px 12px;
      border-radius: 999px;
    }
    .notes-area {
      margin-top: 18px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      background: rgba(249,115,22,0.12);
      color: var(--accent);
    }
    .rider-card {
      border: 1px solid #e5e9f5;
      border-radius: 14px;
      padding: 14px;
      background: #fff;
      position: relative;
    }
    .rider-card h3 {
      margin: 0 0 12px;
      font-size: 1rem;
    }
    .rider-remove {
      position: absolute;
      top: 10px;
      right: 12px;
      border: none;
      background: rgba(249,115,22,0.12);
      color: var(--accent);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.8rem;
      cursor: pointer;
    }
    .actions {
      margin-top: 28px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .btn-primary {
      background: linear-gradient(135deg, #2b59c3, #4f8ae2);
      border: none;
      color: #fff;
      padding: 14px;
      border-radius: 12px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 10px 25px rgba(43,89,195,0.18);
    }
    .btn-secondary {
      border: 1px dashed #94a3b8;
      background: transparent;
      color: var(--primary);
      padding: 12px;
      border-radius: 12px;
      font-size: 0.95rem;
      cursor: pointer;
    }
    .model-warning {
      background: rgba(249,115,22,0.12);
      color: var(--accent);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 20px;
      text-align: center;
      font-weight: 600;
    }
    footer {
      margin-top: 48px;
      text-align: center;
      font-size: 0.8rem;
      color: var(--muted);
      line-height: 1.6;
    }
    @media (max-width: 600px) {
      header h1 { font-size: 1.5rem; }
      form { padding: 16px; }
    }
  </style>
</head>
<body>
  <div class="page">
    <header>
      <h1>競輪 高配当予測</h1>
      <p>レース前に得られる情報だけで荒れ度合いを推定。スマートフォンからの操作に最適化しています。</p>
    </header>
    {% if not model_ready %}
    <div class="model-warning">
      モデルが未学習です。ターミナルで <code>python analysis/train_prerace_lightgbm.py</code> を実行してください。
    </div>
    {% endif %}

    <form method="post" action="/predict">
      <div class="section-title"><span>1</span>レース情報</div>
      <div class="grid two">
        <div>
          <label for="race_date">レース日</label>
          <input type="date" id="race_date" name="race_date" required>
        </div>
        <div>
          <label for="track">開催場</label>
          <input type="text" id="track" name="track" placeholder="例: 京王閣">
        </div>
        <div>
          <label for="keirin_cd">会場コード</label>
          <input type="text" id="keirin_cd" name="keirin_cd" placeholder="例: 27" required>
        </div>
        <div>
          <label for="race_no">レース番号</label>
          <input type="number" id="race_no" name="race_no" min="1" max="12" value="7" required>
        </div>
        <div>
          <label for="grade">グレード</label>
          <select id="grade" name="grade">
            <option value="">-- 選択 --</option>
            <option value="GP">GP</option>
            <option value="G1">G1</option>
            <option value="G2">G2</option>
            <option value="G3" selected>G3</option>
            <option value="F1">F1</option>
            <option value="F2">F2</option>
            <option value="L">L</option>
          </select>
        </div>
      </div>
      <div class="checkbox-group">
        <label class="checkbox-item">
          <input type="checkbox" name="is_first_day"> 初日
        </label>
        <label class="checkbox-item">
          <input type="checkbox" name="is_second_day"> 2日目
        </label>
        <label class="checkbox-item">
          <input type="checkbox" name="is_final_day"> 最終日
        </label>
      </div>

      <div class="section-title" style="margin-top:32px;"><span>2</span>追加コンディション（任意）</div>
      <details style="background: rgba(43,89,195,0.06); padding: 16px; border-radius: 14px;">
        <summary style="cursor:pointer; font-weight:600;">開くと天候やバンク状態などを入力できます（空欄でも予測可能）</summary>
        <div class="grid two" style="margin-top:16px;">
          <div>
            <label for="meeting_day">開催日程</label>
            <select id="meeting_day" name="meeting_day">
              <option value="">-- 未選択 --</option>
              <option value="1">1日目（初日）</option>
              <option value="2">2日目</option>
              <option value="3">3日目</option>
              <option value="4">4日目</option>
              <option value="5">5日目</option>
              <option value="6">6日目</option>
            </select>
          </div>
          <div>
            <label for="weather_condition">天候</label>
            <select id="weather_condition" name="weather_condition">
              <option value="">-- 選択 --</option>
              <option value="晴れ">晴れ</option>
              <option value="曇り">曇り</option>
              <option value="雨">雨</option>
              <option value="豪雨">豪雨</option>
              <option value="雪">雪</option>
            </select>
          </div>
          <div>
            <label for="track_condition">バンク状態</label>
            <select id="track_condition" name="track_condition">
              <option value="">-- 選択 --</option>
              <option value="良">良</option>
              <option value="やや重">やや重</option>
              <option value="重">重</option>
            </select>
          </div>
          <div>
            <label for="temperature">気温 (℃)</label>
            <input type="number" step="0.1" id="temperature" name="temperature" placeholder="例: 18.5">
          </div>
          <div>
            <label for="wind_speed">風速 (m/s)</label>
            <input type="number" step="0.1" id="wind_speed" name="wind_speed" placeholder="例: 6.0">
          </div>
          <div>
            <label for="wind_direction">風向</label>
            <input type="text" id="wind_direction" name="wind_direction" placeholder="例: 向かい風">
          </div>
        </div>
        <div class="checkbox-group" style="margin-top:12px;">
          <label class="checkbox-item">
            <input type="checkbox" name="is_night_race"> ナイター開催
          </label>
        </div>
        <div class="notes-area">
          <label for="notes">気になるメモ</label>
          <textarea id="notes" name="notes" placeholder="例: 地元ラインが主導権／当日は雨模様など"></textarea>
        </div>
      </details>

      <div class="section-title" style="margin-top:32px;"><span>3</span>選手情報</div>
      <p class="pill">最大9名まで入力可能。空欄は自動的に除外されます。</p>
      <div id="rider-list" class="grid" style="gap:18px; margin-top:18px;"></div>

      <div class="actions">
        <button type="button" class="btn-secondary" id="add-rider">＋ 選手を追加</button>
        <button type="submit" class="btn-primary" {% if not model_ready %}disabled{% endif %}>予測を実行する</button>
      </div>
    </form>

    <footer>
      研究・教育目的のツールです。投票は自己責任で行ってください。<br>
      モデルは過去データに基づく統計的推定であり、結果を保証するものではありません。
    </footer>
  </div>

  <template id="rider-template">
    <div class="rider-card">
      <button type="button" class="rider-remove">削除</button>
      <h3>選手<span class="rider-index"></span></h3>
      <div class="grid two">
        <div>
          <label>名前</label>
          <input type="text" name="rider_names" placeholder="例: 山田太郎">
        </div>
        <div>
          <label>府県</label>
          <input type="text" name="rider_prefectures" placeholder="例: 東京">
        </div>
        <div>
          <label>階級</label>
          <select name="rider_grades">
            <option value="">--</option>
            <option value="SS">SS</option>
            <option value="S1" selected>S1</option>
            <option value="S2">S2</option>
            <option value="A1">A1</option>
            <option value="A2">A2</option>
            <option value="A3">A3</option>
            <option value="L1">L1</option>
          </select>
        </div>
        <div>
          <label>脚質</label>
          <select name="rider_styles">
            <option value="逃">逃げ・先行</option>
            <option value="追">追い込み</option>
            <option value="両">両方自在</option>
          </select>
        </div>
        <div>
          <label>得点</label>
          <input type="number" step="0.01" name="rider_scores" placeholder="例: 109.3">
        </div>
      </div>
    </div>
  </template>

  <script>
    const riderList = document.getElementById("rider-list");
    const riderTemplate = document.getElementById("rider-template");
    const addRiderBtn = document.getElementById("add-rider");
    const MAX_RIDERS = 9;

    function updateIndices() {
      riderList.querySelectorAll(".rider-card").forEach((card, idx) => {
        const label = card.querySelector(".rider-index");
        if (label) {
          label.textContent = idx + 1;
        }
        const removeButton = card.querySelector(".rider-remove");
        removeButton.style.display = riderList.children.length > 1 ? "inline-flex" : "none";
      });
    }

    function addRider(initial = {}) {
      if (riderList.children.length >= MAX_RIDERS) {
        alert("入力できるのは最大9名です。");
        return;
      }
      const node = riderTemplate.content.cloneNode(true);
      const card = node.querySelector(".rider-card");
      card.querySelector('input[name="rider_names"]').value = initial.name || "";
      card.querySelector('input[name="rider_prefectures"]').value = initial.prefecture || "";
      card.querySelector('select[name="rider_grades"]').value = initial.grade || "S1";
      card.querySelector('select[name="rider_styles"]').value = initial.style || "逃";
      card.querySelector('input[name="rider_scores"]').value = initial.avg_score || "";
      card.querySelector(".rider-remove").addEventListener("click", () => {
        card.remove();
        updateIndices();
      });
      riderList.appendChild(node);
      updateIndices();
    }

    addRiderBtn.addEventListener("click", () => addRider());
    for (let i = 0; i < 7; i++) {
      addRider();
    }
  </script>
</body>
</html>
"""

RESULT_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>競輪 高配当予測 : 結果</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f0f3ff;
      --card: #ffffff;
      --primary: #2b59c3;
      --accent: #f97316;
      --text: #1f2933;
      --muted: #6b7280;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #dde7ff, #f6f7fb 45%);
      color: var(--text);
    }
    .page {
      max-width: 960px;
      margin: 0 auto;
      padding: 32px 18px 80px;
    }
    .score-card {
      background: linear-gradient(135deg, #2b59c3, #4f8ae2);
      border-radius: 18px;
      padding: 24px;
      color: #fff;
      text-align: center;
      box-shadow: 0 18px 48px rgba(43,89,195,0.28);
    }
    .score-card h2 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.08em;
      opacity: 0.85;
    }
    .probability {
      font-size: 3.2rem;
      margin: 12px 0 6px;
      font-weight: 700;
    }
    .confidence {
      font-size: 1rem;
      opacity: 0.85;
    }
    .recommendation {
      margin-top: 12px;
      display: inline-block;
      padding: 6px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.18);
      font-size: 0.9rem;
    }
    .risk-chip {
      display: inline-flex;
      align-items: center;
      padding: 6px 14px;
      border-radius: 999px;
      background: rgba(249,115,22,0.18);
      color: #fff;
      font-size: 0.85rem;
      margin-bottom: 8px;
      font-weight: 600;
    }
    .section {
      margin-top: 26px;
      background: var(--card);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 12px 32px rgba(43,89,195,0.08);
    }
    .section h3 {
      margin: 0 0 12px;
      font-size: 1.02rem;
    }
    .reasons {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .reasons li {
      padding: 12px;
      border-radius: 12px;
      background: rgba(43,89,195,0.08);
      margin-bottom: 10px;
      line-height: 1.6;
    }
    .ticket-list {
      display: grid;
      gap: 12px;
      margin: 16px 0 12px;
    }
    .ticket-item {
      background: rgba(43,89,195,0.08);
      border-radius: 12px;
      padding: 14px;
    }
    .ticket-item strong {
      display: block;
      margin-bottom: 6px;
    }
    .money-note {
      margin-top: 10px;
      font-size: 0.9rem;
      line-height: 1.6;
    }
    .summary-grid,
    .condition-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit,minmax(150px,1fr));
      gap: 12px;
    }
    .summary-item,
    .condition-item {
      border: 1px solid #e5e9f5;
      border-radius: 12px;
      padding: 12px;
      background: rgba(43,89,195,0.04);
    }
    .summary-item span,
    .condition-item span {
      display: block;
      font-size: 0.7rem;
      color: var(--muted);
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }
    th, td {
      border-bottom: 1px solid #e5e9f5;
      padding: 10px;
      text-align: left;
    }
    th {
      background: rgba(43,89,195,0.08);
      font-weight: 600;
    }
    .btn-back {
      margin-top: 28px;
      display: inline-block;
      padding: 12px 20px;
      border-radius: 12px;
      border: 1px solid #94a3b8;
      text-decoration: none;
      color: var(--primary);
      font-weight: 600;
    }
    .warning {
      margin-top: 28px;
      font-size: 0.85rem;
      color: var(--muted);
      line-height: 1.6;
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="score-card">
      <h2>{{ race.track or '未入力' }} {{ race.race_no }}R / {{ race.race_date }}</h2>
      <div class="probability">{{ '%.0f'|format(probability * 100) }}%</div>
      <div class="confidence">
        信頼度: {{ result.confidence }} ｜ グレード: {{ race.grade or '-' }}
      </div>
      <p style="margin:8px 0 0;font-size:0.95rem;">
        → このレースが「三連単で1万円超」の高配当になる確率は <strong>{{ '%.0f'|format(probability * 100) }}%</strong> です。
      </p>
      <div class="recommendation">{{ result.recommendation }}</div>
    </div>

    <div class="section">
      <h3>推奨ベットプラン</h3>
      <div class="risk-chip">
        {{ result.betting_plan.risk_level }} ｜ 指標 {{ '%.0f'|format(result.betting_plan.effective_score * 100) }}%
      </div>
      <p>{{ result.betting_plan.plan_summary }}</p>
      <div class="ticket-list">
        {% for ticket in result.betting_plan.ticket_plan %}
        <div class="ticket-item">
          <strong>{{ ticket.label }}</strong>
          <p>{{ ticket.description }}</p>
        </div>
        {% endfor %}
      </div>
      <div class="money-note">
        <strong>資金配分</strong><br>
        {{ result.betting_plan.money_management }}
      </div>
      <div class="money-note">
        <strong>ヘッジのヒント</strong><br>
        {{ result.betting_plan.hedge_note }}
      </div>
    </div>

    <div class="section">
      <h3>予測ロジックのハイライト</h3>
      <ul class="reasons">
        {% for reason in result.reasons %}
        <li>{{ reason }}</li>
        {% endfor %}
      </ul>
    </div>

    <div class="section">
      <h3>レースコンディション</h3>
      <div class="condition-grid">
        <div class="condition-item">
          <span>天候</span>
          {{ summary.get('weather_condition', '') or race.weather_condition or '未入力' }}
        </div>
        <div class="condition-item">
          <span>バンク状態</span>
          {{ summary.get('track_condition', '') or race.track_condition or '未入力' }}
        </div>
        <div class="condition-item">
          <span>気温</span>
          {% if summary.get('temperature_c') is not none %}
            {{ '%.1f℃'|format(summary.get('temperature_c')) }}
          {% elif race.temperature %}
            {{ race.temperature }}℃
          {% else %}
            未入力
          {% endif %}
        </div>
        <div class="condition-item">
          <span>風速 / 風向</span>
          {% if summary.get('wind_speed_mps') is not none %}
            {{ '%.1f m/s'|format(summary.get('wind_speed_mps')) }} {{ summary.get('wind_direction') or '' }}
          {% elif race.wind_speed %}
            {{ race.wind_speed }} m/s {{ race.wind_direction or '' }}
          {% else %}
            未入力
          {% endif %}
        </div>
        <div class="condition-item">
          <span>開催日程</span>
          {% if summary.get('meeting_day') %}
            {{ summary.get('meeting_day') }}日目
          {% else %}
            {{ race.meeting_day or '未入力' }}
          {% endif %}
        </div>
        <div class="condition-item">
          <span>ナイター</span>
          {% if summary.get('is_night_race') %}
            ナイター
          {% elif race.is_night_race %}
            ナイター
          {% else %}
            昼開催
          {% endif %}
        </div>
      </div>
      {% if race.notes %}
      <div class="money-note" style="margin-top:14px;">
        <strong>メモ</strong><br>{{ race.notes }}
      </div>
      {% endif %}
    </div>

    <div class="section">
      <h3>特徴量サマリ</h3>
      <div class="summary-grid">
        <div class="summary-item">
          <span>平均得点</span>
          {{ '%.2f'|format(summary.score_mean or 0.0) }}
        </div>
        <div class="summary-item">
          <span>得点レンジ</span>
          {{ '%.2f'|format(summary.score_range or 0.0) }}
        </div>
        <div class="summary-item">
          <span>得点CV</span>
          {{ '%.2f'|format(summary.score_cv or 0.0) }}
        </div>
        <div class="summary-item">
          <span>脚質多様性</span>
          {{ '%.2f'|format(summary.style_diversity or 0.0) }}
        </div>
        <div class="summary-item">
          <span>脚質構成</span>
          {% set ratios = summary.get('style_ratios', {}) %}
          {% if ratios %}
            {% for key, value in ratios.items() %}
              {{ key }} {{ '%.0f'|format(value * 100) }}%{% if not loop.last %} / {% endif %}
            {% endfor %}
          {% else %}
            -
          {% endif %}
        </div>
        <div class="summary-item">
          <span>地元エリア数</span>
          {{ summary.prefecture_unique_count }}
        </div>
      </div>
    </div>

    <div class="section">
      <h3>入力した選手一覧</h3>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>名前</th>
            <th>府県</th>
            <th>階級</th>
            <th>脚質</th>
            <th>得点</th>
          </tr>
        </thead>
        <tbody>
          {% for rider in riders %}
          <tr>
            <td>{{ loop.index }}</td>
            <td>{{ rider.name }}</td>
            <td>{{ rider.prefecture }}</td>
            <td>{{ rider.grade }}</td>
            <td>{{ rider.style }}</td>
            <td>
              {% if rider.avg_score is not none %}
                {{ '%.2f'|format(rider.avg_score) }}
              {% else %}
                -
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <a href="/" class="btn-back">↩ 新しいレースを入力する</a>

    <div class="warning">
      免責: 本ツールは過去データに基づく統計的推定です。実際の結果を保証するものではありません。投票は自己責任で行ってください。
    </div>
  </div>
</body>
</html>
"""

ERROR_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>エラー</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7fb; color: #1f2933; margin: 0; }
    .page { max-width: 720px; margin: 0 auto; padding: 80px 16px; text-align: center; }
    .card { background: #fff; border-radius: 16px; padding: 32px; box-shadow: 0 18px 45px rgba(43,89,195,0.12); }
    h1 { font-size: 1.6rem; margin: 0 0 16px; }
    p { color: #64748b; line-height: 1.7; }
    a { display: inline-block; margin-top: 24px; padding: 12px 24px; border-radius: 12px; border: 1px solid #94a3b8; color: #2b59c3; text-decoration: none; font-weight: 600; }
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h1>⚠ モデルが見つかりません</h1>
      <p>{{ message }}</p>
      <a href="/">↩ 入力フォームに戻る</a>
    </div>
  </div>
</body>
</html>
"""

# Load model artefacts once at startup.
try:
    MODEL = prerace_model.load_model()
    METADATA = prerace_model.load_metadata()
    MODEL_READY = True
except FileNotFoundError:
    MODEL = None
    METADATA = {}
    MODEL_READY = False


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    create_templates()
    context = {"request": request, "model_ready": MODEL_READY}
    return templates.TemplateResponse("index.html", context)


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    race_date: str = Form(...),
    track: str = Form(""),
    keirin_cd: str = Form(...),
    race_no: int = Form(...),
    grade: str = Form(""),
    meeting_day: Optional[str] = Form(None),
    is_first_day: Optional[str] = Form(None),
    is_second_day: Optional[str] = Form(None),
    is_final_day: Optional[str] = Form(None),
    weather_condition: str = Form(""),
    track_condition: str = Form(""),
    temperature: Optional[str] = Form(None),
    wind_speed: Optional[str] = Form(None),
    wind_direction: str = Form(""),
    is_night_race: Optional[str] = Form(None),
    notes: str = Form(""),
    rider_names: List[str] = Form([]),
    rider_prefectures: List[str] = Form([]),
    rider_grades: List[str] = Form([]),
    rider_styles: List[str] = Form([]),
    rider_scores: List[str] = Form([]),
) -> HTMLResponse:
    create_templates()

    if not MODEL_READY:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "モデルがまだ訓練されていません。analysis/train_prerace_lightgbm.py を実行してください。",
            },
        )

    race_date_digits = race_date.replace("-", "")
    riders = []
    for name, pref, grade_val, style, score in zip(
        rider_names, rider_prefectures, rider_grades, rider_styles, rider_scores
    ):
        if not name.strip():
            continue
        try:
            avg_score = float(score) if score else None
        except ValueError:
            avg_score = None
        riders.append(
            {
                "name": name.strip(),
                "prefecture": pref.strip(),
                "grade": grade_val.strip().upper(),
                "style": style.strip(),
                "avg_score": avg_score,
            }
        )

    race_info = {
        "race_date": race_date_digits,
        "track": track.strip(),
        "keirin_cd": keirin_cd.strip().zfill(2),
        "race_no": race_no,
        "grade": grade.strip().upper(),
        "meeting_day": (meeting_day or "").strip(),
        "is_first_day": is_first_day == "on",
        "is_second_day": is_second_day == "on",
        "is_final_day": is_final_day == "on",
        "weather_condition": weather_condition.strip(),
        "track_condition": track_condition.strip(),
        "temperature": (temperature or "").strip(),
        "wind_speed": (wind_speed or "").strip(),
        "wind_direction": wind_direction.strip(),
        "is_night_race": is_night_race == "on",
        "notes": notes.strip(),
        "riders": riders,
    }

    bundle = prerace_model.build_manual_feature_row(race_info)
    feature_frame, summary = prerace_model.align_features(bundle, METADATA["feature_columns"])
    probability = prerace_model.predict_probability(feature_frame, MODEL, METADATA)
    result = prerace_model.build_prediction_response(probability, summary, METADATA)

    context = {
        "request": request,
        "race": race_info,
        "probability": probability,
        "result": result,
        "riders": riders,
        "summary": summary,
    }
    return templates.TemplateResponse("result.html", context)


def create_templates() -> None:
    """Write template files if they do not exist."""

    index_html = TEMPLATES_DIR / "index.html"
    if not index_html.exists():
        index_html.write_text(INDEX_TEMPLATE, encoding="utf-8")

    result_html = TEMPLATES_DIR / "result.html"
    if not result_html.exists():
        result_html.write_text(RESULT_TEMPLATE, encoding="utf-8")

    error_html = TEMPLATES_DIR / "error.html"
    if not error_html.exists():
        error_html.write_text(ERROR_TEMPLATE, encoding="utf-8")


if __name__ == "__main__":
    create_templates()
    print("=" * 70)
    print("競輪 高配当予測 Web アプリ")
    print("=" * 70)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"スタート: http://{host}:{port}")
    print("スマホからアクセスするには PC と同じネットワークに接続し、PC の IP を指定してください。")
    print("（例）http://<PCのIPアドレス>:{port}/")
    print("終了するには Ctrl+C を押してください。")
    print("=" * 70)
    uvicorn.run(app, host=host, port=port)
