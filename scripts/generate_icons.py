#!/usr/bin/env python3
"""
PWA用のアイコン画像を生成
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """競輪アプリのアイコンを生成"""
    # グラデーション背景を作成
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)

    # グラデーション風の背景（紫系）
    for i in range(size):
        ratio = i / size
        r = int(102 + (118 - 102) * ratio)
        g = int(126 + (75 - 126) * ratio)
        b = int(234 + (162 - 234) * ratio)
        draw.line([(0, i), (size, i)], fill=(r, g, b))

    # 円を描画（トラックのイメージ）
    center = size // 2
    radius = int(size * 0.35)
    draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        outline='white',
        width=int(size * 0.08)
    )

    # 内側の小さい円
    small_radius = int(size * 0.15)
    draw.ellipse(
        [center - small_radius, center - small_radius,
         center + small_radius, center + small_radius],
        fill='white'
    )

    # テキストを追加
    try:
        # システムフォントを試す
        font_size = int(size * 0.2)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    text = "競輪"
    # テキストのバウンディングボックスを取得
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # テキストを中央下部に配置
    text_x = (size - text_width) // 2
    text_y = int(size * 0.7)

    # 影を追加
    draw.text((text_x + 2, text_y + 2), text, font=font, fill=(0, 0, 0, 128))
    draw.text((text_x, text_y), text, font=font, fill='white')

    # 保存
    img.save(output_path, 'PNG')
    print(f"Generated: {output_path}")

def main():
    # 出力ディレクトリ
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

    # 192x192 と 512x512 のアイコンを生成
    create_icon(192, os.path.join(frontend_dir, 'icon-192.png'))
    create_icon(512, os.path.join(frontend_dir, 'icon-512.png'))

    print("アイコンの生成が完了しました！")

if __name__ == '__main__':
    main()
