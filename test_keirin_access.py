#!/usr/bin/env python3
"""Test different approaches to access KEIRIN.JP"""
import requests
import time

def test_approach_1():
    """Basic approach with minimal headers"""
    print("\n=== Test 1: Basic approach ===")
    try:
        resp = requests.get("https://keirin.jp/sp/json?type=JSJ058&kday=20250110", timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.ok:
            print(f"Success! Data length: {len(resp.text)}")
    except Exception as e:
        print(f"Error: {e}")

def test_approach_2():
    """With browser-like headers"""
    print("\n=== Test 2: Browser-like headers ===")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://keirin.jp/sp/',
        'X-Requested-With': 'XMLHttpRequest',
    }
    try:
        resp = requests.get("https://keirin.jp/sp/json?type=JSJ058&kday=20250110",
                           headers=headers, timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.ok:
            print(f"Success! Data length: {len(resp.text)}")
    except Exception as e:
        print(f"Error: {e}")

def test_approach_3():
    """With session initialization"""
    print("\n=== Test 3: Session with initialization ===")
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    })

    try:
        # First visit the top page
        print("  Step 1: Visiting top page...")
        resp1 = session.get("https://keirin.jp/sp/", timeout=10)
        print(f"  Top page status: {resp1.status_code}")
        print(f"  Cookies received: {len(session.cookies)}")

        # Wait a bit
        time.sleep(1)

        # Now try the API
        print("  Step 2: Calling API...")
        session.headers.update({
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': 'https://keirin.jp/sp/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        })
        resp2 = session.get("https://keirin.jp/sp/json?type=JSJ058&kday=20250110", timeout=10)
        print(f"  API status: {resp2.status_code}")
        if resp2.ok:
            print(f"  Success! Data length: {len(resp2.text)}")
    except Exception as e:
        print(f"  Error: {e}")

def test_approach_4():
    """Try PC version instead of SP"""
    print("\n=== Test 4: PC version ===")
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    })

    try:
        # Try accessing PC version
        resp = session.get("https://keirin.jp/pc/", timeout=10)
        print(f"PC top page status: {resp.status_code}")

        # Try different endpoints
        resp2 = session.get("https://keirin.jp/pc/dfw/portal/guest/race/race-schedule?date=20250110", timeout=10)
        print(f"PC race schedule status: {resp2.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_approach_1()
    test_approach_2()
    test_approach_3()
    test_approach_4()

    print("\n" + "="*60)
    print("Summary: Testing multiple access approaches to KEIRIN.JP")
    print("="*60)
