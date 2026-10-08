"""
Playwright Headless 기반 위키피디아 에피소드 데이터 수집기
- Docker 및 리눅스 환경에 최적화된 Headless Chromium 실행
- DOM 파싱 및 텍스트 정규화(공백 분리, 불필요한 줄바꿈 제거)
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

def scrape_episodes_playwright(
    url: str = "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_1",
    headless: bool = True,
    timeout_ms: int = 30000,
) -> List[Dict[str, Any]]:
    """
    Playwright headless 브라우저를 사용하여 위키피디아 에피소드 줄거리를 수집합니다.
    """
    from playwright.sync_api import sync_playwright

    season_match = re.search(r"season_(\d+)", url, re.IGNORECASE)
    season = int(season_match.group(1)) if season_match else 1
    print(f"🌐 [Playwright Headless] Season {season} 수집 시작: {url}")

    browser_args = [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--single-process",
    ]

    episodes: List[Dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=browser_args)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        try:
            print("  ⏳ 페이지 로딩 중...")
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            time.sleep(1.0)  # DOM 안정화 대기

            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            table = soup.select_one("table.wikitable.plainrowheaders.wikiepisodetable")

            if not table:
                print("  ⚠️ 에피소드 테이블을 찾을 수 없습니다. 기본 selector 폴백 적용")
                table = soup.select_one("table.wikiepisodetable")

            if not table:
                raise ValueError("위키피디아 페이지에서 에피소드 테이블을 찾을 수 없습니다.")

            rows = table.select("tr.vevent.module-episode-list-row")
            print(f"  ✅ 에피소드 행 {len(rows)}개 발견")

            for i, row in enumerate(rows, start=1):
                synopsis = None
                synopsis_row = row.find_next_sibling("tr", class_="expand-child")
                if synopsis_row:
                    synopsis_cell = synopsis_row.select_one("td.description div.shortSummaryText")
                    if synopsis_cell:
                        raw_text = synopsis_cell.get_text(separator=" ", strip=True)
                        synopsis = re.sub(r"\s+", " ", raw_text).strip()

                episodes.append({
                    "season": season,
                    "episode_in_season": i,
                    "synopsis": synopsis,
                })

            print(f"  ✨ 총 {len(episodes)}개 에피소드 수집 완료")

        finally:
            browser.close()

    return episodes

def save_scraped_data(episodes: List[Dict[str, Any]], output_path: str = "output/raw_data.json") -> None:
    """수집된 데이터를 JSON 파일로 저장합니다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
    print(f"💾 데이터 저장 완료: {output_path} ({len(episodes)}건)")

if __name__ == "__main__":
    episodes = scrape_episodes_playwright()
    save_scraped_data(episodes)
