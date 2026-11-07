"""
Playwright Headless 기반 위키피디아 에피소드 정밀 데이터 수집기
(src/utils/playwright_scraper.py -> src/lpg_builder/1_scrape_episodes.py 연동)
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# 신규 lpg_builder 모듈로부터 import하여 단일 구현 유지
from src.lpg_builder.scrape_episodes import (
    parse_title,
    clean_synopsis,
    scrape_episodes_playwright,
    save_scraped_data,
    validate_dataset
)

if __name__ == "__main__":
    episodes = scrape_episodes_playwright()
    save_scraped_data(episodes, "output/raw_data.json")
    is_valid = validate_dataset(episodes)
    if not is_valid:
        exit(1)
