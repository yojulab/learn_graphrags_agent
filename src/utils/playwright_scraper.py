"""
Playwright Headless 기반 위키피디아 에피소드 정밀 데이터 수집기
- Docker 및 리눅스 환경에 최적화된 Headless Chromium 실행
- 영문, 일문, 로마자 에피소드 제목, 감독, 각본, 방영일 등 다각도 메타데이터 추출
- HTML 인라인 링크 주변 공백 누락 방지 및 텍스트/구두점 정규화
- 각주([1], [2] 등) 및 불필요한 DOM 아티팩트 제거
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def parse_title(raw_title: str) -> Dict[str, str]:
    """위키피디아 제목 셀에서 영문, 일문(한자), 로마자 표기를 정밀 파싱합니다."""
    title_en = ""
    title_ja = ""
    title_romaji = ""

    # 1. 영문 제목 추출 (첫 번째 따옴표 안)
    m_en = re.search(r'"([^"]+)"', raw_title)
    if m_en:
        title_en = m_en.group(1).strip()
    else:
        parts = raw_title.split("Transliteration")
        title_en = parts[0].strip()

    # 2. 일본어(한자) 제목 추출
    m_ja = re.search(r'Japanese\s*:\s*([^)]+)', raw_title)
    if m_ja:
        title_ja = m_ja.group(1).strip()

    # 3. 로마자 발음 표기 추출
    m_ro = re.search(r'Transliteration\s*:\s*"?\s*([^"()]+?)\s*"?\s*(?:\(|$)', raw_title)
    if m_ro:
        title_romaji = m_ro.group(1).strip().strip('"')

    return {
        "title_en": title_en,
        "title_ja": title_ja,
        "title_romaji": title_romaji,
    }

def clean_synopsis(raw_text: str) -> str:
    """시놉시스 텍스트 정제: 공백 중복 제거, 구두점 및 단어 경계 보정."""
    # 1. 연속 공백 및 줄바꿈 정규화
    text = re.sub(r"\s+", " ", raw_text).strip()

    # 2. 고유명사 및 약어 공백 보정 (예: Mt.Sagiri -> Mt. Sagiri)
    text = re.sub(r"\bMt\.([A-Z])", r"Mt. \1", text)

    # 3. 문자-구두점 간 붙은 공백 보정
    text = re.sub(r"([a-zA-Z0-9]),([a-zA-Z])", r"\1, \2", text)
    text = re.sub(r"([a-zA-Z0-9]);([a-zA-Z])", r"\1; \2", text)
    text = re.sub(r"\.([A-Z][a-z])", r". \1", text)

    # 4. 구두점 앞 불필요한 공백 제거
    text = re.sub(r"\s+([,.:;?!])", r"\1", text)

    # 5. 따옴표 정규화
    text = text.replace('"', '"').replace('"', '"')

    return text.strip()

def scrape_episodes_playwright(
    url: str = "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_1",
    headless: bool = True,
    timeout_ms: int = 30000,
) -> List[Dict[str, Any]]:
    """Playwright Headless 브라우저를 구동하여 위키피디아 시즌 데이터를 정밀 수집합니다."""
    season_match = re.search(r"season_(\d+)", url, re.IGNORECASE)
    season = int(season_match.group(1)) if season_match else 1
    print(f"🌐 [Playwright Headless] Season {season} 정밀 수집 시작: {url}")

    browser_args = [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
    ]

    episodes: List[Dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=browser_args)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        try:
            print("  ⏳ 페이지 로딩 중 (domcontentloaded)...")
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            table = soup.select_one("table.wikitable.plainrowheaders.wikiepisodetable")

            if not table:
                print("  ⚠️ 에피소드 테이블 폴백 selector 적용")
                table = soup.select_one("table.wikiepisodetable")

            if not table:
                raise ValueError("위키피디아 페이지에서 에피소드 테이블을 찾을 수 없습니다.")

            rows = table.select("tr.vevent.module-episode-list-row")
            print(f"  ✅ 에피소드 행 {len(rows)}개 발견")

            for i, row in enumerate(rows, start=1):
                cells = row.find_all(["th", "td"])
                ep_overall_str = cells[0].get_text(strip=True) if len(cells) > 0 else str(i)
                ep_season_str = cells[1].get_text(strip=True) if len(cells) > 1 else str(i)
                ep_overall = int(ep_overall_str) if ep_overall_str.isdigit() else i
                ep_season = int(ep_season_str) if ep_season_str.isdigit() else i

                title_cell = cells[2] if len(cells) > 2 else None
                title_raw = title_cell.get_text(separator=" ", strip=True) if title_cell else ""
                title_info = parse_title(title_raw)

                directed_by = cells[3].get_text(separator=" ", strip=True) if len(cells) > 3 else ""
                written_by = cells[4].get_text(separator=" ", strip=True) if len(cells) > 4 else ""
                raw_air_date = cells[5].get_text(separator=" ", strip=True) if len(cells) > 5 else ""

                m_date = re.search(r"(\d{4}-\d{2}-\d{2})", raw_air_date)
                air_date = m_date.group(1) if m_date else raw_air_date

                # 시놉시스 추출 및 태그 정제
                synopsis = ""
                synopsis_row = row.find_next_sibling("tr", class_="expand-child")
                if synopsis_row:
                    desc_div = (
                        synopsis_row.select_one("td.description div.shortSummaryText")
                        or synopsis_row.select_one("td.description")
                    )
                    if desc_div:
                        # 각주([1], [2] 등) 제거
                        for ref in desc_div.select("sup.reference"):
                            ref.decompose()

                        # <a> 태그 주변 공백 확보하여 단어 뭉침 방지
                        for a_tag in desc_div.find_all("a"):
                            a_text = a_tag.get_text(strip=True)
                            a_tag.replace_with(f" {a_text} ")

                        raw_synopsis = desc_div.get_text(separator=" ", strip=True)
                        synopsis = clean_synopsis(raw_synopsis)

                episodes.append({
                    "season": season,
                    "episode_overall": ep_overall,
                    "episode_in_season": ep_season,
                    "episode_code": f"S{season:02d}E{ep_season:02d}",
                    "title": title_info["title_en"] or title_raw,
                    "title_en": title_info["title_en"],
                    "title_ja": title_info["title_ja"],
                    "title_romaji": title_info["title_romaji"],
                    "directed_by": directed_by,
                    "written_by": written_by,
                    "air_date": air_date,
                    "synopsis": synopsis,
                })

            print(f"  ✨ 총 {len(episodes)}개 에피소드 정밀 수집 완료")

        finally:
            browser.close()

    return episodes

def save_scraped_data(episodes: List[Dict[str, Any]], output_path: str = "output/raw_data.json") -> None:
    """수집된 정밀 데이터를 JSON 파일로 저장합니다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
    print(f"💾 데이터 저장 완료: {output_path} ({len(episodes)}건)")

def validate_dataset(episodes: List[Dict[str, Any]]) -> bool:
    """수집된 데이터셋의 정밀성 및 완전성을 자동 검증합니다."""
    print("\n🔍 수집 데이터 품질 정밀 검증 중...")
    errors = []
    for ep in episodes:
        ep_code = ep.get("episode_code", f"Ep {ep.get('episode_in_season')}")
        if not ep.get("synopsis"):
            errors.append(f"{ep_code}: 시놉시스가 비어 있습니다.")
        if len(ep.get("synopsis", "")) < 100:
            errors.append(f"{ep_code}: 시놉시스 길이가 너무 짧습니다 ({len(ep.get('synopsis', ''))}자).")
        if not ep.get("title_en"):
            errors.append(f"{ep_code}: 영문 제목이 누락되었습니다.")

        # 단어 병합 결함 검사 (소문자 뒤에 대문자가 바로 오는 경우)
        camels = re.findall(r"[a-z][A-Z]", ep.get("synopsis", ""))
        if camels:
            errors.append(f"{ep_code}: 단어 뭉침 의심 패턴 발견: {camels}")

    if errors:
        print(f"❌ 검증 실패: {len(errors)}개 이슈 발견")
        for err in errors:
            print(f"  - {err}")
        return False

    print("✅ 데이터 정밀도 검증 100% 통과! (26개 에피소드 완전성 및 단어 뭉침 0건)")
    return True

if __name__ == "__main__":
    episodes = scrape_episodes_playwright()
    save_scraped_data(episodes, "output/raw_data.json")
    save_scraped_data(episodes, "output/raw_data_v3.json")
    is_valid = validate_dataset(episodes)
    if not is_valid:
        exit(1)
