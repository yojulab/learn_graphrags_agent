#!/bin/bash
# ============================================================
# GraphRAG Agent Docker Compose Runner
# .env 파일의 COMPOSE_PROFILES 설정을 읽어 선택적으로 컨테이너를 구동합니다.
# 사용법:
#   ./run.sh up -d        # .env의 COMPOSE_PROFILES에 따라 백그라운드 실행
#   ./run.sh down         # 실행 중인 컨테이너 중지 및 종료
#   ./run.sh ps           # 현재 컨테이너 상태 확인
#   ./run.sh logs -f [서비스명]  # 로그 실시간 모니터링
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. .env 파일 로드 및 COMPOSE_PROFILES 환경변수 export
if [ -f .env ]; then
  PROFILES=$(grep -E '^COMPOSE_PROFILES=' .env | cut -d '=' -f2- | tr -d '\r' | tr -d '"' | tr -d "'")
  if [ -n "$PROFILES" ]; then
    export COMPOSE_PROFILES="$PROFILES"
    echo ">>> [Docker] Loaded COMPOSE_PROFILES from .env: '${COMPOSE_PROFILES}'"
  fi
fi

# 2. Docker Compose CLI 판별 (docker compose v2 우선, docker-compose v1 폴백)
if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo ">>> [Error] Neither 'docker compose' nor 'docker-compose' command was found." >&2
  exit 1
fi

echo ">>> [Docker] Executing: ${COMPOSE_CMD} $@"
exec ${COMPOSE_CMD} "$@"

