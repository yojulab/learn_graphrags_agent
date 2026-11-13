# Docker 인프라 및 Langfuse 설정 가이드

본 디렉토리(`dockers/`)는 **GraphRAG Agent (Demon Slayer Season 1 Knowledge Graph)** 시스템을 실행하기 위한 컨테이너 인프라 및 **Langfuse LLM 관측성(Observability) 플랫폼** 구성을 제공합니다.

---

## 🏗️ 서비스 구성 요약

| 서비스명 | 컨테이너명 | 프로필명 | 기본 포트 | 설명 |
| :--- | :--- | :--- | :--- | :--- |
| **db_neo4j** | `${PROJECT_NAME}_db_neo4j` | `neo4j`, `all` | `7474`, `7687` | Neo4j 5+ 네이티브 그래프 데이터베이스 & HNSW 벡터 인덱스 |
| **db_postgres** | `${PROJECT_NAME}_db_postgres` | `postgres`, `all` | `5432` | PostgreSQL 16+ & Apache AGE (Graph 확장을 통한 openCypher 지원 및 Langfuse DB) |
| **db_mongo** | `${PROJECT_NAME}_db_mongo` | `mongo`, `all` | `27017` | MongoDB NoSQL 데이터베이스 |
| **ollama** | `${PROJECT_NAME}_ai_ollama` | `ollama`, `all` | `11434` | Ollama AI 추론(`sam860/exaone-4.0:1.2b-thinking-Q8_0`) 및 임베딩(`bge-m3:567m`) |
| **langfuse** | `${PROJECT_NAME}_langfuse` | `langfuse`, `all` | `3000` | **Langfuse LLM Observability & Tracing 플랫폼 (자동 초기화 탑재)** |
| **app_fullstack** | `${PROJECT_NAME}_app_fullstack` | `app`, `all` | - | FastAPI 기반 GraphRAG 에이전트 풀스택 애플리케이션 컨테이너 |

---

## 🔍 Langfuse 연동 아키텍처

1. **경량 및 안정적 컨테이너 구성**:
   - `ghcr.io/langfuse/langfuse:2` 이미지를 채택하여 무거운 ClickHouse나 MinIO 클러스터 없이 기존 PostgreSQL과 완벽하게 연동됩니다.
   - 필요 시 `LANGFUSE_IMAGE_TAG` 환경변수로 버전을 전환할 수 있습니다.
2. **PostgreSQL 멀티 DB 자동 프로비저닝 (`02_init_langfuse.sh`)**:
   - `db_postgres` 컨테이너 최초 기동 시 `dockers/postgres/init/02_init_langfuse.sh` 스크립트가 실행되어, GraphRAG용 DB(`db_graphrag_agent`)와 격리된 **`db_langfuse`** 데이터베이스를 자동으로 생성하고 권한을 부여합니다.
3. **Headless 초기화 (자동 부트스트래핑)**:
   - 컨테이너 시작 시 조직, 프로젝트, 관리자 계정, API Key가 자동으로 생성되어 UI에서 수동 가입 절차 없이 애플리케이션 코드가 즉시 트레이싱을 시작할 수 있습니다.
   - **기본 관리자 계정**: `admin@example.com` / `admin123456`
   - **기본 프로젝트**: `Demon Slayer GraphRAG` (`graphrag-project`)
   - **기본 Public Key**: `pk-lf-graphrag-agent-public`
   - **기본 Secret Key**: `sk-lf-graphrag-agent-secret`
4. **선택적 프로필 실행 (Docker Compose Profiles)**:
   - `COMPOSE_PROFILES`에 `langfuse`를 포함하거나 제외하여 유연하게 기동을 제어할 수 있습니다.

---

## 🚀 실행 방법

### 1. 전체 인프라 기동 (Langfuse 포함)
```bash
cd dockers
docker-compose up -d
```
기본적으로 `COMPOSE_PROFILES=neo4j,postgres,mongo,ollama,app,langfuse`로 설정되어 있어 모든 서비스가 한 번에 실행됩니다.

### 2. 특정 프로필만 선택 기동
```bash
# Langfuse와 PostgreSQL만 실행할 경우
docker-compose --profile postgres --profile langfuse up -d

# 또는 .env 파일에서 COMPOSE_PROFILES 변경:
# COMPOSE_PROFILES=postgres,langfuse
```

### 3. 접속 주소 및 대시보드
- **Langfuse Web UI**: [http://localhost:3000](http://localhost:3000)
  - 로그인: `admin@example.com` / `admin123456`
- **Neo4j 브라우저**: [http://localhost:7474](http://localhost:7474) (`neo4j` / `admin123`)
- **Ollama API**: [http://localhost:11434](http://localhost:11434)
- **PostgreSQL / Apache AGE**: `localhost:5432`

---

## ⚙️ Langfuse 환경 변수 명세

`.env` 및 `dockers/.env`에서 아래 변수들을 커스터마이징할 수 있습니다:

```env
# ============================================================
# Langfuse 설정
# ============================================================
LANGFUSE_IMAGE_TAG=2
LANGFUSE_PORT=3000
LANGFUSE_DB_NAME=db_langfuse
LANGFUSE_NEXTAUTH_SECRET=7f884a1e94b29bb88a2e1d7cf9d2c2d431c172428dfabf192b0058b8772a8187
LANGFUSE_SALT=8b4c27806e57ab2702758155bb3f9ff3a216c02698971f11467be5cd8198f7df
LANGFUSE_NEXTAUTH_URL=http://localhost:3000
LANGFUSE_TELEMETRY_ENABLED=false

# Headless Initialization (초기 자동 프로비저닝)
LANGFUSE_INIT_ORG_ID=graphrag-org
LANGFUSE_INIT_ORG_NAME=GraphRAG Organization
LANGFUSE_INIT_PROJECT_ID=graphrag-project
LANGFUSE_INIT_PROJECT_NAME=Demon Slayer GraphRAG
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-graphrag-agent-public
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-graphrag-agent-secret
LANGFUSE_INIT_USER_EMAIL=admin@example.com
LANGFUSE_INIT_USER_NAME=Admin
LANGFUSE_INIT_USER_PASSWORD=admin123456

# Client SDK 연동 정보 (config.py 및 애플리케이션)
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-graphrag-agent-public
LANGFUSE_SECRET_KEY=sk-lf-graphrag-agent-secret
```

---

## 🐍 Python SDK 연동 예제

`requirements.txt`에 `langfuse>=2.0.0`이 포함되어 있으므로, Python 코드에서 아래와 같이 즉시 트레이싱을 시작할 수 있습니다:

```python
from langfuse import Langfuse
from langfuse.openai import openai  # OpenAI API 래핑 지원
import config

# 초기화 (config.py SSOT 설정 자동 사용)
langfuse = Langfuse(
    public_key=config.LANGFUSE_PUBLIC_KEY,
    secret_key=config.LANGFUSE_SECRET_KEY,
    host=config.LANGFUSE_HOST
)

# 기본 Trace 생성
trace = langfuse.trace(
    name="demon-slayer-graphrag-query",
    user_id="user_test_01",
    metadata={"version": "v3", "backend": "neo4j"}
)

# 자식 Span 또는 Generation 기록
generation = trace.generation(
    name="text2cypher-generation",
    model=config.LLM_MODEL,
    input="카마도 탄지로와 네즈코의 관계는 무엇인가?"
)
# 작업 완료 후 업데이트
generation.end(output="MATCH (a:인간 {name: '카마도 탄지로'})-[r:SIBLING_OF]->(b:도깨비 {name: '카마도 네즈코'}) RETURN r")
```
