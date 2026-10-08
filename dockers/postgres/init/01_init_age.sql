-- Apache AGE 초기화 (최초 컨테이너 생성 시 POSTGRES_DB 에 대해 1회 실행)
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';

-- 접속 시 ag_catalog 를 기본 search_path 에 포함
DO $$
BEGIN
    EXECUTE format(
        'ALTER DATABASE %I SET search_path = ag_catalog, "$user", public',
        current_database()
    );
END
$$;
