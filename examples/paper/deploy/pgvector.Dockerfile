FROM postgres:16-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates postgresql-server-dev-16 \
    && git clone --depth 1 --branch v0.6.2 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && make -C /tmp/pgvector && make -C /tmp/pgvector install \
    && rm -rf /tmp/pgvector /var/lib/apt/lists/*
