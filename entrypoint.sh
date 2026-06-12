#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/data}"
mkdir -p "$DATA_DIR"

# 首次启动：将默认 etf_config.json 复制到数据目录（用户可修改组合配置）
if [ ! -f "$DATA_DIR/etf_config.json" ]; then
    cp /app/etf_config.json "$DATA_DIR/etf_config.json" 2>/dev/null || true
fi

# 如果 data/ 下有旧缓存文件，链接到工作目录（Streamlit 不支持自定义数据路径）
# CSV 缓存直接指向 /data
ln -sf "$DATA_DIR" /app/data 2>/dev/null || true

echo "[entrypoint] DATA_DIR=$DATA_DIR"
echo "[entrypoint] Starting Streamlit on port ${STREAMLIT_SERVER_PORT:-8501}..."

exec streamlit run etf_app.py \
    --server.port="${STREAMLIT_SERVER_PORT:-8501}" \
    --server.address="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}" \
    --server.headless="${STREAMLIT_SERVER_HEADLESS:-true}" \
    --browser.gatherUsageStats="${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false}"
