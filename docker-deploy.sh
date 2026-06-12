#!/bin/bash
# etf-backtrader 部署/更新脚本
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

case "${1:-help}" in
  build)
    echo "==> 构建镜像..."
    docker-compose build --pull
    echo "==> 启动服务..."
    docker-compose up -d
    ;;

  update)
    echo "==> 拉取最新代码..."
    git pull
    echo "==> 重新构建并启动..."
    docker-compose up -d --build
    echo "==> 清理旧镜像..."
    docker image prune -f
    ;;

  restart)
    echo "==> 重启服务..."
    docker-compose restart
    ;;

  logs)
    docker-compose logs -f --tail=100
    ;;

  status)
    docker-compose ps
    echo "---"
    docker stats --no-stream etf-backtrader 2>/dev/null || true
    ;;

  backup)
    BACKUP_DIR="${2:-./backups}"
    mkdir -p "$BACKUP_DIR"
    TS=$(date +%Y%m%d_%H%M%S)
    echo "==> 备份数据卷 -> $BACKUP_DIR/etf_data_$TS.tar.gz"
    docker run --rm -v etf-backtrader_etf_data:/data -v "$(realpath "$BACKUP_DIR")":/backup alpine \
      tar czf "/backup/etf_data_$TS.tar.gz" -C /data .
    echo "==> 完成"
    ;;

  restore)
    if [ -z "$2" ]; then
      echo "用法: $0 restore <backup_file.tar.gz>"
      exit 1
    fi
    echo "==> 从 $2 恢复数据..."
    docker run --rm -v etf-backtrader_etf_data:/data -v "$(realpath "$(dirname "$2")")":/backup alpine \
      tar xzf "/backup/$(basename "$2")" -C /data
    echo "==> 重启服务..."
    docker-compose restart
    ;;

  migrate-csv)
    echo "==> 从 CSV 迁移已有数据到 SQLite..."
    docker-compose exec etf-backtrader python3 -c "
import os
os.environ['ETF_CSV_FALLBACK'] = '1'
from etf_data import load_config, load_prices
cfg = load_config()
for src in ['tencent', 'akshare', 'em']:
    for gname, etfs in cfg.get('groups', {}).items():
        try:
            load_prices({v: k for k, v in etfs.items()}, gname, source=src)
            print(f'  {src}/{gname}: 完成')
        except Exception as e:
            print(f'  {src}/{gname}: 跳过 ({e})')
print('迁移完成')
"
    ;;

  shell)
    docker exec -it etf-backtrader bash
    ;;

  help|*)
    echo "用法: $0 <command> [args]"
    echo ""
    echo "命令:"
    echo "  build              构建镜像并启动"
    echo "  update             拉取代码 + 重新构建 + 启动"
    echo "  restart            重启服务"
    echo "  logs               查看日志"
    echo "  status             查看状态"
    echo "  backup [dir]       备份数据卷"
    echo "  restore <file>     从备份恢复"
    echo "  migrate-csv        将已有 CSV 数据导入 SQLite"
    echo "  shell              进入容器"
    ;;
esac
