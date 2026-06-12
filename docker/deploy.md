# 腾讯云部署指南

## 前提

- 腾讯云服务器（建议 2C4G 以上）
- 安装 Docker + Docker Compose

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | bash
sudo systemctl enable docker
sudo systemctl start docker

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## 首次部署

```bash
# 1. 克隆项目
git clone <your-repo-url> etf-backtrader
cd etf-backtrader

# 2. 切换到 docker 分支
git checkout feature/docker-deploy

# 3. 构建并启动
./docker-deploy.sh build

# 4. 查看启动日志
./docker-deploy.sh logs

# 5. 访问
# http://<服务器IP>:8501
```

## 日常更新代码

```bash
cd etf-backtrader

# 拉取最新代码 + 重建容器 + 启动
./docker-deploy.sh update
```

相当于执行：
```bash
git pull                    # 拉取最新代码
docker-compose up -d --build  # 重新构建镜像并启动
docker image prune -f         # 清理旧镜像
```

**数据安全：** Docker 卷 `etf_data` 独立于容器文件系统，重建容器不会丢失任何数据（行情缓存、用户配置均不受影响）。

## 数据持久化

所有持久数据存储在 Docker 命名卷 `etf_data` 中：
- `etf.db` — SQLite 数据库（用户配置、分析缓存）
- `etf_prices_*.csv` — 行情缓存
- `etf_config.json` — ETF 组合配置（可手动编辑）
- `grid_*.csv` — 网格数据
- `cache/` — 多因子分析缓存

```bash
# 查看数据卷位置
docker volume inspect etf-backtrader_etf_data

# 备份数据
docker run --rm -v etf-backtrader_etf_data:/data -v /tmp/backup:/backup alpine tar czf /backup/etf_data_$(date +%Y%m%d).tar.gz -C /data .

# 恢复数据
docker run --rm -v etf-backtrader_etf_data:/data -v /tmp/backup:/backup alpine tar xzf /backup/etf_data_20260101.tar.gz -C /data
```

## 已有 CSV 数据迁移

首次部署后，已有 CSV 缓存文件中的数据不会自动写入 SQLite。运行以下命令导入：

```bash
./docker-deploy.sh migrate-csv
```

该命令遍历所有组合和数据源，触发 `load_prices()` 将数据写入 `daily_close` 和 `daily_open` 表。
CSV 文件仍然保留，不会删除。

## Nginx 反向代理（可选）

```
/etc/nginx/sites-available/etf-backtrader
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50m;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

启用 HTTPS（推荐）：

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 腾讯云安全组

开放以下端口：
- **8501** — Streamlit Web UI
- **80** — HTTP（如配置 Nginx）
- **443** — HTTPS（如配置 Nginx）

## 维护

```bash
# 重启
docker-compose restart

# 停止
docker-compose down

# 停止并删除数据卷（⚠️ 会清空所有缓存和配置）
docker-compose down -v

# 查看资源占用
docker stats etf-backtrader

# 进入容器
docker exec -it etf-backtrader bash
```
