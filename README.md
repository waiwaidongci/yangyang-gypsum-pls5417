# 再生石膏板试条校准

`yangyang-gypsum-pls` 是工坊的纯后端校准工具。它将六个近红外波段、密度和厚度压缩为与抗折强度相关的潜变量，用独立生产批检验误差，再为本班待测试条排出实测顺序。业务流程、数据列和计算口径见 [BUSINESS.md](BUSINESS.md)。

## 启动

```sh
cp .env.example .env
docker compose build backend
docker compose up -d --no-build --wait
docker compose exec -T backend python -m scripts.seed_demo
```

- API 基址为 `http://127.0.0.1:19432/api`，OpenAPI 定义位于 `/api/openapi.json`，数据库宿主机端口为 `17932`。
- Compose 启动 FastAPI、Python worker、PostgreSQL 16、Redis 7。PostgreSQL 保存台账、运行与复核，Redis 唤醒 worker。数据库和 Redis 使用具名卷。
- `.env` 可设置 `COMPOSE_PROJECT_NAME`、`BACKEND_PORT`、`DB_PORT`、`SEED_DEMO` 和 `JWT_SECRET`。默认启动时载入演示数据，重跑 seed 按账号和台账名称复用已有记录。演示凭据用于本地环境。

| 演示账号 | 工坊角色 | 权限 |
|---|---|---|
| `analyst@demo.local` | 校准分析员 | 导入、提交运行、查看结果 |
| `admin@demo.local` | 质量负责人 | 录入、运行、批准或驳回、查询审计 |
| `viewer@demo.local` | 实测班查阅员 | 查询台账、查看及下载已批准运行 |

三组账号共用演示密码 `Demo-2611!`。通过 `POST /api/auth/login` 提交 `email` 和 `password`，其余业务请求使用 `Authorization: Bearer <access_token>`。

## API 主流程

| 操作 | 接口 | 数据 |
|---|---|---|
| 查阅模板 | `GET /api/spec` | `columns` 按导入表头顺序列出，`settings` 提供默认值与边界；`paired_columns`、`conversion_settings` 为换机对照模板与换算参数 |
| 导入试条 | `POST /api/datasets` | JSON 的 `name` 和 `csv`，其中 `csv` 为 UTF-8 CSV 全文 |
| 查看台账 | `GET /api/datasets`、`GET /api/datasets/{id}` | 台账摘要、校验散列和分页试条数据 |
| 启动校准 | `POST /api/runs` | `dataset_id`、`label`、可选 `settings` |
| 查看校准 | `GET /api/runs/{id}` | 等待 `completed`，在 `result` 中读摘要、指标及各表 |
| 质量复核 | `POST /api/runs/{id}/review` | `decision` 为 `approved` 或 `rejected`，`note` 为复核意见 |
| 下载表 | `GET /api/runs/{id}/export/{key}` | 复核批准后可下载，`key` 对应结果表标识 |

换近红外仪后仍用原抗折模型时，走同一套台账/复核/下载模式：

| 操作 | 接口 | 数据 |
|---|---|---|
| 导入双机对照 | `POST /api/paired-datasets` | JSON 的 `name` 和 `csv`，列见 `paired_columns`（含 `instrument`） |
| 查看对照台账 | `GET /api/paired-datasets`、`GET /api/paired-datasets/{id}` | 台账摘要、试条数、仪器列表及分页读数 |
| 启动换机交接 | `POST /api/handoffs` | `base_run_id`（原冻结模型运行）、`dataset_ids`（要合并的对照台账）、`anchor_instrument`、可选 `previous_handoff_id`（晚到对照复核）、可选 `settings`（`max_hops`、`error_mpa`） |
| 查看交接 | `GET /api/handoffs/{id}` | 等待 `completed`，读 `ready`、`reasons` 及 `result` 各表 |
| 交接复核 | `POST /api/handoffs/{id}/review` | `decision`、`note`；批准后冻结该次判定与依据 |
| 下载交接表 | `GET /api/handoffs/{id}/export/{key}` | 交接已批准且未暂缓时可用，`key` 见 BUSINESS.md |

交接暂缓（存在没有合格链的试条，或合格链判在筛查线两侧）时不能批准，也没有正式下载表；复核交接的正式表整张一次性切换。

可在项目根生成完整的可重导演示 CSV。文件包含已测和待测分区，待测抗折一列为空。双机对照演示 CSV 用 `demo_pairs` 生成，每条演示试条在 `NIR-01`、`NIR-02`、`NIR-03` 上各一次读数。

```sh
docker compose exec -T backend python -c 'from src.domain.project import SPEC, demo_rows; from src.utils.csv_files import write_rows; print(write_rows(SPEC["columns"], demo_rows()), end="")' > /tmp/gypsum-demo.csv
docker compose exec -T backend python -c 'from src.domain.project import SPEC, demo_pairs; from src.utils.csv_files import write_rows; print(write_rows(SPEC["paired_columns"], demo_pairs()), end="")' > /tmp/gypsum-pairs.csv
```

可直接使用 seed 台账（含双机对照台账），或把 `/tmp/gypsum-demo.csv`、`/tmp/gypsum-pairs.csv` 的内容按上述导入接口提交。导出的正式 CSV 带 UTF-8 BOM，各结果表的技术表头与 JSON 的 `columns` 对应。

## 后端开发

使用 Python 3.11，在 `backend` 安装 `requirements.txt`。可用 `DATABASE_URL` 和 `REDIS_URL` 指向已有 PostgreSQL、Redis，分别启动 `uvicorn src.main:app --port 19432` 和 `python -m src.worker`。`backend/src/domain` 按录入校验、模拟台账、PLS 路径、分量选型、独立回测和交接导出分工。`backend/src/{routes,services,models}` 提供 API、业务状态与持久化，`backend/{scripts,migrations}` 提供种子数据和数据库启动初始化。

`backend/src/constants/workflow.py` 定义角色、运行状态、交接状态和复核决定，分别用于路由授权、运行队列持久化和导出门槛。`train`、`holdout`、`pending` 由 `backend/src/domain/records.py` 校验，并用于 `project.py` 的分区编排；双机对照由 `backend/src/domain/instruments.py` 校验与拟合换算边，`backend/src/domain/conversion.py` 负责接力链验收与换机交接编排。PLS1 生产计算使用 NumPy，随批次提供的 `test-artifacts/yangyang-gypsum-pls/tests/oracle.py` 使用 Krylov 子空间最小二乘独立核对。第三方依赖遵循各自许可证。
