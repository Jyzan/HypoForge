# HypoForge Refinement Assistant

HypoForge 的**方案细化工作台**：一个 Claude Code 风格的对话式 Agent（Flask + SSE 流式），
把 Pipeline 产出的最终研究方案细化为可执行的研究计划——材料与试剂清单、实验步骤拆解、
时间线、风险与替代方案、可运行的干实验脚本（蛋白质/FoldX、AI/视觉、通用计算）。

## 特性

- **方案自动载入**：在 HypoForge Web UI 的方案卡片点击「固定此方案并细化」，
  方案自动写入本工作台并立即开始对话式细化
- **历史会话**：对话自动持久化，支持改名、切换、删除；Token 预算按会话隔离
- **权限审批**：写文件、执行命令前必须经用户确认，未授权时只生成脚本与说明
- **本地知识检索（RAG）**：对已下载论文、固定方案、细化产出建立分块索引，
  关键词检索开箱即用，配置 `QWEN_EMBEDDING_MODEL` 后自动升级为向量混合检索
- **执行子代理**：多步操作委派给带安全审计的 Worker，支持取消与进度回显
- **自动压缩**：按模型上下文窗口压力自动压缩历史（Claude Code 风格 compact），
  上下文溢出时强制压缩重试

## 启动

```powershell
pip install -r requirements.txt
cd refinement_assistant
python app.py            # http://127.0.0.1:5000
```

LLM 通过环境变量读取（与 HypoForge 主程序共用同一份 `.env`）：
`QWEN_API_KEY` / `QWEN_BASE_URL` / `QWEN_MODEL`，可选 `QWEN_EMBEDDING_MODEL`。

## 工具一览

| 工具 | 用途 |
|---|---|
| `run_powershell` | 执行命令（需权限审批），用于安装/运行干实验脚本 |
| `list_files` / `read_file` / `create_file` / `modify_file` / `delete_file` | 文件操作 |
| `load_refinement_input` | 读取已固定的研究方案 |
| `web_search` | 通用网络搜索（Serper / DuckDuckGo 回退） |
| `read_downloaded_papers` | 读取当前 HypoForge 运行已下载的论文 |
| `search_workspace_knowledge` | 本地知识库 RAG 检索 |
| `generate_dry_lab_script` | 生成干实验脚本草稿 |
| `save_refinement_output` | 保存细化结果（JSON + Markdown）到 `output/refined/<run_id>/` |

## 与 HypoForge 的集成

Web UI 最终方案卡片上的「固定此方案并细化」按钮会把方案导出为
`refinement_assistant/refinement_input.json`，自动拉起本工作台；
工作台启动时检测到未载入的固定方案即自动开始细化对话。

## 测试

```powershell
python test_refinement_driver.py "你的测试消息"   # 命令行驱动对话（自动批准只读命令）
```

## 运行时数据（已 gitignore）

`sessions/`（历史对话）、`log/`、`allowlist/`、`config.json`、`token_state.json`、
`llm_response_cache.sqlite3`、`sandbox/`、`refinement_input.json`。
