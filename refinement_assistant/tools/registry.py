"""Refinement Assistant tool registry.

Only tools needed for HypoForge plan refinement are exposed.  All system-level
commands go through the permission manager in core/permission_manager.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from .file_ops import list_files, read_file, create_file, modify_file, delete_file
from .powershell_ops import run_powershell
from .hypoforge_tools import (
    web_search,
    read_downloaded_papers,
    save_refinement_output,
    generate_dry_lab_script,
)
from core.rag import get_workspace_index


def search_workspace_knowledge(query: str, top_k: int = 5, rebuild: bool = False) -> str:
    """在本地知识库（论文/方案/细化产出/记忆）中检索相关片段。"""
    try:
        return get_workspace_index().search(query, top_k=top_k, rebuild=rebuild)
    except Exception as exc:
        return f"Knowledge search failed: {type(exc).__name__}: {exc}"


def load_refinement_input(path: str = "") -> str:
    """Read a refinement input JSON (the frozen HypoForge plan) from disk."""
    if not path:
        # Default input file inside the assistant workspace.
        candidates = [
            Path(__file__).resolve().parents[1] / "refinement_input.json",
            Path(__file__).resolve().parents[2] / "output" / "refined" / "refinement_input.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = str(candidate)
                break
    if not path:
        return "No refinement input file found."
    try:
        data = Path(path).read_text(encoding="utf-8")
        return data[:20000]
    except Exception as exc:
        return f"Failed to read refinement input: {exc}"


TOOL_REGISTRY = {
    "run_powershell": {
        "function": run_powershell,
        "summary": "执行 PowerShell 命令（需权限审批），用于安装/运行干实验脚本",
        "definition": {
            "type": "function",
            "function": {
                "name": "run_powershell",
                "description": "在本地执行 PowerShell 命令。所有可能修改系统/安装软件/写文件的命令都会先向用户申请权限。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要执行的 PowerShell 命令"}
                    },
                    "required": ["command"],
                },
            },
        },
    },
    "list_files": {
        "function": list_files,
        "summary": "列出目录文件",
        "definition": {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "列出指定目录下的文件。",
                "parameters": {
                    "type": "object",
                    "properties": {"directory": {"type": "string", "description": "目录路径"}},
                    "required": [],
                },
            },
        },
    },
    "read_file": {
        "function": read_file,
        "summary": "读取本地文件",
        "definition": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取指定文件内容，用于查看方案、论文摘要或脚本。",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string", "description": "文件路径"}},
                    "required": ["filename"],
                },
            },
        },
    },
    "create_file": {
        "function": create_file,
        "summary": "创建文件",
        "definition": {
            "type": "function",
            "function": {
                "name": "create_file",
                "description": "在可写沙箱/工作目录中创建文件。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["filename", "content"],
                },
            },
        },
    },
    "modify_file": {
        "function": modify_file,
        "summary": "修改本地文件",
        "definition": {
            "type": "function",
            "function": {
                "name": "modify_file",
                "description": "追加或覆盖文件内容。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {"type": "string", "enum": ["append", "overwrite"]},
                    },
                    "required": ["filename", "content"],
                },
            },
        },
    },
    "delete_file": {
        "function": delete_file,
        "summary": "删除文件",
        "definition": {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "删除可写沙箱中的文件。",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string"}},
                    "required": ["filename"],
                },
            },
        },
    },
    "load_refinement_input": {
        "function": load_refinement_input,
        "summary": "读取已固定的研究方案作为细化任务输入",
        "definition": {
            "type": "function",
            "function": {
                "name": "load_refinement_input",
                "description": "读取 HypoForge 固定的研究方案 JSON 文件。",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "可选的输入文件路径"}},
                    "required": [],
                },
            },
        },
    },
    "web_search": {
        "function": web_search,
        "summary": "通用网络搜索，用于查材料、试剂、设备、协议、代码等",
        "definition": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "搜索互联网上的实际执行信息，例如具体蛋白名称、试剂、软件安装、实验步骤、代码库。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "limit": {"type": "integer", "description": "返回结果数量"},
                    },
                    "required": ["query"],
                },
            },
        },
    },
    "read_downloaded_papers": {
        "function": read_downloaded_papers,
        "summary": "读取当前 HypoForge 运行已下载的论文",
        "definition": {
            "type": "function",
            "function": {
                "name": "read_downloaded_papers",
                "description": "读取已有论文/运行上下文，供细化 Agent 学习方案引用的原始方法。可传入 HypoForge run_id。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "HypoForge 运行 ID"},
                        "limit": {"type": "integer"},
                    },
                    "required": [],
                },
            },
        },
    },
    "generate_dry_lab_script": {
        "function": generate_dry_lab_script,
        "summary": "生成干实验脚本（蛋白质/FoldX、AI/视觉、通用计算）",
        "definition": {
            "type": "function",
            "function": {
                "name": "generate_dry_lab_script",
                "description": "生成一个可运行的干实验 Python 脚本草稿。不会自动执行；需要运行时要通过 PowerShell 工具并申请权限。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string", "description": "领域：protein/ai/generic"},
                        "task": {"type": "string", "description": "具体实验任务"},
                        "detail": {"type": "string", "description": "附加细节"},
                    },
                    "required": ["domain"],
                },
            },
        },
    },
    "save_refinement_output": {
        "function": save_refinement_output,
        "summary": "保存细化结果 JSON + Markdown",
        "definition": {
            "type": "function",
            "function": {
                "name": "save_refinement_output",
                "description": "保存最终细化执行计划到 output/refined/<run_id>。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "json_data": {"type": "string"},

                        "markdown": {"type": "string"},
                    },
                    "required": ["run_id", "json_data"],
                },
            },
        },
    },
    "search_workspace_knowledge": {
        "function": search_workspace_knowledge,
        "summary": "在本地知识库（论文/固定方案/细化产出/记忆）中检索相关片段（RAG）",
        "definition": {
            "type": "function",
            "function": {
                "name": "search_workspace_knowledge",
                "description": "在本地知识库中语义检索与查询相关的文本片段。知识库覆盖已下载的论文摘要、固定的研究方案、历史细化产出和工作记忆，回答方案依据、引用方法、既往结论时优先使用，避免整篇读文件。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索问题或关键词，中英文均可"},
                        "top_k": {"type": "integer", "description": "返回片段数量，默认 5"},
                        "rebuild": {"type": "boolean", "description": "强制重建索引（工作区文件有更新时用）"},
                    },
                    "required": ["query"],
                },
            },
        },
    },
}


TOOL_CATALOGUE = {name: info["summary"] for name, info in TOOL_REGISTRY.items()}
