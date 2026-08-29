import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PermissionDecision:
    action: str
    reason: str = ""
    audit_type: str = ""
    command: str = ""
    sandbox_path: str = ""
    message: str = ""
    rewritten_args: dict = None


class PermissionManager:
    FILE_WRITE_TOOLS = {"create_file", "modify_file", "delete_file"}
    FILE_PATH_KEYS = ("filename", "file_path", "path", "directory")

    WRITE_COMMAND_PATTERNS = [
        r"\bset-content\b",
        r"\badd-content\b",
        r"\bout-file\b",
        r"\bnew-item\b",
        r"\bremove-item\b",
        r"\bdel\b",
        r"\brm\b",
        r"\brmdir\b",
        r"\bmove-item\b",
        r"\bcopy-item\b",
        r"\bmkdir\b",
        r">\s*[^&|]+",
        r">>\s*[^&|]+",
        r"\bpip\s+install\b",
        r"\bnpm\s+install\b",
        r"\buv\s+add\b",
    ]

    DANGEROUS_COMMAND_PATTERNS = [
        r"remove-item\s+['\"]?[a-z]:\\?['\"]?\s+.*\b-recurse\b",
        r"remove-item\s+['\"]?[a-z]:\\\s+.*\b-recurse\b",
        r"\brm\s+-rf\s+[/~]",
    ]

    CRITICAL_COMMAND_PATTERNS = [
        r"\bwinget\s+(install|uninstall)\b",
        r"\bchoco\s+(install|uninstall|upgrade)\b",
        r"\bmsiexec(?:\.exe)?\s+/(i|x)\b",
        r"\breg(?:\.exe)?\s+(add|delete|import|restore)\b",
        r"\b(new|set|remove|start|stop)-service\b",
        r"\bsc(?:\.exe)?\s+(create|delete|config|start|stop)\b",
        r"\bschtasks(?:\.exe)?\s+/(create|delete|change|run|end)\b",
        r"\b(register|unregister|set)-scheduledtask\b",
        r"\bset-mppreference\b",
        r"\bnetsh\s+advfirewall\b",
        r"\bset-executionpolicy\b",
        r"\bstop-process\b.*\b(lsass|csrss|wininit|services|smss|system)\b",
        r"\btaskkill\b.*(?:/im\s+\*|lsass|csrss|wininit|services|smss)\b",
        r"\b(restart|stop)-computer\b",
        r"\bshutdown(?:\.exe)?\b",
        r"\bformat-volume\b",
        r"\bformat\s+[a-z]:",
        r"\b(clear|initialize|remove|resize|new|set)-disk\b",
        r"\b(clear|remove|resize|new|set)-partition\b",
    ]

    def __init__(self, workspace_path, config):
        self.workspace_path = Path(workspace_path).resolve()
        self.config = config or {}

    @property
    def sandbox_path(self):
        configured = self.config.get("sandbox_path")
        if configured:
            return Path(configured).resolve()
        return (self.workspace_path / "sandbox").resolve()

    def refresh(self, config):
        self.config = config or {}

    def evaluate(self, tool_name, args):
        args = args or {}
        if tool_name in self.FILE_WRITE_TOOLS:
            return self._evaluate_file_write(tool_name, args)
        if tool_name == "run_powershell":
            return self._evaluate_powershell(args.get("command", ""))
        return PermissionDecision(action="allow")

    def _evaluate_file_write(self, tool_name, args):
        if not self._sandbox_confirmed():
            return self._sandbox_ask(tool_name)

        target = self._target_path(args)
        if target and not self._is_inside_sandbox(target):
            return PermissionDecision(
                action="ask",
                reason="outside_sandbox",
                audit_type="CRITICAL",
                command=str(target),
                message=f"File operation outside sandbox requires one-time approval: {target}",
            )
        return PermissionDecision(action="allow")

    def _evaluate_powershell(self, command):
        normalized = command.lower().strip()
        if self._is_root_recursive_delete(normalized):
            return PermissionDecision(
                action="deny",
                reason="dangerous_command",
                audit_type="DANGER",
                command=command,
                message="Blocked dangerous PowerShell command.",
            )
        if self._matches_any(normalized, self.CRITICAL_COMMAND_PATTERNS):
            return PermissionDecision(
                action="ask",
                reason="critical_command",
                audit_type="CRITICAL",
                command=command,
                message="This system-level or irreversible command requires one-time approval.",
            )
        if self._matches_any(normalized, self.DANGEROUS_COMMAND_PATTERNS):
            return PermissionDecision(
                action="deny",
                reason="dangerous_command",
                audit_type="DANGER",
                command=command,
                message="Blocked dangerous PowerShell command.",
            )
        if self._matches_any(normalized, self.WRITE_COMMAND_PATTERNS):
            if not self._sandbox_confirmed():
                return self._sandbox_ask(command)
            if self._has_absolute_path_outside_sandbox(command):
                return PermissionDecision(
                    action="ask",
                    reason="outside_sandbox",
                    audit_type="CRITICAL",
                    command=command,
                    message="PowerShell modification outside sandbox requires one-time approval.",
                )
            if self._is_sandbox_auto_allowed_command(command):
                return PermissionDecision(action="allow")
            return PermissionDecision(
                action="ask",
                reason="powershell_write",
                audit_type="WARNING",
                command=command,
                message="PowerShell command may modify files or install software.",
                rewritten_args={"command": self._sandboxed_powershell(command)},
            )
        return PermissionDecision(action="allow")

    def _sandbox_confirmed(self):
        return bool(self.config.get("sandbox_confirmed") and self.config.get("sandbox_path"))

    def _sandbox_ask(self, command):
        sandbox_path = str(self.sandbox_path)
        return PermissionDecision(
            action="ask",
            reason="sandbox_unconfirmed",
            audit_type="SANDBOX",
            command=str(command),
            sandbox_path=sandbox_path,
            message=f"Confirm writable sandbox: {sandbox_path}",
        )

    def _target_path(self, args):
        raw = None
        for key in self.FILE_PATH_KEYS:
            if args.get(key):
                raw = str(args[key])
                break
        if not raw:
            return None
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.sandbox_path / raw).resolve()

    def _is_inside_sandbox(self, target):
        try:
            os.path.commonpath([str(self.sandbox_path), str(target)])
            return os.path.commonpath([str(self.sandbox_path), str(target)]) == str(self.sandbox_path)
        except ValueError:
            return False

    def _matches_any(self, text, patterns):
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    def _is_root_recursive_delete(self, command):
        if "remove-item" not in command and not re.search(r"\brm\b", command):
            return False
        if "-recurse" not in command and "-rf" not in command:
            return False
        return bool(re.search(r"\b[a-z]:\\(?:\s|$)", command) or re.search(r"\brm\s+-rf\s+[/~]", command))

    def _has_absolute_path_outside_sandbox(self, command):
        for raw_path in self._absolute_paths_in_command(command):
            target = Path(raw_path).resolve()
            if not self._is_inside_sandbox(target):
                return True
        return False

    def _absolute_paths_in_command(self, command):
        paths = []
        for raw_path in re.findall(r"[a-zA-Z]:\\[^\s'\"`|;&)}]+", command):
            paths.append(raw_path.rstrip(".,"))
        return paths

    def _working_directory_from_command(self, command):
        match = re.search(
            r"(?:^|[;&]\s*)(?:cd|set-location)(?:\s+-literalpath)?\s+['\"]?([^;'\"]+)",
            command,
            re.IGNORECASE,
        )
        if not match:
            return None
        raw = match.group(1).strip()
        if not raw:
            return None
        try:
            return Path(raw).resolve()
        except Exception:
            return None

    def _is_sandbox_scoped_command(self, command):
        working_dir = self._working_directory_from_command(command)
        if working_dir and self._is_inside_sandbox(working_dir):
            return True
        paths = self._absolute_paths_in_command(command)
        return bool(paths) and all(self._is_inside_sandbox(Path(path).resolve()) for path in paths)

    def _is_sandbox_auto_allowed_command(self, command):
        if not self._is_sandbox_scoped_command(command):
            return False
        normalized = command.lower()
        if re.search(r"\bpip\s+install\b", normalized) and "--dry-run" not in normalized:
            return False
        auto_allowed_patterns = [
            r"\buv\s+(init|add|sync|run|lock)\b",
            r"\bpip\s+index\b",
            r"\bpip\s+install\b.*--dry-run",
            r"\binvoke-webrequest\b",
            r"\bexpand-archive\b",
            r"\bnew-item\b",
            r"\bmkdir\b",
            r"\bffmpeg(?:\.exe)?\b",
        ]
        return self._matches_any(normalized, auto_allowed_patterns)

    def _sandboxed_powershell(self, command):
        sandbox = str(self.sandbox_path).replace("'", "''")
        return f"Set-Location -LiteralPath '{sandbox}'; {command}"
