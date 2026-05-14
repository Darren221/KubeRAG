import subprocess
from collections.abc import Iterable
from pathlib import Path

_DEFAULT_SUBTREES: tuple[str, ...] = (
    "content/en/docs/concepts",
    "content/en/docs/tasks",
    "content/en/docs/reference",
)


def find_markdown_files(
    repo_root: Path,
    subtrees: Iterable[str] = _DEFAULT_SUBTREES,
) -> list[Path]:
    paths: list[Path] = []
    for subtree in subtrees:
        root = Path(repo_root) / subtree
        if root.exists():
            paths.extend(root.rglob("*.md"))
    return sorted(paths)


class K8sDocsSource:
    REPO_URL = "https://github.com/kubernetes/website.git"
    DEFAULT_BRANCH = "main"
    SUBTREES = _DEFAULT_SUBTREES
    REPO_DIR_NAME = "kubernetes-website"

    def __init__(
        self,
        *,
        repo_url: str | None = None,
        branch: str | None = None,
        commit: str | None = None,
    ) -> None:
        self.repo_url = repo_url or self.REPO_URL
        self.branch = branch or self.DEFAULT_BRANCH
        self.commit = commit

    def fetch(self, target_dir: Path) -> list[Path]:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        repo_root = target / self.REPO_DIR_NAME

        if not repo_root.exists():
            self._clone(repo_root)

        if self.commit is not None:
            self._checkout(repo_root, self.commit)

        return find_markdown_files(repo_root, subtrees=self.SUBTREES)

    def current_sha(self, repo_root: Path) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _clone(self, repo_root: Path) -> None:
        cmd: list[str]
        if self.commit is None:
            cmd = [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                self.branch,
                self.repo_url,
                str(repo_root),
            ]
        else:
            cmd = ["git", "clone", self.repo_url, str(repo_root)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _checkout(self, repo_root: Path, commit: str) -> None:
        subprocess.run(
            ["git", "checkout", commit],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
