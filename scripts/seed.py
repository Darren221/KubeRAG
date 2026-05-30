"""Seed the indexes with a small curated subset of the Kubernetes docs.

Designed to run once on first boot (e.g. as a `seed` service in docker-compose).
If the indexes are already populated, the script exits 0 without doing work.

The seed corpus is ~28 canonical docs covering the topics in the golden Q&A
dataset, fetched directly from the kubernetes/website GitHub raw URLs.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import urllib.request
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory

from openai import AsyncOpenAI

from kuberag.config import Settings
from kuberag.ingest.chunkers import Chunker, FixedSizeChunker, RecursiveChunker
from kuberag.ingest.embedder import Embedder, EmbeddingCache
from kuberag.ingest.pipeline import IngestPipeline, IngestResult
from kuberag.stores import BM25Store, ChromaStore

SEED_REPO_RAW = "https://raw.githubusercontent.com/kubernetes/website"

# A valid git ref: SHA, tag, or branch name. Disallows '/' and other characters
# that could escape the URL path segment — without this, KUBERAG_SEED_REF
# could be set to '../../attacker/repo/main' and silently swap the corpus.
_REF_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _validate_seed_ref(ref: str) -> str:
    if not _REF_RE.match(ref):
        raise ValueError(
            f"invalid KUBERAG_SEED_REF: {ref!r} (must match {_REF_RE.pattern})"
        )
    return ref


# Pinned for reproducibility — override with KUBERAG_SEED_REF if you need a
# different commit, branch, or tag. Default is the k8s/website main HEAD at
# the time the README eval numbers were verified.
SEED_REF = _validate_seed_ref(
    os.environ.get("KUBERAG_SEED_REF", "06a3cd92aed8ca35c9fd966bb153dd46e21306e2")
)

SEED_DOCS: list[str] = [
    # Core workload concepts
    "content/en/docs/concepts/workloads/pods/_index.md",
    "content/en/docs/concepts/workloads/pods/pod-lifecycle.md",
    "content/en/docs/concepts/workloads/pods/init-containers.md",
    # Controllers
    "content/en/docs/concepts/workloads/controllers/deployment.md",
    "content/en/docs/concepts/workloads/controllers/replicaset.md",
    "content/en/docs/concepts/workloads/controllers/statefulset.md",
    "content/en/docs/concepts/workloads/controllers/daemonset.md",
    "content/en/docs/concepts/workloads/controllers/job.md",
    "content/en/docs/concepts/workloads/controllers/cron-jobs.md",
    # Networking
    "content/en/docs/concepts/services-networking/service.md",
    "content/en/docs/concepts/services-networking/ingress.md",
    "content/en/docs/concepts/services-networking/network-policies.md",
    # Configuration
    "content/en/docs/concepts/configuration/configmap.md",
    "content/en/docs/concepts/configuration/secret.md",
    "content/en/docs/concepts/configuration/manage-resources-containers.md",
    # Storage
    "content/en/docs/concepts/storage/volumes.md",
    "content/en/docs/concepts/storage/persistent-volumes.md",
    # Scheduling
    "content/en/docs/concepts/scheduling-eviction/kube-scheduler.md",
    "content/en/docs/concepts/scheduling-eviction/taint-and-toleration.md",
    # Architecture / overview
    "content/en/docs/concepts/architecture/nodes.md",
    "content/en/docs/concepts/overview/components.md",
    "content/en/docs/concepts/overview/working-with-objects/namespaces.md",
    "content/en/docs/concepts/overview/working-with-objects/labels.md",
    # Auth
    "content/en/docs/reference/access-authn-authz/rbac.md",
    "content/en/docs/reference/access-authn-authz/service-accounts-admin.md",
    # Tasks
    "content/en/docs/tasks/access-application-cluster/port-forward-access-application-cluster.md",
    "content/en/docs/tasks/debug/debug-application/debug-pods.md",
    "content/en/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes.md",
]


class SeedAction(Enum):
    SKIP = "skip"      # both stores populated and consistent — no work needed
    SEED = "seed"      # both stores empty — first-boot seed
    RESEED = "reseed"  # counts disagree — reset both stores and re-seed


def decide_seed_action(
    chroma_store: ChromaStore, bm25_store: BM25Store
) -> SeedAction:
    """Pure predicate: inspect the stores and decide what to do.

    Any disagreement between the two counts is treated as a partial-seed
    artifact and resolved by re-seeding from scratch. Printing/logging and
    the actual reset are the caller's responsibility.
    """
    chroma_count = chroma_store.count()
    bm25_count = bm25_store.count()
    if chroma_count != bm25_count:
        return SeedAction.RESEED
    if chroma_count == 0:
        return SeedAction.SEED
    return SeedAction.SKIP


def download_doc(rel_path: str, target_dir: Path, *, ref: str = SEED_REF) -> Path:
    # Defence-in-depth: reject any rel_path that would write outside
    # target_dir. SEED_DOCS is a hardcoded constant today, but this guard
    # makes the function safe if anyone later wires it to read paths from
    # external input.
    target_root = target_dir.resolve()
    local_path = (target_dir / rel_path).resolve()
    if not local_path.is_relative_to(target_root):
        raise ValueError(f"path traversal in rel_path: {rel_path!r}")

    url = f"{SEED_REPO_RAW}/{ref}/{rel_path}"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        local_path.write_bytes(resp.read())
    return local_path


def download_seed_corpus(
    target_dir: Path, *, ref: str = SEED_REF
) -> list[Path]:
    return [download_doc(p, target_dir, ref=ref) for p in SEED_DOCS]


def build_pipeline(
    settings: Settings,
    client: AsyncOpenAI,
    chroma_store: ChromaStore,
    bm25_store: BM25Store,
) -> IngestPipeline:
    embedder = Embedder(
        model=settings.embedding_model,
        client=client,
        cache=EmbeddingCache(settings.chroma_path.parent / "embedding_cache"),
    )
    chunkers: dict[str, Chunker] = {
        "fixed": FixedSizeChunker(
            size=settings.chunk_size, overlap=settings.chunk_overlap
        ),
        "recursive": RecursiveChunker(
            size=settings.chunk_size, overlap=settings.chunk_overlap
        ),
    }
    return IngestPipeline(
        chunkers=chunkers,
        embedder=embedder,
        chroma_store=chroma_store,
        bm25_store=bm25_store,
        dedupe_threshold=settings.dedupe_threshold,
    )


async def seed() -> IngestResult | None:
    settings = Settings()  # type: ignore[call-arg]
    chroma = ChromaStore(settings.chroma_path)
    bm25 = BM25Store(settings.bm25_path)

    action = decide_seed_action(chroma, bm25)

    if action is SeedAction.SKIP:
        print(
            f"[seed] indexes already populated ({chroma.count()} chunks); skipping."
        )
        return None

    if action is SeedAction.RESEED:
        # Wipe both stores so the next ingest starts from a clean slate.
        # Otherwise pipeline.run would deduplicate against the stale chroma
        # entries and raise IngestIntegrityError when counts still diverge.
        print(
            f"[seed] WARNING: chroma={chroma.count()} != bm25={bm25.count()}; "
            "indexes are out of sync (likely from a prior crashed seed). "
            "Resetting both stores and re-seeding from scratch."
        )
        chroma.reset()
        bm25.reset()

    print(f"[seed] downloading {len(SEED_DOCS)} k8s docs from kubernetes/website...")
    with TemporaryDirectory() as td:
        target_dir = Path(td)
        paths = download_seed_corpus(target_dir)
        print(f"[seed] downloaded {len(paths)} files; running ingest...")

        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        pipeline = build_pipeline(settings, client, chroma, bm25)
        result = await pipeline.run(paths, chunker_name="recursive")

    print(
        f"[seed] done. inserted={result.chunks_inserted}, "
        f"skipped_duplicate={result.chunks_skipped_duplicate}, "
        f"chroma_count={result.chroma_count}"
    )
    return result


def main() -> int:
    asyncio.run(seed())
    return 0


if __name__ == "__main__":
    sys.exit(main())
