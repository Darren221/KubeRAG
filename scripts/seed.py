"""Seed the indexes with a small curated subset of the Kubernetes docs.

Designed to run once on first boot (e.g. as a `seed` service in docker-compose).
If the indexes are already populated, the script exits 0 without doing work.

The seed corpus is ~28 canonical docs covering the topics in the golden Q&A
dataset, fetched directly from the kubernetes/website GitHub raw URLs.
"""

from __future__ import annotations

import asyncio
import sys
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

from openai import AsyncOpenAI

from kuberag.config import Settings
from kuberag.ingest.chunkers import Chunker, FixedSizeChunker, RecursiveChunker
from kuberag.ingest.embedder import Embedder, EmbeddingCache
from kuberag.ingest.pipeline import IngestPipeline, IngestResult
from kuberag.stores import BM25Store, ChromaStore

# Pin to a specific commit for reproducibility. Update by setting SEED_COMMIT in env.
SEED_REPO_RAW = "https://raw.githubusercontent.com/kubernetes/website"
SEED_REF = "main"

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


def should_skip_seeding(chroma_store: ChromaStore) -> bool:
    return chroma_store.count() > 0


def download_doc(rel_path: str, target_dir: Path, *, ref: str = SEED_REF) -> Path:
    url = f"{SEED_REPO_RAW}/{ref}/{rel_path}"
    local_path = target_dir / rel_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        local_path.write_bytes(resp.read())
    return local_path


def download_seed_corpus(
    target_dir: Path, *, ref: str = SEED_REF
) -> list[Path]:
    return [download_doc(p, target_dir, ref=ref) for p in SEED_DOCS]


def build_pipeline(settings: Settings, client: AsyncOpenAI) -> IngestPipeline:
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
        chroma_store=ChromaStore(settings.chroma_path),
        bm25_store=BM25Store(settings.bm25_path),
        dedupe_threshold=settings.dedupe_threshold,
    )


async def seed() -> IngestResult | None:
    settings = Settings()  # type: ignore[call-arg]
    chroma = ChromaStore(settings.chroma_path)
    if should_skip_seeding(chroma):
        print(
            f"[seed] indexes already populated ({chroma.count()} chunks); skipping."
        )
        return None

    print(f"[seed] downloading {len(SEED_DOCS)} k8s docs from kubernetes/website...")
    with TemporaryDirectory() as td:
        target_dir = Path(td)
        paths = download_seed_corpus(target_dir)
        print(f"[seed] downloaded {len(paths)} files; running ingest...")

        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        pipeline = build_pipeline(settings, client)
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
