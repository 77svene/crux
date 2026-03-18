```markdown
# ⚡ CRUX

### What Mem0 Should Have Been.

---

[![PyPI Version](https://img.shields.io/pypi/v/crux-mem.svg)](https://pypi.org/project/crux-mem/)
[![Python](https://img.shields.io/python-version/crux-mem)](https://pypi.org/project/crux-mem/)
[![Stars](https://img.shields.io/github/stars/mem0rias/crux?style=flat&logo=github)](https://github.com/mem0rias/crux)
[![License](https://img.shields.io/github/license/mem0rias/crux)](LICENSE)
[![Discord](https://img.shields.io/discord/join?style=flat&logo=discord)](https://discord.gg/crux)
[![Last Commit](https://img.shields.io/github/last-commit/mem0rias/crux/main)](https://github.com/mem0rias/crux/commits/main)

> **High-performance memory layer for AI agents.** Crux replaces Mem0's flat vector store with a production-ready graph architecture, multi-agent collaboration, temporal versioning, and enterprise observability — all while maintaining drop-in compatibility.

---

## 🚀 Why Switch Now?

Mem0 solved the right problem but built it on borrowed time. Flat embeddings, no versioning, no observability, single-agent only.

**Crux is what you get when you rebuild it for production from day one.**

| Feature | Mem0 | Crux |
|---------|:----:|:----:|
| **Memory Architecture** | Flat vector store | Graph-based with entity relationships |
| **Multi-Agent Support** | ❌ Per-agent isolation | ✅ Shared memory with conflict resolution |
| **Version Control** | ❌ None | ✅ Full temporal versioning with rollback |
| **Time-Travel Queries** | ❌ No | ✅ Query any past memory state |
| **Observability** | ❌ Basic metrics | ✅ Built-in dashboard, decay viz, hygiene |
| **Memory Pruning** | Manual | ✅ Automatic with configurable policies |
| **LLM Flexibility** | OpenAI-first | ✅ Hot-swappable (OpenAI, Anthropic, Ollama, Azure) |
| **Backend Options** | Redis/PostgreSQL | Redis, PostgreSQL, **or both together** |
| **Kubernetes Ready** | Community contribs | ✅ First-class support |
| **Async Architecture** | Sync-only | ✅ Full async with sync wrappers |
| **API Compatibility** | — | ✅ 90%+ drop-in for Mem0 |

---

## ⚡ Quickstart

```bash
pip install crux-mem
```

### Basic Memory Operations

```python
import crux

# Initialize with your preferred backend
memory = crux.Memory(
    llm="anthropic/claude-3-5-sonnet",
    backend={"type": "postgres", "url": "postgresql://localhost/crux"},
)

# Store memories with automatic entity extraction
await memory.add(
    "User prefers dark mode. Name is Sarah. Works at Acme Corp as ML Engineer.",
    user_id="sarah_001",
    metadata={"source": "onboarding", "priority": "high"}
)

# Retrieve with temporal awareness — get state at any point
context = await memory.search(
    "What are Sarah's preferences?",
    user_id="sarah_001",
)

# Query historical state
historical = await memory.search(
    "What did Sarah say about her preferences last Tuesday?",
    user_id="sarah_001",
    temporal={"at": "2024-01-09T14:30:00Z"},
)
```

### Multi-Agent Collaboration

```python
# Multiple agents sharing a memory space
orchestrator = crux.MultiAgent(
    memory=memory,
    namespace="project-phoenix",
    conflict_resolution="last-write-wins",  # or "agent-vote", "manual"
)

# Agent 1 writes context
await orchestrator.add(
    "Analyzed Q4 revenue: 23% growth, EMEA outperforming APAC",
    agent_id="analyst-alpha",
    entities=["Q4", "revenue", "EMEA", "APAC"],
)

# Agent 2 builds on it — gets linked context automatically
await orchestrator.add(
    "EMEA growth driven by enterprise segment. APAC needs pipeline attention.",
    agent_id="strategist-beta",
    related_to=["Q4", "EMEA", "APAC"],  # Explicit graph links
)

# Agent 3 queries the collective knowledge
insights = await orchestrator.query(
    "What's driving EMEA performance?",
    agent_id="strategist-beta",
    include_agents=["analyst-alpha", "strategist-beta"],
)
```

### Temporal Versioning & Rollback

```python
# View memory history
history = await memory.history(
    user_id="sarah_001",
    limit=50,
    filter={"changed_entities": ["preferences"]},
)

# Compare two states
diff = await memory.diff(
    user_id="sarah_001",
    from_version="2024-01-08T00:00:00Z",
    to_version="2024-01-10T00:00:00Z",
)
# Returns: {added: [...], removed: [...], modified: [...]}

# Rollback to previous state
await memory.rollback(
    user_id="sarah_001",
    version="2024-01-08T00:00:00Z",
)
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CRUX CORE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Ingestion │    │   Query     │    │  Temporal Engine   │  │
│  │   Pipeline  │───▶│   Engine    │◀───│  (Version Control)  │  │
│  └──────┬──────┘    └──────┬──────┘    └─────────┬───────────┘  │
│         │                  │                     │              │
│         ▼                  ▼                     ▼              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              GRAPH MEMORY STORE                          │  │
│  │  ┌─────────┐    ┌────────────┐    ┌─────────────────┐    │  │
│  │  │ Nodes   │◀──▶│  Edges     │◀──▶│  Temporal       │    │  │
│  │  │(Facts)  │    │(Relations) │    │  Snapshots      │    │  │
│  │  └─────────┘    └────────────┘    └─────────────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Entity     │    │  Relevance  │    │  Multi-Agent       │  │
│  │  Extractor  │    │  Scorer     │    │  Coordinator       │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Redis    │   │PostgreSQL│   │ Hybrid   │
        │ (Cache)  │   │(Primary) │   │ (Both)   │
        └──────────┘   └──────────┘   └──────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Graph Memory Store** | Stores facts as nodes, relationships as typed edges. Enables contextual recall, not just semantic search. |
| **Temporal Engine** | Maintains immutable history of every change. Supports branching, diffing, and instant rollback. |
| **Entity Extractor** | Identifies and normalizes entities (people, places, concepts) across memories. |
| **Relevance Scorer** | Scores memories by recency, access frequency, importance, and decay. |
| **Multi-Agent Coordinator** | Manages shared namespaces, resolves write conflicts, tracks agent contributions. |
| **Memory Hygiene** | Automatic pruning of stale entries, deduplication, and compression. |

---

## 📦 Installation

### Standard Installation

```bash
pip install crux-mem
```

### With Optional Dependencies

```bash
# PostgreSQL support
pip install "crux-mem[postgres]"

# Redis cache layer
pip install "crux-mem[redis]"

# Full production stack
pip install "crux-mem[postgres,redis,monitoring]"

# All backends and LLMs
pip install "crux-mem[all]"
```

### Docker (One-Command Deploy)

```bash
# Standalone microservice
docker run -d \
  --name crux \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/crux \
  -e REDIS_URL=redis://host:6379 \
  crux-mem/server:latest
```

### Kubernetes

```yaml
# helm install crux ./charts/crux
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crux
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: crux
        image: crux-mem/server:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: crux-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://crux-redis:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

---

## 🔧 Configuration

```python
from crux import Memory, MemoryConfig

config = MemoryConfig(
    # LLM Configuration
    llm="openai/gpt-4-turbo",  # or "anthropic/claude-3", "ollama/llama2"
    llm_config={"temperature": 0.7, "max_tokens": 2000},
    
    # Storage Backend
    backend={
        "type": "hybrid",  # "postgres", "redis", or "hybrid"
        "postgres": {"url": "postgresql://localhost/crux"},
        "redis": {"url": "redis://localhost:6379", "ttl": 3600},
    },
    
    # Memory Hygiene
    retention={
        "max_age_days": 90,
        "min_importance": 0.1,
        "prune_on_write": True,
        "deduplicate": True,
    },
    
    # Temporal Settings
    versioning={
        "enabled": True,
        "retention_days": 365,
        "auto_snapshot_interval": 3600,  # seconds
    },
    
    # Observability
    observability={
        "metrics_enabled": True,
        "dashboard_port": 9090,
        "export_prometheus": True,
    },
)

memory = Memory(config)
```

### Environment Variables

```bash
# Core
CRUX_LLM=openai/gpt-4-turbo
CRUX_API_KEY=sk-...

# Storage
DATABASE_URL=postgresql://localhost/crux
REDIS_URL=redis://localhost:6379

# Observability
CRUX_DASHBOARD_PORT=9090
CRUX_METRICS_ENABLED=true
```

---

## 🌐 REST API

```bash
# Start the API server
crux server --port 8000

# Store memory
curl -X POST http://localhost:8000/memory \
  -H "Content-Type: application/json" \
  -H "X-User-ID: sarah_001" \
  -d '{"content": "User prefers dark mode", "metadata": {"source": "settings"}}'

# Search memories
curl "http://localhost:8000/memory/search?q=preferences&user_id=sarah_001"

# Query historical state
curl "http://localhost:8000/memory/search?q=preferences&temporal_at=2024-01-08T00:00:00Z"

# Get history
curl "http://localhost:8000/memory/history?user_id=sarah_001&limit=50"

# Rollback
curl -X POST "http://localhost:8000/memory/rollback" \
  -d '{"user_id": "sarah_001", "version": "2024-01-08T00:00:00Z"}'
```

---

## 📊 Enterprise Dashboard

Memory hygiene visualization, entity graphs, and relevance scoring — built-in.

```bash
crux dashboard --port 9090
```

```
┌─────────────────────────────────────────────────────────────────────┐
│  CRUX OBSERVABILITY DASHBOARD                        [Last 24h ▼]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Memory Health          Entity Graph          Relevance Dist.      │
│  ┌─────────────┐       ┌──────────┐        ┌──────────────────┐   │
│  │ ████████░░  │       │  User ●──┼──●Pref │ ▁▂▄▇█▇▄▂▁▁▂▄▇█▇▄ │   │
│  │  78% fresh  │       │          │    │   │                  │   │
│  │             │       │  ● Acme  │    │   │                  │   │
│  │ Entries: 2.4K│       │  (org)──┼──●ML  │  Memory Decay     │   │
│  │ Stale: 523  │       │          │ (role)│  ████████████░░░ │   │
│  └─────────────┘       └──────────┘       └──────────────────┘   │
│                                                                     │
│  Agent Activity                  Actions                            │
│  ┌────────────────────┐       ┌────────────────────────────────┐ │
│  │ analyst-alpha  ████ │       │ [Prune Stale] [Export] [Backup]│ │
│  │ strategist-β   ███ │       └────────────────────────────────┘ │
│  │ monitor-γ      ██  │                                           │
│  └────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🤝 Migration from Mem0

```python
# Before (Mem0)
from mem0 import Memory
memory = Memory()
memory.add("User likes dark mode", user_id="123")
results = memory.search("preferences", user_id="123")

# After (Crux) — 90% compatible
from crux import Memory
memory = Memory()
await memory.add("User likes dark mode", user_id="123")
results = await memory.search("preferences", user_id="123")
```

**Key differences:**
1. Async-first: `add()` and `search()` are async — wrap with `asyncio.run()` or use in async context
2. Better returns: Results include entity extraction, temporal metadata, and relevance scores
3. Explicit types: `user_id` parameter added to all methods for clarity

```bash
# Migration helper
npx @crux/migrate --from mem0 --to crux --input ./mem0-export.json
```

---

## 🧪 Testing

```bash
# Run the full suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=crux --cov-report=html

# Quick smoke test
pytest tests/test_smoke.py -k "basic"
```

---

## 📝 License

Apache 2.0 — same as Mem0. Drop-in compatible, fully open.

---

<div align="center">

**Built for agents that ship.** Star to stay updated.

[![Stars](https://img.shields.io/github/stars/mem0rias/crux?style=social)](https://github.com/mem0rias/crux)
[![Twitter](https://img.shields.io/twitter/follow/cruxmem?style=social)](https://twitter.com/cruxmem)

</div>
```