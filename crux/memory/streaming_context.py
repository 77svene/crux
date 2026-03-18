import asyncio
import heapq
import time
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class RelevanceTier(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FILTERED = "filtered"


@dataclass(order=True)
class ScoredChunk:
    sort_key: float
    chunk_id: str = field(compare=False)
    content: str = field(compare=False)
    metadata: Dict[str, Any] = field(compare=False)
    score: float = field(compare=False)
    relevance_tier: RelevanceTier = field(compare=False)
    retrieval_latency_ms: float = field(compare=False)
    source: str = field(compare=False)
    timestamp: Optional[float] = field(compare=False)


@dataclass
class StreamingConfig:
    max_concurrent_scoring: int = 10
    score_timeout_seconds: float = 30.0
    enable_early_exit: bool = True
    min_relevance_threshold: float = 0.3
    high_relevance_threshold: float = 0.7
    medium_relevance_threshold: float = 0.5
    batch_yield_size: int = 1
    yield_on_count: Optional[List[int]] = None
    max_chunks_to_score: int = 1000
    use_cache: bool = True


@dataclass
class StreamingResult:
    query: str
    chunks: List[ScoredChunk]
    total_scored: int
    total_available: int
    retrieval_time_ms: float
    is_complete: bool
    early_exit_reason: Optional[str] = None


class PriorityStreamQueue:
    def __init__(self, max_size: int = 10):
        self._heap: List[ScoredChunk] = []
        self._seen: set = set()
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def try_add(self, chunk: ScoredChunk) -> bool:
        async with self._lock:
            if chunk.chunk_id in self._seen:
                return False
            self._seen.add(chunk.chunk_id)

            if len(self._heap) < self._max_size:
                heapq.heappush(self._heap, chunk)
                return True
            elif chunk.score > self._heap[0].score:
                heapq.heapreplace(self._heap, chunk)
                return True
            return False

    async def get_all(self) -> List[ScoredChunk]:
        async with self._lock:
            return sorted(self._heap, key=lambda x: -x.score)

    async def get_top(self, n: int) -> List[ScoredChunk]:
        async with self._lock:
            if n >= len(self._heap):
                return sorted(self._heap, key=lambda x: -x.score)
            return sorted(self._heap, key=lambda x: -x.score)[:n]

    async def peek_top_score(self) -> Optional[float]:
        async with self._lock:
            return self._heap[0].score if self._heap else None

    @property
    def size(self) -> int:
        return len(self._heap)


class ChunkScorer:
    def __init__(
        self,
        embedding_fn: Optional[Callable] = None,
        config: Optional[StreamingConfig] = None
    ):
        self.config = config or StreamingConfig()
        self.embedding_fn = embedding_fn
        self._score_cache: Dict[str, float] = {}
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_scoring)

    def _compute_relevance_tier(self, score: float) -> RelevanceTier:
        if score >= self.config.high_relevance_threshold:
            return RelevanceTier.HIGH
        elif score >= self.config.medium_relevance_threshold:
            return RelevanceTier.MEDIUM
        elif score >= self.config.min_relevance_threshold:
            return RelevanceTier.LOW
        return RelevanceTier.FILTERED

    def _get_cache_key(self, chunk_id: str, query: str) -> str:
        return f"{chunk_id}:{hash(query)}"

    async def score_chunk(
        self,
        chunk_id: str,
        chunk_content: str,
        query: str,
        metadata: Dict[str, Any]
    ) -> Optional[ScoredChunk]:
        cache_key = self._get_cache_key(chunk_id, query)

        if self.config.use_cache and cache_key in self._score_cache:
            score = self._score_cache[cache_key]
        else:
            score = await self._compute_similarity(chunk_content, query)
            if self.config.use_cache:
                self._score_cache[cache_key] = score

        if score < self.config.min_relevance_threshold:
            return None

        start_time = metadata.get('_retrieval_start', time.time())
        retrieval_latency = (time.time() - start_time) * 1000

        scored_chunk = ScoredChunk(
            sort_key=-score,
            chunk_id=chunk_id,
            content=chunk_content,
            metadata=metadata,
            score=score,
            relevance_tier=self._compute_relevance_tier(score),
            retrieval_latency_ms=retrieval_latency,
            source=metadata.get('source', 'unknown'),
            timestamp=metadata.get('timestamp')
        )

        return scored_chunk

    async def _compute_similarity(
        self,
        chunk_content: str,
        query: str
    ) -> float:
        loop = asyncio.get_event_loop()

        if self.embedding_fn:
            def _sync_embed():
                return self.embedding_fn(chunk_content), self.embedding_fn(query)

            chunk_emb, query_emb = await loop.run_in_executor(
                self._executor, _sync_embed
            )
            return self._cosine_similarity(chunk_emb, query_emb)
        else:
            return await loop.run_in_executor(
                self._executor,
                self._keyword_similarity,
                chunk_content,
                query
            )

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def clear_cache(self):
        self._score_cache.clear()


class StreamingContext:
    def __init__(
        self,
        memory_store=None,
        embedding_fn: Optional[Callable] = None,
        config: Optional[StreamingConfig] = None
    ):
        self.memory_store = memory_store
        self.config = config or StreamingConfig()
        self.scorer = ChunkScorer(
            embedding_fn=embedding_fn,
            config=self.config
        )
        self._yield_callbacks: List[Callable] = []
        self._result_buffer: Dict[str, List[ScoredChunk]] = {}
        self._active_streams: Dict[str, asyncio.Task] = {}

    def add_yield_callback(self, callback: Callable):
        self._yield_callbacks.append(callback)

    async def streaming_retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        memory_type: Optional[str] = None
    ) -> AsyncGenerator[StreamingResult, None]:
        stream_id = f"{user_id}:{query}:{time.time()}"
        priority_queue = PriorityStreamQueue(max_size=top_k)
        scored_count = 0
        total_available = 0
        start_time = time.time()
        early_exit_reason = None
        high_quality_found = 0

        yield_count_targets = self.config.yield_on_count or [1, 3, 5]
        if top_k not in yield_count_targets:
            yield_count_targets = sorted(set(yield_count_targets + [top_k]))

        try:
            chunks = await self._fetch_chunks(
                query=query,
                user_id=user_id,
                filters=filters,
                memory_type=memory_type
            )
            total_available = len(chunks)

            scoring_tasks = []
            for chunk in chunks[:self.config.max_chunks_to_score]:
                task = asyncio.create_task(
                    self._score_and_queue(chunk, query, priority_queue)
                )
                scoring_tasks.append(task)

                if len(scoring_tasks) >= self.config.max_concurrent_scoring:
                    done, pending = await asyncio.wait(
                        scoring_tasks,
                        timeout=self.config.score_timeout_seconds,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        result = await task
                        scored_count += 1
                        if result:
                            high_quality_found += 1

                        if self._should_yield(scored_count, priority_queue.size, yield_count_targets):
                            current_chunks = await priority_queue.get_top(top_k)
                            yield StreamingResult(
                                query=query,
                                chunks=current_chunks,
                                total_scored=scored_count,
                                total_available=total_available,
                                retrieval_time_ms=(time.time() - start_time) * 1000,
                                is_complete=False
                            )

                        if self.config.enable_early_exit:
                            if await self._check_early_exit(
                                priority_queue, high_quality_found, scored_count
                            ):
                                early_exit_reason = "sufficient_high_quality_results"
                                break

                    scoring_tasks = list(pending)

            if scoring_tasks:
                remaining_results = await asyncio.gather(*scoring_tasks, return_exceptions=True)
                for result in remaining_results:
                    if isinstance(result, ScoredChunk):
                        scored_count += 1
                        if result.relevance_tier == RelevanceTier.HIGH:
                            high_quality_found += 1

            final_chunks = await priority_queue.get_all()

            for callback in self._yield_callbacks:
                try:
                    await callback(stream_id, final_chunks)
                except Exception as e:
                    logger.warning(f"Yield callback error: {e}")

            yield StreamingResult(
                query=query,
                chunks=final_chunks,
                total_scored=scored_count,
                total_available=total_available,
                retrieval_time_ms=(time.time() - start_time) * 1000,
                is_complete=True,
                early_exit_reason=early_exit_reason
            )

        except asyncio.CancelledError:
            logger.info(f"Stream {stream_id} cancelled")
            yield StreamingResult(
                query=query,
                chunks=await priority_queue.get_all(),
                total_scored=scored_count,
                total_available=total_available,
                retrieval_time_ms=(time.time() - start_time) * 1000,
                is_complete=False,
                early_exit_reason="cancelled"
            )
        except Exception as e:
            logger.error(f"Streaming retrieval error: {e}")
            raise

    async def _fetch_chunks(
        self,
        query: str,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        chunks = []

        if self.memory_store:
            try:
                if hasattr(self.memory_store, 'search'):
                    results = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.memory_store.search(
                            query=query,
                            user_id=user_id,
                            limit=self.config.max_chunks_to_score,
                            filters=filters
                        )
                    )
                    if isinstance(results, list):
                        chunks = results
                    elif hasattr(results, 'results'):
                        chunks = results.results
                    elif hasattr(results, 'data'):
                        chunks = results.data

                elif hasattr(self.memory_store, 'get_all'):
                    all_memories = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.memory_store.get_all(user_id=user_id)
                    )
                    for memory in all_memories or []:
                        chunks.append({
                            'id': memory.get('id', memory.get('_id')),
                            'content': memory.get('content', memory.get('text', '')),
                            'metadata': memory.get('metadata', {}),
                            'source': memory.get('source', 'memory'),
                            'timestamp': memory.get('created_at', memory.get('timestamp'))
                        })
                        if len(chunks) >= self.config.max_chunks_to_score:
                            break

                for chunk in chunks:
                    if '_retrieval_start' not in chunk.get('metadata', {}):
                        if 'metadata' not in chunk:
                            chunk['metadata'] = {}
                        chunk['metadata']['_retrieval_start'] = time.time()

            except Exception as e:
                logger.warning(f"Memory store fetch error: {e}")

        return chunks

    async def _score_and_queue(
        self,
        chunk: Dict[str, Any],
        query: str,
        queue: PriorityStreamQueue
    ) -> Optional[ScoredChunk]:
        try:
            scored = await self.scorer.score_chunk(
                chunk_id=str(chunk.get('id', chunk.get('_id', 'unknown'))),
                chunk_content=chunk.get('content', chunk.get('text', '')),
                query=query,
                metadata=chunk.get('metadata', {})
            )

            if scored:
                await queue.try_add(scored)
                return scored
            return None

        except Exception as e:
            logger.warning(f"Scoring error for chunk {chunk.get('id')}: {e}")
            return None

    def _should_yield(
        self,
        scored_count: int,
        queue_size: int,
        yield_targets: List[int]
    ) -> bool:
        return queue_size in yield_targets or (
            self.config.batch_yield_size > 0 and
            scored_count % self.config.batch_yield_size == 0
        )

    async def _check_early_exit(
        self,
        queue: PriorityStreamQueue,
        high_quality_count: int,
        total_scored: int
    ) -> bool:
        if queue.size < 3:
            return False

        top_score = await queue.peek_top_score()
        if top_score and top_score >= self.config.high_relevance_threshold:
            if high_quality_count >= 3:
                coverage = high_quality_count / total_scored if total_scored > 0 else 0
                if coverage >= 0.1 or high_quality_count >= min(5, queue.size):
                    return True

        if queue.size >= 5 and total_scored >= 20:
            return True

        return False

    async def get_progressive_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        max_results: int = 5,
        format_fn: Optional[Callable] = None
    ) -> AsyncGenerator[str, None]:
        async for result in self.streaming_retrieve(
            query=query,
            user_id=user_id,
            top_k=max_results
        ):
            if format_fn:
                yield format_fn(result.chunks)
            else:
                yield self._default_format(result.chunks)

    def _default_format(self, chunks: List[ScoredChunk]) -> str:
        if not chunks:
            return ""

        formatted_parts = []
        for i, chunk in enumerate(chunks):
            tier_indicator = {
                RelevanceTier.HIGH: "+++",
                RelevanceTier.MEDIUM: "++",
                RelevanceTier.LOW: "+"
            }.get(chunk.relevance_tier, "")

            formatted_parts.append(
                f"[{tier_indicator}] (score: {chunk.score:.3f}) {chunk.content}"
            )

        return "\n\n".join(formatted_parts)


class StreamingMemoryRetrieval:
    def __init__(
        self,
        memory_instance=None,
        config: Optional[StreamingConfig] = None
    ):
        self.memory_instance = memory_instance
        self.config = config or StreamingConfig()
        self.streaming_context = StreamingContext(
            memory_store=memory_instance,
            config=self.config
        )

    async def retrieve_streaming(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> AsyncGenerator[StreamingResult, None]:
        async for result in self.streaming_context.streaming_retrieve(
            query=