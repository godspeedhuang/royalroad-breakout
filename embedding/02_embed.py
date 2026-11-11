"""步驟 4：把 chunks_input.parquet 送去 Parley 做 embedding。

- 每次請求塞多塊文字，同時送多個請求（Parley 沒有 OpenAI 的 Batch API）
- 每個請求的結果寫成 out/emb_parts/ 底下一個 parquet，中斷後重跑會跳過已完成的塊
- 失敗自動重試；金鑰錯誤直接停止

用法：
  python 02_embed.py --limit-books 100   # 試跑
  python 02_embed.py                      # 全部
"""
import argparse
import asyncio
import random
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import dotenv_values
from openai import APIStatusError, AsyncOpenAI, AuthenticationError, PermissionDeniedError

HERE = Path(__file__).parent
OUT = HERE / "out"
PARTS = OUT / "emb_parts"
DIM = 1536


def load_done():
    files = list(PARTS.glob("*.parquet"))
    if not files:
        return set()
    return set(pq.read_table(files, columns=["chunk_uid"]).column("chunk_uid").to_pylist())


def pack(df, max_inputs, max_tokens):
    batch, n = [], 0
    for r in df.itertuples(index=False):
        if batch and (len(batch) >= max_inputs or n + r.n_tokens > max_tokens):
            yield batch
            batch, n = [], 0
        batch.append(r); n += r.n_tokens
    if batch:
        yield batch


async def embed_one(client, model, batch, sem, stats):
    texts = [r.text for r in batch]
    for attempt in range(8):
        try:
            async with sem:
                resp = await client.embeddings.create(model=model, input=texts, encoding_format="float")
            break
        except (AuthenticationError, PermissionDeniedError):
            raise
        except Exception as e:  # 429、5xx、逾時都重試
            stats["retries"] += 1
            wait = min(60, 2 ** attempt) + random.random()
            if attempt == 7:
                stats["failed"] += len(batch)
                print(f"  放棄一批（{len(batch)} 塊）：{type(e).__name__}: {str(e)[:120]}")
                return
            await asyncio.sleep(wait)
    vecs = np.asarray([d.embedding for d in sorted(resp.data, key=lambda d: d.index)], dtype=np.float32)
    assert vecs.shape == (len(batch), DIM), vecs.shape
    table = pa.table({
        "chunk_uid": [r.chunk_uid for r in batch],
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel()), DIM),
    })
    pq.write_table(table, PARTS / f"{uuid.uuid4().hex}.parquet")
    stats["done"] += len(batch); stats["tokens"] += sum(r.n_tokens for r in batch)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-books", type=int)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-inputs", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=120_000)
    a = ap.parse_args()

    env = dotenv_values(HERE / ".env")
    client = AsyncOpenAI(api_key=env["PARLEY_API_KEY"], base_url=env["PARLEY_BASE_URL"], timeout=120, max_retries=0)
    model = env.get("EMBEDDING_MODEL") or "text-embedding-3-small"

    PARTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(OUT / "chunks_input.parquet", columns=["chunk_uid", "fiction_id", "n_tokens", "text"])
    if a.limit_books:
        keep = pd.Series(df.fiction_id.unique()).sample(a.limit_books, random_state=0)
        df = df[df.fiction_id.isin(keep)]
    done = load_done()
    todo = df[~df.chunk_uid.isin(done)].sort_values(["fiction_id", "chunk_uid"])
    print(f"model={model}  待做 {len(todo):,} 塊（已完成 {len(df) - len(todo):,}），約 {todo.n_tokens.sum() / 1e6:.1f}M token")
    if todo.empty:
        return

    stats = dict(done=0, tokens=0, retries=0, failed=0)
    sem = asyncio.Semaphore(a.concurrency)
    batches = list(pack(todo, a.max_inputs, a.max_tokens))
    t0 = time.time()
    tasks = [asyncio.create_task(embed_one(client, model, b, sem, stats)) for b in batches]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        await fut
        if i % 50 == 0 or i == len(tasks):
            el = time.time() - t0
            print(f"  {i}/{len(tasks)} 請求｜{stats['done']:,} 塊｜{stats['tokens'] / 1e6:.1f}M token｜"
                  f"{el / 60:.1f} 分｜重試 {stats['retries']}｜失敗 {stats['failed']}", flush=True)
    print("完成", stats)


if __name__ == "__main__":
    asyncio.run(main())
