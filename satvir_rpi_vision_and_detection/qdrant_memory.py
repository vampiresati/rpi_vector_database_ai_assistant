"""Local Qdrant Edge memory for taught, seen, and ignored objects."""
import itertools
import re
import threading
import time
from pathlib import Path

from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    Point,
    PointVectors,
    Query,
    QueryRequest,
    RangeFloat,
    UpdateOperation,
)

import config

RECOGNIZE_THRESHOLD = config.RECOGNIZE_THRESHOLD
MAYBE_MARGIN = 0.05
SIGHTING_APART = 600
SHARD_DIRECTORY = "./qdrant-edge-directory"

CONFIG = EdgeConfig(
    vectors={
        "text": EdgeVectorParams(
            size=config.TEXT_EMBEDDING_DIM,
            distance=Distance.Cosine,
        ),
        "image": EdgeVectorParams(
            size=config.IMAGE_EMBEDDING_DIM,
            distance=Distance.Cosine,
        ),
    }
)


class Memory:
    """Store object memories and search them by image or text vector."""

    def __init__(self, data_dir=SHARD_DIRECTORY, threshold=RECOGNIZE_THRESHOLD,
                 where=None):
        self.threshold = threshold
        self.dir = Path(data_dir)
        self._lock = threading.RLock()
        self._where_file = self.dir / "where.txt"
        if where is None and self._where_file.exists():
            where = self._where_file.read_text(encoding="utf-8").strip() or None
        self.where = where

        others = [p for p in self.dir.glob("*")
                  if p.name not in ("thumbs", "where.txt")]
        if (self.dir / "segments").exists() or others:
            self.shard = EdgeShard.load(str(self.dir))
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.shard = EdgeShard.create(str(self.dir), CONFIG)
            for key, schema in (
                ("kind", PayloadSchemaType.Keyword),
                ("ts", PayloadSchemaType.Float),
            ):
                self.shard.update(UpdateOperation.create_field_index(
                    key, schema))
        self._ids = itertools.count(time.time_ns())

    def close(self):
        with self._lock:
            self.shard.close()

    def reopen(self):
        with self._lock:
            self.shard.close()
            self.shard = EdgeShard.load(str(self.dir))

    def set_where(self, place):
        place = " ".join((place or "").split())[:60] or None
        with self._lock:
            self.where = place
            self._where_file.write_text(place or "", encoding="utf-8")
        return place

    def count(self):
        from qdrant_edge import CountRequest
        with self._lock:
            return self.shard.count(CountRequest(exact=True))

    def _upsert(self, vector, payload):
        with self._lock:
            pid = next(self._ids)
            self.shard.update(UpdateOperation.upsert_points([
                Point(id=pid, vector=vector, payload=payload)
            ]))
            self.shard.flush()
            return pid

    def teach(self, image_vec, text_vec, label, transcript, ts=None, thumb=None,
              scene=None):
        return self._upsert(
            {"image": image_vec, "text": text_vec},
            {
                "kind": "taught",
                "label": label,
                "transcript": transcript,
                "ts": ts or time.time(),
                "thumb": thumb,
                "scene": scene,
                "where": self.where,
            },
        )

    def remember_sighting(self, image_vec, label, ts=None, thumb=None):
        return self._upsert(
            {"image": image_vec},
            {
                "kind": "seen",
                "label": label,
                "ts": ts or time.time(),
                "thumb": thumb,
                "where": self.where,
            },
        )

    def ignore(self, image_vec, thumb=None, ts=None):
        return self._upsert(
            {"image": image_vec},
            {
                "kind": "ignored",
                "ts": ts or time.time(),
                "thumb": thumb,
                "where": self.where,
            },
        )

    def forget(self, label):
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(
                ScrollRequest(limit=10000, with_payload=True))
            ids = [p.id for p in records
                   if (p.payload.get("label") or "").lower() == label.lower()]
            if ids:
                self.shard.update(UpdateOperation.delete_points(ids))
                self.shard.flush()
            return len(ids)

    def rename(self, label, new_label, text_vec=None):
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(
                ScrollRequest(limit=10000, with_payload=True))
            ids = [p.id for p in records
                   if (p.payload.get("label") or "").lower() == label.lower()]
            if ids:
                self.shard.update(
                    UpdateOperation.set_payload(ids, {"label": new_label}))
                if text_vec is not None:
                    taught = [p.id for p in records
                              if p.id in set(ids)
                              and p.payload.get("kind") == "taught"]
                    if taught:
                        self.shard.update(UpdateOperation.update_vectors(
                            [PointVectors(pid, {"text": text_vec})
                             for pid in taught]))
                self.shard.flush()
            return len(ids)

    def _ids_of_kind(self, kind, label):
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000,
                with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind", match=MatchValue(value=kind)),
                ]),
            ))
            return [p.id for p in records
                    if (p.payload.get("label") or "").lower() == label.lower()]

    def taught_ids(self, label):
        return self._ids_of_kind("taught", label)

    def seen_ids(self, label):
        return self._ids_of_kind("seen", label)

    def forget_point(self, pid):
        with self._lock:
            self.shard.update(UpdateOperation.delete_points([pid]))
            self.shard.flush()

    def unignore(self, pid):
        with self._lock:
            if pid not in [r.id for r in self.ignored()]:
                return False
            self.shard.update(UpdateOperation.delete_points([pid]))
            self.shard.flush()
            return True

    def ignored(self):
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000,
                with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="ignored")),
                ]),
            ))
        records.sort(key=lambda r: r.payload.get("ts") or 0, reverse=True)
        return records

    def match_ignored(self, image_vec):
        with self._lock:
            hits = self.shard.query(QueryRequest(
                query=Query.Nearest(image_vec, using="image"),
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="ignored")),
                ]),
                limit=1,
                with_payload=True,
            ))
        if hits and hits[0].score >= self.threshold:
            return hits[0]
        return None

    def count_sightings(self, label):
        from qdrant_edge import CountRequest
        with self._lock:
            return self.shard.count(CountRequest(
                exact=True,
                filter=Filter(must=[
                    FieldCondition(key="kind", match=MatchValue(value="seen")),
                    FieldCondition(key="label", match=MatchValue(value=label)),
                ]),
            ))

    def objects(self):
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000,
                with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="taught")),
                ]),
            ))
            groups = {}
            for r in records:
                label = r.payload.get("label") or ""
                g = groups.setdefault(label.lower(),
                                      {"label": label, "views": []})
                g["views"].append(dict(r.payload, id=r.id))
            out = []
            for g in groups.values():
                g["views"].sort(key=lambda p: p.get("ts") or 0, reverse=True)
                g["label"] = g["views"][0].get("label") or g["label"]
                g["seen"] = self.count_sightings(g["label"])
                g["sightings"] = [dict(r.payload, id=r.id) for r in
                                  self.last_sightings(g["label"], limit=None)]
                out.append(g)
            out.sort(key=lambda g: g["views"][0].get("ts") or 0, reverse=True)
            return out

    def recognize(self, image_vec):
        with self._lock:
            hits = self.shard.query(QueryRequest(
                query=Query.Nearest(image_vec, using="image"),
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="taught")),
                ]),
                limit=1,
                with_payload=True,
            ))
        if not hits:
            return None, 0.0, None
        top = hits[0]
        if top.score >= self.threshold:
            return top, top.score, None
        if top.score >= self.threshold - MAYBE_MARGIN:
            return None, top.score, top.payload.get("label")
        return None, top.score, None

    def names_in(self, question):
        from qdrant_edge import ScrollRequest

        def toks(text):
            out = []
            for word in re.split(r"[^0-9a-z]+", text.lower()):
                if word:
                    out.append(word[:-1]
                               if len(word) >= 4 and word.endswith("s")
                               else word)
            return out

        asked = toks(question)
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000,
                with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="taught")),
                ]),
            ))
        hits = {}
        for r in records:
            label = r.payload.get("label") or ""
            want = toks(label)
            if not want or (len(want) == 1 and len(want[0]) < 3):
                continue
            if any(asked[i:i + len(want)] == want
                   for i in range(len(asked) - len(want) + 1)):
                hits[label] = max(r.payload.get("ts") or 0,
                                  hits.get(label, 0))
        return [label for label, _ in
                sorted(hits.items(), key=lambda item: -item[1])]

    def best_taught(self, text_vec, labels=None):
        must = [FieldCondition(key="kind", match=MatchValue(value="taught"))]
        if labels:
            must.append(FieldCondition(
                key="label",
                match=MatchAny(list(labels)),
            ))
        with self._lock:
            hits = self.shard.query(QueryRequest(
                query=Query.Nearest(text_vec, using="text"),
                filter=Filter(must=must),
                limit=1,
                with_payload=True,
            ))
        return hits[0] if hits else None

    def forget_sighting_burst(self, pid):
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000,
                with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="seen")),
                ]),
            ))
            target = next((r for r in records if r.id == pid), None)
            if target is None:
                return 0
            label = (target.payload.get("label") or "").lower()
            ts = target.payload.get("ts") or 0
            ids = [r.id for r in records
                   if (r.payload.get("label") or "").lower() == label
                   and ts - SIGHTING_APART < (r.payload.get("ts") or 0) <= ts]
            self.shard.update(UpdateOperation.delete_points(ids))
            self.shard.flush()
            return len(ids)

    def last_sightings(self, label, limit=3, kinds=("seen",)):
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000,
                with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind", match=MatchAny(list(kinds))),
                ]),
            ))
        rows = [r for r in records
                if (r.payload.get("label") or "").lower() == label.lower()]
        rows.sort(key=lambda r: r.payload.get("ts") or 0, reverse=True)
        out = []
        for r in rows:
            if out and ((out[-1].payload.get("ts") or 0)
                        - (r.payload.get("ts") or 0)) < SIGHTING_APART:
                continue
            out.append(r)
            if limit is not None and len(out) >= limit:
                break
        return out

    def seen_since(self, since_ts, limit=4):
        from qdrant_edge import ScrollRequest
        with self._lock:
            records, _ = self.shard.scroll(ScrollRequest(
                limit=10000,
                with_payload=True,
                filter=Filter(must=[
                    FieldCondition(key="kind",
                                   match=MatchValue(value="seen")),
                    FieldCondition(key="ts",
                                   range=RangeFloat(gte=since_ts)),
                ]),
            ))
        records.sort(key=lambda r: r.payload.get("ts") or 0, reverse=True)
        out, seen = [], set()
        for r in records:
            key = (r.payload.get("label") or "").lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= limit:
                break
        return out
