"""Detection, recognition, teaching, and recall glue for the flat project."""
import time
from datetime import date
from pathlib import Path

import cv2

from camera_detector import Detector
from embeddings import (
    embed_image_crop_single,
    embed_query_textstring,
    embed_text_textstring,
)
from qdrant_memory import Memory, RECOGNIZE_THRESHOLD
from voice_labels import norm_label, parse_label


def day_start_ts():
    return time.mktime(date.today().timetuple())


FOCUS_MARGIN = 1.6
SCENE_PAD = 0.3
SCENE_PX = 480


class Robot:
    """High-level assistant state built from detector, encoders, and memory."""

    def __init__(self, data_dir="edge-data", weights="yoloe-11l-seg-pf.pt",
                 threshold=RECOGNIZE_THRESHOLD, conf=None, max_area=None,
                 where=None):
        self.memory = Memory(data_dir, threshold=threshold, where=where)
        opts = {k: v for k, v in
                (("conf", conf), ("max_area", max_area)) if v is not None}
        self.detector = Detector(weights, **opts)
        self.thumbs = Path(data_dir) / "thumbs"
        self.thumbs.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.attention = None
        self.teachable = None
        self._incumbent = {}

    def log(self, msg):
        self.events.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        del self.events[:-8]

    def _thumb(self, crop, tag, quality=None):
        path = self.thumbs / f"{time.time_ns()}_{tag}.jpg"
        opts = [] if quality is None else [cv2.IMWRITE_JPEG_QUALITY, quality]
        cv2.imwrite(str(path), crop, opts)
        return str(path)

    def _scene(self, frame, box):
        if frame is None or not frame.size:
            return None
        h, w = frame.shape[:2]
        img = frame
        if box is not None:
            x1, y1, x2, y2 = box
            px, py = (x2 - x1) * SCENE_PAD, (y2 - y1) * SCENE_PAD
            x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
            region = frame[y1:y2, x1:x2]
            if region.size:
                img = region
        h, w = img.shape[:2]
        scale = SCENE_PX / max(h, w)
        if scale < 1:
            img = cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return self._thumb(img, "scene", quality=80)

    def process_frame(self, frame, now=None):
        now = now or time.time()
        tracks = self.detector.process(frame, now)
        live = []
        for t in tracks:
            if t.due_for_query(now):
                t.vec = embed_image_crop_single(t.crop)
                hit, score, guess = self.memory.recognize(t.vec)
                t.score = score
                t.label = hit.payload["label"] if hit else None
                t.guess = guess
                t.note = hit.payload["transcript"] if hit else None
                t.thumb = hit.payload.get("thumb") if hit else None
                t.last_query = now
                t.crop_q = 0.0
                if hit is None:
                    ignored = self.memory.match_ignored(t.vec)
                    if ignored is not None:
                        self.detector.ignore(t.tid, pid=ignored.id)
                        continue
            live.append(t)

        knowns = self._one_per_label([t for t in live if t.label])
        primary = self.pick_focus(
            [t for t in live if not t.label and t.last_query], pool="unknown")
        display = knowns + ([primary] if primary else [])
        self.teachable = primary
        self.attention = self.pick_focus(display, pool="view")

        for t in display:
            if t.label and not t.sighted and t.vec is not None:
                self.memory.remember_sighting(
                    t.vec,
                    t.label,
                    ts=now,
                    thumb=self._scene(frame, t.box),
                )
                t.sighted = True
                self.log(f"seen: {t.label} ({t.score:.2f})")
        return display

    def _one_per_label(self, tracks):
        attending = self._incumbent.get("view")
        best = {}
        for t in tracks:
            key = t.label.lower()
            cur = best.get(key)
            if cur is None or t is attending:
                best[key] = t
            elif cur is not attending and (t.score, t.salience) > (
                    cur.score, cur.salience):
                best[key] = t
        return list(best.values())

    def pick_focus(self, tracks, pool):
        if not tracks:
            return None
        best = max(tracks, key=lambda t: t.salience)
        held = self._incumbent.get(pool)
        if held in tracks and best.salience < held.salience * FOCUS_MARGIN:
            best = held
        self._incumbent[pool] = best
        return best

    def teach(self, crop, transcript, frame=None, box=None):
        label = parse_label(transcript)
        pid = self.memory.teach(
            image_vec=embed_image_crop_single(crop),
            text_vec=embed_text_textstring(transcript),
            label=label,
            transcript=transcript,
            thumb=self._thumb(crop, "taught"),
            scene=self._scene(frame, box),
        )
        self.log(f'taught "{label[:18]}" -> image + text')
        return {"id": pid, "label": label, "transcript": transcript}

    def confirm(self, label, frame=None):
        t = self.attention
        if (t is None or t.label or not t.guess or t.crop is None
                or t.guess.lower() != label.lower()):
            return None
        res = self.teach(t.crop, t.guess, frame=frame, box=t.box)
        t.label = res["label"]
        t.note = res["transcript"]
        t.guess = None
        t.requery_now()
        return res

    def forget(self, label):
        n = self.memory.forget(label)
        for t in self.detector.tracks.values():
            if t.label and t.label.lower() == label.lower():
                t.label = t.note = t.thumb = None
                t.requery_now()
            if t.guess and t.guess.lower() == label.lower():
                t.guess = None
                t.requery_now()
        self.log(f'forgot "{label[:18]}" -> {n} point(s)')
        return n

    def rename(self, label, to, text_vec=None):
        to = norm_label(to)
        n = self.memory.rename(label, to, text_vec=text_vec)
        for t in self.detector.tracks.values():
            if t.label and t.label.lower() == label.lower():
                t.label = to
        self.log(f'renamed "{label[:14]}" -> "{to[:14]}" ({n} point(s))')
        return n

    def set_where(self, place):
        where = self.memory.set_where(place)
        self.log(f'here: "{where}"' if where else "location cleared")
        return where

    def forget_view(self, pid, label):
        ids = self.memory.taught_ids(label)
        if pid in ids:
            if len(ids) <= 1:
                return self.forget(label), True
            self.memory.forget_point(pid)
            self.log(f'dropped one view of "{label[:18]}"')
            return 1, False
        if pid in self.memory.seen_ids(label):
            n = self.memory.forget_sighting_burst(pid)
            self.log(f'dropped {n} sighting photo(s) of "{label[:18]}"')
            return n, False
        return 0, False

    def ignore(self, track, frame=None):
        vec = track.vec if track.vec is not None else embed_image_crop_single(
            track.crop)
        pid = self.memory.ignore(vec, thumb=self._scene(frame, track.box))
        self.detector.ignore(track.tid, pid=pid)
        if self.attention is track:
            self.attention = None
        if self.teachable is track:
            self.teachable = None
        self.log("ignored one unknown")

    def unignore(self, pid):
        ok = self.memory.unignore(pid)
        if ok:
            self.detector.unblock(pid)
        return ok

    def ask(self, question, since_ts=None):
        if "what did you see" in question.lower():
            since = day_start_ts() if since_ts is None else since_ts
            return {
                "inventory": True,
                "label": None,
                "note": None,
                "score": 0.0,
                "sightings": self.memory.seen_since(since),
            }
        qv = embed_query_textstring(question)
        named = self.memory.names_in(question)
        hit = self.memory.best_taught(qv, labels=named)
        if hit is None and named:
            hit = self.memory.best_taught(qv)
        if hit is None:
            return {
                "inventory": False,
                "label": None,
                "note": None,
                "score": 0.0,
                "sightings": [],
            }
        label = hit.payload.get("label")
        return {
            "inventory": False,
            "label": label,
            "note": hit.payload.get("transcript"),
            "score": hit.score,
            "sightings": self.memory.last_sightings(
                label, kinds=("seen", "taught")),
        }

    def reboot(self):
        start = time.perf_counter()
        self.memory.reopen()
        ms = (time.perf_counter() - start) * 1000
        self.detector.reset()
        self.attention = self.teachable = None
        n = self.memory.count()
        self.log(f"shard reopened in {ms:.0f} ms: {n} memories")
        return n, ms

    def close(self):
        self.memory.close()
