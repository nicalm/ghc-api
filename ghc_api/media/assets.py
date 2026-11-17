"""Owner-scoped immutable image snapshots and durable preview byte budgets."""

import hashlib
import os
import re
import sqlite3
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from ..utils import get_config_dir
from .previews import ImageToolError, describe_image


_LOCK = threading.RLock()
_ASSET = re.compile(r'^[a-f0-9]{32}$')
_CONTEXT = re.compile(r'^[A-Za-z0-9_.:-]{1,128}$')


class AssetStore:
    def __init__(self, config):
        self.config = config
        self.root = Path(config.data_dir or str(Path(get_config_dir()) / 'image-tools')).resolve()

    @contextmanager
    def connection(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            db = sqlite3.connect(str(self.root / 'assets.sqlite'), timeout=10)
            db.row_factory = sqlite3.Row
            db.execute('CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, owner TEXT NOT NULL, sha TEXT NOT NULL, name TEXT NOT NULL, bytes INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, alpha INTEGER NOT NULL, transparent INTEGER NOT NULL, created REAL NOT NULL, UNIQUE(owner,sha))')
            db.execute('CREATE TABLE IF NOT EXISTS budgets (owner TEXT NOT NULL, context TEXT NOT NULL, bytes INTEGER NOT NULL, updated REAL NOT NULL, PRIMARY KEY(owner,context))')
            try:
                yield db
            finally:
                db.close()

    def _path(self, owner, asset_id):
        if not _ASSET.fullmatch(asset_id):
            raise ImageToolError('asset_not_found', 'Image asset not found for this caller.')
        folder = hashlib.sha256(owner.encode()).hexdigest()
        return self.root / folder / (asset_id + '.bin')

    def _cleanup(self, db):
        cutoff = time.time() - self.config.asset_ttl_seconds
        for row in db.execute('SELECT owner,id FROM assets WHERE created < ?', (cutoff,)).fetchall():
            self._path(row['owner'], row['id']).unlink(missing_ok=True)
        db.execute('DELETE FROM assets WHERE created < ?', (cutoff,))
        db.execute('DELETE FROM budgets WHERE updated < ?', (cutoff,))

    def register(self, owner, source):
        if not isinstance(source, str) or len(source) > 4096:
            raise ImageToolError('invalid_path', 'Supply an absolute path under an allowed image directory.')
        try:
            requested = Path(source)
            if not requested.is_absolute():
                raise ValueError('relative path')
            path = requested.resolve(strict=True)
            roots = [Path(p) for p in self.config.allowed_roots]
            if (not any(path == root or root in path.parents for root in roots)
                    or path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp')):
                raise ValueError('outside roots or unsupported suffix')
            stat_before = path.stat()
            if not path.is_file() or stat_before.st_size > self.config.max_source_bytes:
                raise ImageToolError('source_too_large', 'Source image exceeds the configured file limit.')
            with path.open('rb') as stream:
                data = stream.read(self.config.max_source_bytes + 1)
                opened = os.fstat(stream.fileno())
            if (len(data) > self.config.max_source_bytes or requested.resolve(strict=True) != path
                    or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                    != (stat_before.st_dev, stat_before.st_ino, stat_before.st_size, stat_before.st_mtime_ns)):
                raise ImageToolError('source_changed', 'Source image changed while being read; register it again.')
        except ImageToolError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            raise ImageToolError('path_not_allowed', 'Cannot read this path under the configured image directories.') from exc
        metadata = describe_image(data, self.config)
        digest = hashlib.sha256(data).hexdigest()
        asset_id = hashlib.sha256((owner + ':' + digest).encode()).hexdigest()[:32]
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._cleanup(db)
            existing = db.execute('SELECT * FROM assets WHERE owner=? AND sha=?', (owner, digest)).fetchone()
            destination = self._path(owner, asset_id)
            intact = False
            if existing:
                try:
                    with destination.open('rb') as stream:
                        snapshot = stream.read(self.config.max_source_bytes + 1)
                    intact = len(snapshot) == len(data) and hashlib.sha256(snapshot).hexdigest() == digest
                except OSError:
                    pass
            if not existing:
                total, count = db.execute('SELECT COALESCE(SUM(bytes),0),COUNT(*) FROM assets').fetchone()
                if total + len(data) > self.config.store_max_bytes or count >= 4096:
                    raise ImageToolError('asset_store_full', 'Image snapshot cache is full; wait for its configured expiry.')
            if not intact:
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd, temp_path = tempfile.mkstemp(prefix='.image-', dir=str(destination.parent))
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(data)
                    os.replace(temp_path, destination)
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            try:
                if not existing:
                    db.execute('INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?)',
                               (asset_id, owner, digest, path.name[:255], len(data), metadata['width'], metadata['height'],
                                int(metadata['has_alpha']), int(metadata['has_transparency']), time.time()))
                db.commit()
            except Exception:
                if not existing:
                    destination.unlink(missing_ok=True)
                raise
        return self.describe(owner, asset_id)

    def _record(self, db, owner, asset_id):
        if not isinstance(asset_id, str) or not _ASSET.fullmatch(asset_id):
            raise ImageToolError('asset_not_found', 'Image asset not found for this caller.')
        row = db.execute('SELECT * FROM assets WHERE owner=? AND id=? AND created>=?',
                         (owner, asset_id, time.time() - self.config.asset_ttl_seconds)).fetchone()
        if row is None:
            raise ImageToolError('asset_not_found', 'Image asset missing or expired; register the original again.')
        return row

    def describe(self, owner, asset_id):
        with self.connection() as db:
            row = self._record(db, owner, asset_id)
            return {'asset_id': row['id'], 'source_name': row['name'], 'original_sha256': row['sha'],
                    'source_bytes': row['bytes'], 'width': row['width'], 'height': row['height'],
                    'has_alpha': bool(row['alpha']), 'has_transparency': bool(row['transparent']),
                    'expires_at': row['created'] + self.config.asset_ttl_seconds}

    def read(self, owner, asset_id):
        with self.connection() as db:
            row = self._record(db, owner, asset_id)
            try:
                with self._path(owner, asset_id).open('rb') as stream:
                    data = stream.read(self.config.max_source_bytes + 1)
            except OSError as exc:
                raise ImageToolError('asset_unavailable', 'Image snapshot unavailable; register the original again.') from exc
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha']:
                raise ImageToolError('asset_changed', 'Image snapshot hash does not match; register the source again.')
            return data

    def consume(self, owner, context_id, byte_count):
        if not isinstance(context_id, str) or not _CONTEXT.fullmatch(context_id):
            raise ImageToolError('invalid_context', 'context_id must be a stable task ID using letters, digits, dot, colon, dash or underscore.')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._cleanup(db)
            row = db.execute('SELECT bytes FROM budgets WHERE owner=? AND context=?', (owner, context_id)).fetchone()
            used = row['bytes'] if row else 0
            if used + byte_count > self.config.context_budget_bytes:
                raise ImageToolError('context_preview_budget_exceeded', 'Preview budget for this task is exhausted. Save observations as text and continue from a compact handoff; do not reload the whole image set.')
            if not row and db.execute('SELECT COUNT(*) FROM budgets').fetchone()[0] >= 4096:
                raise ImageToolError('context_store_full', 'Too many preview contexts; wait for expiry.')
            db.execute('INSERT INTO budgets VALUES (?,?,?,?) ON CONFLICT(owner,context) DO UPDATE SET bytes=excluded.bytes,updated=excluded.updated',
                       (owner, context_id, used + byte_count, time.time()))
            db.commit()
            return {'used_bytes': used + byte_count, 'limit_bytes': self.config.context_budget_bytes,
                    'context_id': context_id, 'includes_repeated_reads': True}
