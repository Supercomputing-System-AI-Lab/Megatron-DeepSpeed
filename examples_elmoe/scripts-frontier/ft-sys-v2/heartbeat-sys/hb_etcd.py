"""hb_etcd.py (ft-sys-v2 heartbeat system, 2026-11-05): a small etcd v3 client over etcd's own HTTP/JSON gateway.

Standard library only, so the training environment needs no new package (design map 1.2: the fast beat goes through etcd).
The calls the heartbeat system uses: lease grant / keepalive / revoke, put, get, range by prefix, delete, create-if-absent with an
optional fence (one transaction), and watch by prefix (a stream read in a thread). Every request goes to the first endpoint that
answers; a dead member is skipped, and a follower forwards writes to the leader, so any live member will do.
The gateway encodes keys and values in base64 and 64-bit numbers as strings (etcd API reference, etcd.io/docs/v3.5)."""
import base64
import http.client
import json
import threading
import time
import urllib.parse


def _b(s):
    return base64.b64encode(s if isinstance(s, bytes) else s.encode()).decode()


def _u(s):
    return base64.b64decode(s).decode() if s else ''


def prefix_end(prefix):
    """The first key after every key that starts with prefix (etcd's range_end for a prefix)."""
    b = bytearray(prefix.encode())
    for i in range(len(b) - 1, -1, -1):
        if b[i] < 0xff:
            b[i] += 1
            return bytes(b[:i + 1])
    return b'\0'


class EtcdError(Exception):
    pass


class Etcd:
    def __init__(self, endpoints, timeout=3.0):
        """endpoints: 'host:port' or 'http://host:port' of the members' client URLs."""
        self.eps = []
        for e in endpoints:
            u = urllib.parse.urlparse(e if '://' in e else 'http://' + e)
            self.eps.append((u.hostname, u.port))
        self.timeout = timeout
        self._first = 0

    def _post(self, path, body):
        last = None
        for k in range(len(self.eps)):
            i = (self._first + k) % len(self.eps)
            host, port = self.eps[i]
            try:
                c = http.client.HTTPConnection(host, port, timeout=self.timeout)
                c.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
                r = c.getresponse()
                data = r.read()
                c.close()
            except (OSError, http.client.HTTPException) as e:
                last = f'{host}:{port}: {type(e).__name__} {e}'
                continue
            if r.status != 200:
                last = f'{host}:{port}{path}: HTTP {r.status} {data[:200]!r}'
                continue
            self._first = i
            return json.loads(data or b'{}')
        raise EtcdError(f'no etcd member answered {path} ({last})')

    # leases: keys tied to a lease are deleted when it is not renewed within its TTL
    def grant(self, ttl):
        return int(self._post('/v3/lease/grant', {'TTL': int(ttl)})['ID'])

    def keepalive(self, lease):
        """Renew a lease; returns the TTL it now has, 0 if the lease has already ended."""
        r = self._post('/v3/lease/keepalive', {'ID': lease})
        return int(r.get('result', r).get('TTL', 0) or 0)

    def revoke(self, lease):
        self._post('/v3/lease/revoke', {'ID': lease})

    # keys
    def put(self, key, value, lease=None):
        body = {'key': _b(key), 'value': _b(value)}
        if lease:
            body['lease'] = lease
        return int(self._post('/v3/kv/put', body)['header']['revision'])

    def get(self, key):
        """(value, mod_revision) of one key, or (None, the store's revision) if it does not exist."""
        r = self._post('/v3/kv/range', {'key': _b(key)})
        kvs = r.get('kvs')
        if kvs:
            return _u(kvs[0].get('value', '')), int(kvs[0]['mod_revision'])
        return None, int(r['header']['revision'])

    def range(self, prefix):
        """({key: value} of every key under prefix, the store's revision at the read)."""
        r = self._post('/v3/kv/range', {'key': _b(prefix), 'range_end': _b(prefix_end(prefix))})
        return {_u(kv['key']): _u(kv.get('value', '')) for kv in r.get('kvs', [])}, int(r['header']['revision'])

    def delete(self, key):
        self._post('/v3/kv/deleterange', {'key': _b(key)})

    def create_if_absent(self, key, value, lease=None, fence=None):
        """One transaction: write key only if it does not exist yet, and (with fence=(k, v)) only if key k still holds v.
        True if this call wrote it. Two callers at once: exactly one succeeds."""
        compare = [{'key': _b(key), 'target': 'CREATE', 'result': 'EQUAL', 'create_revision': '0'}]
        if fence:
            compare.append({'key': _b(fence[0]), 'target': 'VALUE', 'result': 'EQUAL', 'value': _b(fence[1])})
        put = {'key': _b(key), 'value': _b(value)}
        if lease:
            put['lease'] = lease
        r = self._post('/v3/kv/txn', {'compare': compare, 'success': [{'request_put': put}]})
        return bool(r.get('succeeded', False))

    # watch: a stream of every change under a prefix
    def watch(self, prefix, on_event, start_revision=None, stop=None, name='etcd-watch'):
        """Call on_event(kind, key, value, revision) for every change under prefix, kind 'PUT' or 'DELETE', in the store's order.
        If the stream breaks (a member died), reconnect to the next member and resume after the last revision seen, so no event is
        lost or repeated. Runs in a daemon thread, which is returned; set stop to end it (it ends at the next event or timeout)."""
        def run():
            rev = start_revision
            while rev is None and not (stop and stop.is_set()):
                try:      # the next revision after now, so a reconnection resumes exactly there (an idle stream times out)
                    rev = int(self._post('/v3/kv/range', {'key': _b(prefix), 'range_end': _b(prefix_end(prefix)),
                                                          'count_only': True})['header']['revision']) + 1
                except EtcdError:
                    time.sleep(0.5)
            k = 0
            while not (stop and stop.is_set()):
                host, port = self.eps[k % len(self.eps)]
                k += 1
                try:
                    c = http.client.HTTPConnection(host, port, timeout=30)
                    req = {'key': _b(prefix), 'range_end': _b(prefix_end(prefix))}
                    if rev:
                        req['start_revision'] = rev
                    c.request('POST', '/v3/watch', json.dumps({'create_request': req}), {'Content-Type': 'application/json'})
                    r = c.getresponse()
                    if r.status != 200:
                        raise EtcdError(f'watch HTTP {r.status}')
                    while not (stop and stop.is_set()):
                        line = r.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        res = json.loads(line).get('result', {})
                        hdr = int(res.get('header', {}).get('revision', 0) or 0)
                        if res.get('created') and hdr:
                            rev = max(rev or 0, hdr + 1) if not res.get('events') else rev
                        if res.get('compact_revision'):
                            rev = int(res['compact_revision'])      # history was compacted: resume from what is left
                            break
                        for ev in res.get('events', []):
                            kv = ev.get('kv', {})
                            rv = int(kv.get('mod_revision', 0))
                            on_event(ev.get('type', 'PUT'), _u(kv.get('key', '')), _u(kv.get('value', '')), rv)
                            rev = rv + 1
                    c.close()
                except (OSError, http.client.HTTPException, EtcdError, ValueError):
                    time.sleep(0.5)
        t = threading.Thread(target=run, name=name, daemon=True)
        t.start()
        return t
