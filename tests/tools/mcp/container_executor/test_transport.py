from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import unittest
from unittest import mock

from tokimon.tools.mcp.container_executor import transport


class TransportTests(unittest.TestCase):
    def test_request_has_total_deadline_even_when_peer_keeps_making_progress(self):
        clock = [0.0]
        class Peer:
            def __enter__(self): return self
            def __exit__(self, *args): self.closed = True
            def settimeout(self, timeout): pass
            def sendall(self, frame): self.frame = frame
            def recv(self, limit):
                clock[0] += 0.04
                return b" "
        peer = Peer()
        with mock.patch.object(transport, "connect", return_value=peer), mock.patch.object(transport.time, "monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(TimeoutError):
                transport.request("ping", timeout=0.1)
        self.assertTrue(peer.closed)
        self.assertEqual(json.loads(peer.frame)["method"], "ping")

    def test_request_returns_complete_response_and_rejects_incomplete_frame(self):
        for response, succeeds in ((b'{"id":1,"result":{}}\n', True), (b'{"id":1}', False)):
            peer, client = socket.socketpair()
            with peer, client:
                peer.sendall(response)
                peer.shutdown(socket.SHUT_WR)
                with mock.patch.object(transport, "connect", return_value=client):
                    if succeeds:
                        self.assertEqual(transport.request("ping")["result"], {})
                    else:
                        with self.assertRaisesRegex(RuntimeError, "bounded"):
                            transport.request("ping")

    def client(self):
        peer, client = socket.socketpair()
        process = subprocess.Popen([sys.executable, "-c",
            "import socket,sys; from tokimon.tools.mcp.container_executor import transport; "
            "transport.connect=lambda:socket.socket(fileno=int(sys.argv[1])); raise SystemExit(transport.relay())",
            str(client.fileno())], pass_fds=(client.fileno(),), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        client.close()
        self.addCleanup(peer.close)
        def cleanup():
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
        self.addCleanup(cleanup)
        return peer, process

    def test_remote_disconnect_exits_while_stdin_is_still_open(self):
        peer, process = self.client()
        request = b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n'
        process.stdin.write(request)
        process.stdin.flush()
        self.assertEqual(peer.recv(4096), request)
        peer.sendall(b'{"jsonrpc":"2.0","id":1,"result":{}}\n')
        self.assertEqual(json.loads(process.stdout.readline())["id"], 1)
        peer.close()
        self.assertEqual(process.wait(timeout=3), 1)

    def test_input_eof_preserves_pending_response(self):
        peer, process = self.client()
        def respond():
            request = peer.recv(4096)
            self.assertTrue(request.endswith(b"\n"))
            peer.sendall(b'{"id":8,"result":{}}\n')
            peer.shutdown(socket.SHUT_WR)
        thread = threading.Thread(target=respond)
        thread.start()
        output, errors = process.communicate(b'{"id":8,"method":"ping"}\n', timeout=3)
        thread.join(3)
        self.assertEqual(process.returncode, 0, errors)
        self.assertEqual(json.loads(output)["id"], 8)
