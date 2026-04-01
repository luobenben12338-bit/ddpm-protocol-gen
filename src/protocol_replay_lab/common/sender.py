from __future__ import annotations

import socket
import time
from contextlib import closing
from typing import Iterable, List, Optional

from .models import RequestResponse, SendSummary


class TcpSender:
    def __init__(self, host: str, port: int, timeout: float = 2.0, recv_size: int = 4096):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.recv_size = recv_size

    def _create_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        return sock

    def send_messages(
        self,
        messages: Iterable[bytes],
        *,
        reconnect_on_error: bool = True,
        delay: float = 0.0,
    ) -> SendSummary:
        message_list: List[bytes] = list(messages)
        summary = SendSummary(host=self.host, port=self.port, total_count=len(message_list))
        start_time = time.time()

        sock: Optional[socket.socket] = None
        try:
            sock = self._create_socket()
            for index, message in enumerate(message_list):
                try:
                    sock.sendall(message)
                    response = sock.recv(self.recv_size)
                    summary.sent_count += 1
                    summary.exchanges.append(RequestResponse(request=message, response=response))
                    if delay > 0:
                        time.sleep(delay)
                except (socket.timeout, socket.error) as exc:
                    summary.failed_indices.append(index)
                    summary.exchanges.append(RequestResponse(request=message, error=str(exc)))
                    if sock is not None:
                        sock.close()
                        sock = None
                    if reconnect_on_error and index != len(message_list) - 1:
                        sock = self._create_socket()
                    else:
                        break
        finally:
            summary.elapsed_seconds = time.time() - start_time
            if sock is not None:
                with closing(sock):
                    pass

        return summary
