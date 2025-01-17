"""Download multiple files concurrently by streaming their content to disk."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from ssl import SSLContext
from typing import TYPE_CHECKING, Literal

from aiohttp import ClientSession, TCPConnector
from aiohttp.client_exceptions import ClientResponseError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["DownloadError", "stream_write"]
CHUNK_SIZE = 1024 * 1024  # Default chunk size of 1 MB
MAX_HOSTS = 5  # Default maximum number of hosts

if sys.platform == "win32":  # pragma: no cover
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class DownloadError(Exception):
    """Exception raised when the requested data is not available on the server.

    Parameters
    ----------
    err : str
        Service error message.
    """

    def __init__(self, err: str, url: str | None = None) -> None:
        self.message = "Service returned the following error message:\n"
        if url is None:
            self.message += err
        else:
            self.message += f"URL: {url}\nERROR: {err}\n"
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


async def _stream_file(session: ClientSession, url: str, filepath: Path) -> None:
    """Stream the response to a file."""
    async with session.get(url) as response:
        try:
            response.raise_for_status()
        except (ClientResponseError, ValueError) as ex:
            raise DownloadError(await response.text(), str(response.url)) from ex

        with filepath.open("wb") as file:
            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                file.write(chunk)


async def _stream_session(urls: Sequence[str], files: Sequence[Path]) -> None:
    """Create an async session to download files."""
    async with ClientSession(connector=TCPConnector(limit_per_host=MAX_HOSTS)) as s:
        tasks = [_stream_file(s, url, filepath) for url, filepath in zip(urls, files)]
        await asyncio.gather(*tasks)


def _get_event_loop() -> tuple[asyncio.AbstractEventLoop, bool]:
    """Get or create an event loop."""
    with contextlib.suppress(RuntimeError):
        return asyncio.get_running_loop(), False

    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    return new_loop, True


def stream_write(urls: Sequence[str], file_paths: Sequence[Path]) -> None:
    """Download multiple files concurrently by streaming their content to disk."""
    file_paths = [Path(filepath) for filepath in file_paths]
    parent_dirs = {filepath.parent for filepath in file_paths}
    for parent_dir in parent_dirs:
        parent_dir.mkdir(parents=True, exist_ok=True)

    loop, is_new_loop = _get_event_loop()

    try:
        loop.run_until_complete(_stream_session(urls, file_paths))
    finally:
        if is_new_loop:
            loop.close()
