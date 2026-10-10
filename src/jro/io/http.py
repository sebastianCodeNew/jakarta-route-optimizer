"""Resilient HTTP client with retries, exponential backoff, and atomic downloads."""

from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from jro.exceptions import DataSourceError
from jro.io.manifest import compute_sha256
from jro.logging_setup import logger

DEFAULT_USER_AGENT = "JakartaRouteOptimizer/0.1.0 (+https://github.com/jakarta-route-optimizer)"
DEFAULT_TIMEOUT_SECONDS = 30.0


def create_session(
    retries: int = 4,
    backoff_factor: float = 1.5,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Creates a requests.Session with exponential backoff on retryable HTTP errors."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


def download_file(
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
    session: requests.Session | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    chunk_size: int = 65536,
) -> Path:
    """Streams download from URL to temporary file and renames atomically.

    Args:
        url: Direct HTTP/HTTPS resource URL.
        destination: Local file path where output will be saved.
        expected_sha256: Optional SHA-256 hash to verify.
        session: Optional custom requests.Session.
        timeout: Socket connection/read timeout in seconds.
        chunk_size: Stream buffer size in bytes.

    Returns:
        Path to the successfully downloaded file.

    Raises:
        DataSourceError: On network failure, non-200 status, or hash mismatch.
    """
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    sess = session or create_session()
    logger.info("Downloading from %s to %s", url, destination)

    try:
        with sess.get(url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                raise DataSourceError(
                    f"HTTP GET failed with status {response.status_code} for URL: {url}"
                )

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        if expected_sha256:
            actual_sha = compute_sha256(temp_path)
            if actual_sha.lower() != expected_sha256.lower():
                temp_path.unlink(missing_ok=True)
                raise DataSourceError(
                    f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual_sha}"
                )

        temp_path.replace(destination)
        logger.info(
            "Successfully downloaded and verified %s (%d bytes)",
            destination.name,
            destination.stat().st_size,
        )
        return destination

    except requests.RequestException as e:
        temp_path.unlink(missing_ok=True)
        raise DataSourceError(f"Network error while downloading {url}: {e}") from e
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        if isinstance(e, DataSourceError):
            raise
        raise DataSourceError(f"Unexpected error while processing download for {url}: {e}") from e
