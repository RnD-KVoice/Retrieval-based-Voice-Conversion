import logging
import subprocess
import time
import re

logger = logging.getLogger(__name__)

_active_tunnel = {"process": None, "url": None, "method": None}


def start_cloudflared(port=8000):
    try:
        subprocess.run(["cloudflared", "--version"], capture_output=True, check=True, timeout=5)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "cloudflared not installed. Install with:\n"
            "  !wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64\n"
            "  !chmod +x cloudflared-linux-amd64\n"
            "  !mv cloudflared-linux-amd64 /usr/local/bin/cloudflared"
        )

    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    logger.info("Waiting for cloudflared URL...")
    url_pattern = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')

    start_time = time.time()
    timeout = 30

    while time.time() - start_time < timeout:
        if process.stdout:
            line = process.stdout.readline()
            if line:
                logger.debug(f"cloudflared: {line.strip()}")
                match = url_pattern.search(line)
                if match:
                    url = match.group(0)
                    _active_tunnel["process"] = process
                    _active_tunnel["url"] = url
                    _active_tunnel["method"] = "cloudflared"
                    logger.info(f"Tunnel established: {url}")
                    return url

        if process.stderr:
            line = process.stderr.readline()
            if line:
                logger.debug(f"cloudflared stderr: {line.strip()}")
                match = url_pattern.search(line)
                if match:
                    url = match.group(0)
                    _active_tunnel["process"] = process
                    _active_tunnel["url"] = url
                    _active_tunnel["method"] = "cloudflared"
                    logger.info(f"Tunnel established: {url}")
                    return url

        if process.poll() is not None:
            raise RuntimeError("cloudflared terminated unexpectedly")

        time.sleep(0.5)

    stop_tunnel()
    raise RuntimeError("Timeout waiting for cloudflared URL")


def start_ngrok(port=8000, token=None):
    try:
        from pyngrok import ngrok
    except ImportError:
        raise RuntimeError(
            "pyngrok not installed. Install with:\n  !pip install pyngrok"
        )

    if token:
        ngrok.set_auth_token(token)
        logger.info("Ngrok auth token configured")

    logger.info(f"Starting ngrok tunnel on port {port}...")
    tunnel = ngrok.connect(port, bind_tls=True)

    url = tunnel.public_url
    _active_tunnel["url"] = url
    _active_tunnel["method"] = "ngrok"

    logger.info(f"Tunnel established: {url}")
    return url


def start_tunnel(port=8000, method="cloudflared", ngrok_token=None):
    logger.info(f"Starting {method} tunnel on port {port}...")

    if method == "cloudflared":
        return start_cloudflared(port)
    elif method == "ngrok":
        return start_ngrok(port, ngrok_token)
    else:
        raise ValueError(f"Unknown tunnel method: {method}")


def stop_tunnel():
    if _active_tunnel["method"] == "cloudflared" and _active_tunnel["process"]:
        logger.info("Stopping cloudflared...")
        _active_tunnel["process"].terminate()

        try:
            _active_tunnel["process"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Killing cloudflared...")
            _active_tunnel["process"].kill()

        _active_tunnel["process"] = None

    elif _active_tunnel["method"] == "ngrok":
        logger.info("Stopping ngrok...")
        try:
            from pyngrok import ngrok
            ngrok.disconnect(_active_tunnel["url"])
        except Exception as e:
            logger.warning(f"Error disconnecting ngrok: {e}")

    _active_tunnel["url"] = None
    _active_tunnel["method"] = None
    logger.info("Tunnel stopped")


def get_url():
    return _active_tunnel["url"]


def is_active():
    if _active_tunnel["method"] == "cloudflared" and _active_tunnel["process"]:
        return _active_tunnel["process"].poll() is None
    elif _active_tunnel["method"] == "ngrok":
        return _active_tunnel["url"] is not None
    return False
