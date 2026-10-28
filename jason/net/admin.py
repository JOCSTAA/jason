"""Send one admin command to a running brain (used by `jason new-day`)."""

import asyncio
from http import HTTPStatus

from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from jason.net.link import ADMIN_PATH, LinkAuthError, auth_header
from jason.net.protocol import Command, CommandResult, decode, encode


def admin_url(link_url: str) -> str:
    """ws://jason-brain:8765 → ws://jason-brain:8765/admin"""
    return link_url.rstrip("/") + ADMIN_PATH


async def send_command(url: str, token: str, name: str, timeout_s: float = 10.0) -> CommandResult:
    """Raises OSError if the brain isn't reachable, LinkAuthError if the token is wrong."""
    try:
        async with connect(
            url,
            additional_headers={"Authorization": auth_header(token)},
            open_timeout=timeout_s,
            compression=None,
        ) as ws:
            await ws.send(encode(Command(name=name)))
            reply = decode(await asyncio.wait_for(ws.recv(), timeout_s))
    except InvalidStatus as e:
        if e.response.status_code == HTTPStatus.UNAUTHORIZED:
            raise LinkAuthError("the brain rejected the link token") from None
        raise
    if not isinstance(reply, CommandResult):
        raise RuntimeError(f"unexpected reply from the brain: {reply.kind}")
    return reply
