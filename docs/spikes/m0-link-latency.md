# Mac ↔ brain link latency (M0)

_Measured 2026-10-29: `jason run --role mac --count 1000 --every 0.1` from the Mac to
the containerized brain on the Windows laptop, over Tailscale, schema 2._

## Result

The link is fast and steady, with one burst of delay: **median 12 ms, p99 45 ms**,
and all 10 pings over 50 ms came in a single 1-second burst (pings 529–539,
up to 525 ms). Outside that burst, p99 is about 25 ms. No events were lost
(no `link_gap`), and the connection never dropped.

| | ms |
|---|---|
| min | 7.2 |
| p10 | 9.9 |
| p50 | 12.0 |
| p90 | 15.4 |
| p95 | 17.3 |
| p99 | 44.9 |
| max | 524.6 |

| Over | Pings (of 1,000) |
|---|---|
| 20 ms | 25 |
| 50 ms | 10 (all in pings 529–539) |
| 100 ms | 8 |
| 200 ms | 5 |

Every block of 100 pings had a median between 11.6 and 12.4 ms.

- **What is measured:** `ping_rtt` is bus to bus: both buses, both raw-log writes,
  JSON, Docker's and WSL2's network hops, and Tailscale. The clock-sync probes
  (`network_rtt`, link only, no bus or log) over the same period: median 7.5 ms,
  range 5.5–13.7 ms.
- **Clock offset drift:** +1.5 ms over the 100 s run (estimates −96,361,764.6 →
  −96,361,763.1 ms, error bound ±2.7 ms at the end). The offset is re-estimated
  every 5 s, so this drift is tracked, not accumulated.
- **The burst:** a single event lasting about a second, most likely Wi-Fi or
  Tailscale re-routing. One run can't say which; repeat runs, and a run with
  the Mac on Ethernet, would.

## What it means for M1

- The median uses about 1.2% of the 1-second budget (design §10). The design's
  "LAN ~5 ms" holds for the network alone (5.5–7.5 ms); bus to bus it is about
  12 ms.
- The tail matters more than the median for "feels real-time": a 0.5-second
  stall during a reply would be noticeable. M1's per-stage latency logging should
  record network time per turn, so stalls can be matched to their cause.
- Events at about 10 per second (for example `user_partial`) are far below
  what this link carries, but a 1-second stall queues them: the outbox (1,000
  events) absorbs that easily, and schema 2 would log any loss.
