"""
services/rate_limiter.py
========================
Lightweight rate-limiting and quota management for free-tier deployments.

Protects Gemini API quotas and server resources:
  1. Per-minute rate limit: Max 5 queries / minute per client IP.
  2. Per-day quota: Max 25 queries / day per client IP.
  3. Session cap: Max 10 turns per conversation session.

Uses Redis when available, with an automatic in-memory fallback.
"""

import time
from typing import Tuple, Dict, List
from collections import defaultdict


class QuotaLimiter:
    """
    Manages client quotas and request rate limits.
    """

    def __init__(
        self,
        redis_client=None,
        max_per_minute: int = 5,
        max_per_day: int = 30,
        max_turns_per_session: int = 10,
    ):
        self.redis = redis_client
        self.max_per_minute = max_per_minute
        self.max_per_day = max_per_day
        self.max_turns_per_session = max_turns_per_session

        # In-memory fallback if Redis is unavailable
        self._minute_buckets: Dict[str, List[float]] = defaultdict(list)
        self._day_counts: Dict[str, Tuple[int, str]] = {}  # ip -> (count, yyyy-mm-dd)

    def check_and_increment(self, client_ip: str, session_turn_count: int = 0) -> Tuple[bool, str]:
        """
        Validates whether the client request is permitted under free-tier quotas.
        Returns: (is_allowed, rejection_reason)
        """
        # 1. Session turn limit check
        if session_turn_count >= self.max_turns_per_session:
            return False, f"Session limit reached ({self.max_turns_per_session} turns). Please start a New Session."

        # 2. Redis-backed rate limiting (preferred in production)
        if self.redis:
            try:
                pipe = self.redis.pipeline()
                minute_key = f"rate:min:{client_ip}"
                day_key = f"rate:day:{client_ip}"

                min_count = self.redis.incr(minute_key)
                if min_count == 1:
                    self.redis.expire(minute_key, 60)

                day_count = self.redis.incr(day_key)
                if day_count == 1:
                    self.redis.expire(day_key, 86400)

                if min_count > self.max_per_minute:
                    ttl = self.redis.ttl(minute_key)
                    return False, f"Rate limit exceeded (max {self.max_per_minute}/min). Please retry in {ttl}s."

                if day_count > self.max_per_day:
                    return False, f"Daily free-tier quota exceeded (max {self.max_per_day}/day). Please return tomorrow."

                return True, "OK"
            except Exception as e:
                print(f"[RateLimiter] Redis check failed ({e}), falling back to in-memory.")

        # 3. In-memory sliding-window fallback
        now = time.time()
        today = time.strftime("%Y-%m-%d")

        # Clean old minute timestamps (>60s)
        recent_timestamps = [t for t in self._minute_buckets[client_ip] if now - t < 60]
        if len(recent_timestamps) >= self.max_per_minute:
            wait_time = int(60 - (now - recent_timestamps[0]))
            return False, f"Rate limit exceeded (max {self.max_per_minute}/min). Please retry in {wait_time}s."

        # Check daily quota
        cnt, date_str = self._day_counts.get(client_ip, (0, today))
        if date_str != today:
            cnt = 0
            date_str = today

        if cnt >= self.max_per_day:
            return False, f"Daily free-tier quota exceeded (max {self.max_per_day}/day). Reset at midnight."

        # Record this request
        recent_timestamps.append(now)
        self._minute_buckets[client_ip] = recent_timestamps
        self._day_counts[client_ip] = (cnt + 1, today)

        return True, "OK"
