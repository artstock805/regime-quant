"""
엔진이 도는 동안 PC 가 절전에 들어가지 않게 한다.

**절전 중에는 아무것도 돌지 않는다.** Windows 가 CPU 를 멈추므로 엔진도 MT5 도
선다. 실제로 있었던 일이다: 엔진을 켜 두고 자리를 비웠더니 PC 가 절전에 들어가,
일주일 동안 엔진이 판단한 시간이 13시간쯤뿐이었다. 손절은 브로커에 걸려 있어
지켜졌지만 진입도 신호 청산도 역행 대응도 멈췄다.

영상 재생 프로그램이 재생 중에 절전을 막는 것과 같은 방법을 쓴다
(`SetThreadExecutionState`).

    · **모니터는 평소처럼 꺼진다.** 막는 것은 PC 의 절전뿐이다.
    · **전원 설정을 바꾸지 않는다.** 요청은 부른 스레드에 묶여 있어서, 그 스레드가
      끝나면(엔진 정지·앱 종료·비정상 종료) 저절로 풀린다. 풀기를 잊어도 PC 가
      영영 안 자는 일은 없다.
    · **막지 못하는 것이 있다.** 노트북 덮개를 닫거나 전원 메뉴에서 직접 절전을
      누르면 그대로 잔다. 그때는 깨어난 뒤 mt5.initialise() 가 다시 붙는다.

Windows 가 아니면 아무것도 하지 않는다.
"""

from __future__ import annotations

import ctypes
import sys

_ES_CONTINUOUS = 0x80000000        # 다음에 바꿀 때까지 유지
_ES_SYSTEM_REQUIRED = 0x00000001   # 시스템이 필요하다 — 유휴 절전 금지


def _set(flags: int) -> bool:
    """이 스레드의 실행 상태를 바꾼다. 됐으면 True."""
    if sys.platform != "win32":
        return False
    try:
        call = ctypes.windll.kernel32.SetThreadExecutionState
        call.argtypes = [ctypes.c_uint]
        call.restype = ctypes.c_uint
        return bool(call(flags))        # 0 이면 실패
    except Exception:
        return False


def hold() -> bool:
    """**부른 스레드가 살아 있는 동안** 유휴 절전을 막는다."""
    return _set(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)


def release() -> bool:
    """막던 것을 푼다. 같은 스레드에서 불러야 한다."""
    return _set(_ES_CONTINUOUS)
