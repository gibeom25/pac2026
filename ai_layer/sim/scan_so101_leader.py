#!/usr/bin/env python
"""SO-101 leader 서보 버스 스캔 — 독립 진단 스크립트.

사용법 (pac2026 conda 환경에서):
  conda activate pac2026
  python ai_layer/sim/scan_so101_leader.py

케이블을 만지거나 보드를 확인한 뒤 그냥 다시 실행하면 됨 (반복 실행 자유).
1~21번 ID를 1,000,000 / 500,000 / 250,000 baud로 순서대로 스캔한다.
"""

import sys

import scservo_sdk as scs

PORT = "/dev/ttyACM0"
BAUDS = [1_000_000, 500_000, 250_000]


def scan(baud: int) -> list[tuple[int, int]]:
    ph = scs.PortHandler(PORT)
    pk = scs.PacketHandler(0)
    if not ph.openPort():
        print(f"  [baud={baud}] 포트 열기 실패 ({PORT})")
        return []
    if not ph.setBaudRate(baud):
        print(f"  [baud={baud}] baudrate 설정 실패")
        ph.closePort()
        return []

    found = []
    for sid in range(0, 21):
        model, comm, _ = pk.ping(ph, sid)
        if comm == scs.COMM_SUCCESS:
            found.append((sid, model))
    ph.closePort()
    return found


def main() -> None:
    print(f"포트: {PORT}")
    any_found = False
    for baud in BAUDS:
        found = scan(baud)
        status = f"{len(found)}개 발견: {found}" if found else "응답 없음"
        print(f"  baud={baud:>8}  ->  {status}")
        if found:
            any_found = True

    print()
    if any_found:
        print("성공 — 서보가 응답합니다.")
        sys.exit(0)
    else:
        print("실패 — 모든 baud rate에서 응답 없음. 다음을 확인:")
        print("  1. 보드 전원 LED가 켜져 있는가")
        print("  2. 서보 버스 케이블(데이지체인)이 보드에 제대로 꽂혀 있는가")
        print("  3. USB 케이블이 데이터 지원 케이블인가 (충전 전용 케이블 아님)")
        print("  4. 서보 자체의 LED가 켜지는가 (서보에도 보통 표시등 있음)")
        sys.exit(1)


if __name__ == "__main__":
    main()
