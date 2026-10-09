"""AI 추론 계층 -> 송지수 제어 계층(ActionChunk) 다리.

- protocol.py        규약 복사본 (messages / chunk_codec / snapshot_codec)
- chunk_builder.py   모델 청크 (T,7) -> ActionChunk (한계 클립, eef 매핑 스위치)
- snapshot_adapter.py StateSnapshot -> observation.state (9D), anchor 선택
- ai_node.py         실행 루프: 스냅샷 수신 -> 이미지 -> 추론 -> 청크 송신 (ZeroMQ)

2026-10-10: Problem.md/Experiment_Plan.md의 "추론-제어 인터페이스 불일치"(버퍼 고갈/청크 경계
간극/트리거 채터링/지연으로 인한 진동) 문제를 다루는 제어 쪽 로직 4개 추가(원래 설계 계획대로
이 패키지에 위치, 기존 protocol/ai_node는 무수정) — 검증 과정(point-mass + MuJoCo 실제 SO-101
관절 동역학)은 ai_layer/tools/interface_benchmark.py, joint_dynamics_bench.py 등 참고.
- chunk_buffer.py         버퍼 고갈: 2/3 지점부터 exponential 감속(DRAINING) 후 하한선(HOLDING)
- anchor_resync.py        청크 경계 간극: 도착한 청크를 실제 현재 위치에 단조증가+방향 유사도로 재정렬
- trigger_logic.py        트리거 채터링: on/off 임계값을 분리한 히스테리시스
- open_loop_corrector.py  센서 노이즈: 등속 모델 칼만 필터로 열린 루프 구간 보정
"""
