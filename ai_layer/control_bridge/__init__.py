"""AI 추론 계층 -> 송지수 제어 계층(ActionChunk) 다리.

- protocol.py        규약 복사본 (messages / chunk_codec / snapshot_codec)
- chunk_builder.py   모델 청크 (T,7) -> ActionChunk (한계 클립, eef 매핑 스위치)
- snapshot_adapter.py StateSnapshot -> observation.state (9D), anchor 선택
- ai_node.py         실행 루프: 스냅샷 수신 -> 이미지 -> 추론 -> 청크 송신 (ZeroMQ)
"""
