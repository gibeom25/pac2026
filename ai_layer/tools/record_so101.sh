#!/usr/bin/env bash
# SO-101 시연 데이터 녹화 (LeRobot 0.4.4, env_lerobot). 녹화 날 이 파일의 <...> 자리만 채워서 실행.
#
# 순서 (처음 한 번):
#   1) 포트 찾기:      lerobot-find-port            (팔로워/리더 USB 를 하나씩 꽂았다 뺐다 하며 확인)
#   2) 모터 ID 설정:   lerobot-setup-motors --robot.type=so101_follower --robot.port=$FOLLOWER_PORT   (이미 됐으면 생략)
#   3) 캘리브레이션:   lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID
#                      lerobot-calibrate --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=$LEADER_ID
#      ※ 민제씨 영점 오프셋 파일과 별개. LeRobot 캘리브는 LeRobot 이 degree 를 만들기 위한 것.
#   4) 카메라 확인:    lerobot-find-cameras realsense   (D405 시리얼 번호 확인)
#   5) 텔레옵 시험:    lerobot-teleoperate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
#                        --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=$LEADER_ID
set -euo pipefail
export PATH=/home/dy/pac2026/env_lerobot/bin:$PATH

FOLLOWER_PORT=${FOLLOWER_PORT:-/dev/ttyACM0}      # <- lerobot-find-port 결과
LEADER_PORT=${LEADER_PORT:-/dev/ttyACM1}
FOLLOWER_ID=${FOLLOWER_ID:-so101_follower_pac}    # 캘리브 파일 이름표
LEADER_ID=${LEADER_ID:-so101_leader_pac}
CAM_SERIAL=${CAM_SERIAL:-<D405_SERIAL>}           # <- lerobot-find-cameras realsense 결과
REPO_ID=${REPO_ID:-d-dong-2/so101-seam-demo}      # 이름표. push_to_hub=false 라 인터넷에 안 올라감
ROOT=${ROOT:-/home/dy/pac2026/datasets/$(basename $REPO_ID)}
TASK=${TASK:-"follow the drawn line with the tool tip"}
NUM_EPISODES=${NUM_EPISODES:-20}
EPISODE_S=${EPISODE_S:-30}
RESET_S=${RESET_S:-10}

# 고정값 (바꾸지 말 것): fps 30 = DT_AI_SEC, 카메라 이름 wrist = observation.images.wrist, 320x240 = 모델 입력
lerobot-record \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{wrist: {type: intelrealsense, serial_number_or_name: '$CAM_SERIAL', width: 320, height: 240, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" \
  --dataset.repo_id="$REPO_ID" --dataset.root="$ROOT" --dataset.single_task="$TASK" \
  --dataset.fps=30 --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_S" --dataset.reset_time_s="$RESET_S" \
  --dataset.push_to_hub=false --dataset.video=true \
  --display_data=true "$@"

echo
echo "녹화 끝. 학습 전에 반드시 점검:"
echo "  cd /home/dy/pac2026/pac2026-team && PYTHONPATH=. python ai_layer/tools/check_dataset.py --repo-id $REPO_ID --root $ROOT"
