import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 로그 디렉토리 경로
log_dir = '/home/nhj/BVRGym/MY/MA/BVRGym/LOG_FILE/tb_logs/MA/'

# EventAccumulator를 사용하여 로그 파일 읽기
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# 모든 키 가져오기
tags = event_acc.Tags()['scalars']

# 각 태그의 값 출력
for tag in tags:
    events = event_acc.Scalars(tag)
    for event in events:
        print(f"Step: {event.step}, Value: {event.value}, Tag: {tag}")
