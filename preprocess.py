import os
import shutil

train_dir = "/4TB_2nd/Seongmin/scene/train"
val_dir = "/4TB_2nd/Seongmin/scene/val"

# train 기준으로 클래스 목록
train_classes = set(os.listdir(train_dir))

# val 내부 클래스 폴더 순회
for cls in os.listdir(val_dir):
    cls_path = os.path.join(val_dir, cls)
    if os.path.isdir(cls_path):  # 폴더일 경우만 확인
        if cls not in train_classes:
            print(f"삭제 대상: {cls_path}")
            shutil.rmtree(cls_path)  # 폴더 전체 삭제
