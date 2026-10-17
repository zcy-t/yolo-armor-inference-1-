#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26n 装甲板四点检测视频推理脚本

功能：
  1. 加载 best.pt 模型权重
  2. 读取输入视频，逐帧推理
  3. 在画面上绘制检测框、类别、置信度，以及每个目标的 4 个二维关键点
  4. 保存推理结果视频

关键点说明：
  - 关键点来自模型预测（results[0].keypoints），不是检测框四角
  - 每个目标输出 4 个二维关键点，按顺序编号为 1~4
  - 关键点之间用线连接，便于看出所属目标
"""

import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


# 6 类装甲板名称（B=蓝色, R=红色, 数字=兵种编号）
CLASS_NAMES = {
    0: "B1",
    1: "B2",
    2: "B3",
    3: "R1",
    4: "R2",
    5: "R3",
}

# 不同类别使用不同框颜色 (B, G, R)
CLASS_COLORS = {
    0: (255, 100, 0),    # B1 - 蓝色
    1: (255, 0, 100),    # B2 - 深蓝
    2: (200, 50, 0),     # B3 - 青蓝
    3: (0, 0, 255),      # R1 - 红色
    4: (0, 100, 255),    # R2 - 橙红
    5: (0, 200, 255),    # R3 - 橙色
}

# 4 个关键点的颜色（循环使用）
KEYPOINT_COLORS = [
    (0, 255, 255),   # 黄
    (0, 255, 0),     # 绿
    (255, 0, 255),   # 紫
    (255, 255, 0),   # 青
]

# 关键点连线颜色
SKELETON_COLOR = (0, 255, 255)


def draw_detection(frame, box, class_id, confidence, keypoints_xy):
    """
    在一帧上绘制一个检测目标的框、标签和关键点。

    参数:
        frame:           BGR 图像 (numpy array)
        box:             [x1, y1, x2, y2] 检测框坐标
        class_id:        类别索引
        confidence:      置信度
        keypoints_xy:    shape (4, 2) 的关键点坐标数组
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    # 框颜色
    color = CLASS_COLORS.get(class_id, (0, 255, 0))

    # 画检测框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 标签文字
    name = CLASS_NAMES.get(class_id, str(class_id))
    label = f"{name} {confidence:.2f}"

    # 标签背景
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 画 4 个关键点（模型预测的，不是框角）
    if keypoints_xy is not None and len(keypoints_xy) >= 4:
        kpts = keypoints_xy[:4]  # 取前 4 个

        # 用线连接 4 个关键点（按 1->2->3->4->1 闭合）
        for i in range(4):
            p1 = (int(kpts[i][0]), int(kpts[i][1]))
            p2 = (int(kpts[(i + 1) % 4][0]), int(kpts[(i + 1) % 4][1]))
            cv2.line(frame, p1, p2, SKELETON_COLOR, 2)

        # 画关键点圆点和编号
        for i, kp in enumerate(kpts):
            kx, ky = int(kp[0]), int(kp[1])
            kp_color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
            cv2.circle(frame, (kx, ky), 5, kp_color, -1)
            cv2.circle(frame, (kx, ky), 5, (0, 0, 0), 1)  # 黑边
            # 编号文字
            cv2.putText(frame, str(i + 1), (kx + 7, ky - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, kp_color, 2)


def run_inference(model_path, video_path, output_path, conf_threshold=0.25,
                  device="", imgsz=640):
    """
    加载模型并对视频逐帧推理，绘制检测结果并保存输出视频。

    参数:
        model_path:      best.pt 权重路径
        video_path:      输入视频路径
        output_path:     输出视频路径
        conf_threshold:  置信度阈值
        device:          "cpu" / "cuda" / "" 自动
        imgsz:           推理输入尺寸
    """
    # 1. 加载模型
    print(f"[INFO] 正在加载模型: {model_path}")
    model = YOLO(model_path)

    # 打印模型类别信息
    if hasattr(model, 'names') and model.names:
        print(f"[INFO] 模型类别: {model.names}")

    # 2. 打开视频
    if not os.path.exists(video_path):
        print(f"[ERROR] 视频文件不存在: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 视频信息: {width}x{height}, {fps:.1f} FPS, 共 {total_frames} 帧")

    # 3. 准备输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"[INFO] 输出视频: {output_path}")

    # 4. 逐帧推理
    frame_idx = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 推理
        results = model.predict(
            frame,
            conf=conf_threshold,
            device=device if device else None,
            imgsz=imgsz,
            verbose=False,
        )

        result = results[0]

        # 提取检测框
        boxes = result.boxes
        n_det = len(boxes) if boxes is not None else 0
        total_detections += n_det

        if n_det > 0:
            # 提取框、类别、置信度
            xyxy = boxes.xyxy.cpu().numpy()       # (N, 4)
            clss = boxes.cls.cpu().numpy().astype(int)   # (N,)
            confs = boxes.conf.cpu().numpy()     # (N,)

            # 提取关键点 (N, K, 2)
            kpts = None
            if result.keypoints is not None:
                kpts = result.keypoints.xy.cpu().numpy()

            for i in range(n_det):
                kpts_i = kpts[i] if kpts is not None else None
                draw_detection(frame, xyxy[i], clss[i], confs[i], kpts_i)

        # 帧计数信息
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}  Detections: {n_det}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)

        if frame_idx % 30 == 0:
            print(f"[INFO] 已处理 {frame_idx}/{total_frames} 帧")

    # 5. 清理
    cap.release()
    out.release()
    print(f"[INFO] 推理完成！共处理 {frame_idx} 帧，累计检测 {total_detections} 个目标")
    print(f"[INFO] 结果视频已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO26n 装甲板四点检测视频推理脚本"
    )
    parser.add_argument(
        "--model", type=str, default="weights/best.pt",
        help="模型权重路径 (默认: weights/best.pt)"
    )
    parser.add_argument(
        "--video", type=str, default="input.mp4",
        help="输入视频路径 (默认: input.mp4)"
    )
    parser.add_argument(
        "--output", type=str, default="output/result.mp4",
        help="输出视频路径 (默认: output/result.mp4)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="置信度阈值 (默认: 0.25)"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="推理设备: cpu / cuda (默认自动选择)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="推理输入尺寸 (默认: 640)"
    )
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    run_inference(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        conf_threshold=args.conf,
        device=args.device,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
