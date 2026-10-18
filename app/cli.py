"""命令行入口：train / predict。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="植物叶片病虫害识别（复现版）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gen", help="生成合成数据集")
    p_train = sub.add_parser("train", help="训练并评估")
    p_train.add_argument("--images", default=str(BASE / "data" / "images"))
    p_pred = sub.add_parser("predict", help="预测单张图片")
    p_pred.add_argument("image")

    args = parser.parse_args()

    if args.cmd == "gen":
        from .dataset import generate_dataset

        samples = generate_dataset(BASE / "data" / "images")
        print(f"已生成 {len(samples)} 张图像 -> {BASE / 'data' / 'images'}")
        return 0

    if args.cmd == "train":
        from . import model

        result = model.train(Path(args.images))
        report = (
            f"# 训练报告\n\n- 训练集：{result.n_train} 张，测试集：{result.n_test} 张\n"
            f"- 准确率：**{result.accuracy:.4f}**\n\n```\n{result.report}\n```\n"
        )
        (BASE / "model").mkdir(exist_ok=True)
        (BASE / "model" / "report.md").write_text(report, encoding="utf-8")
        print(report)
        return 0

    if args.cmd == "predict":
        from .model import predict

        print(predict(Path(args.image)))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
