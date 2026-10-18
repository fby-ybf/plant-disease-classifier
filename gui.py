"""Tkinter 桌面应用：选择图片 -> 显示类别、置信度与各类别概率。

仅在本地有桌面环境时运行（python gui.py）；CI / 无显示环境不要导入本模块。
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from app.model import predict


def main() -> None:
    root = tk.Tk()
    root.title("植物叶片病虫害识别（复现版）")
    root.geometry("520x320")

    result_var = tk.StringVar(value="请选择一张叶片图片")
    detail_var = tk.StringVar(value="")

    def choose() -> None:
        path = filedialog.askopenfilename(
            title="选择叶片图片", filetypes=[("图片", "*.png *.jpg *.jpeg")]
        )
        if not path:
            return
        try:
            out = predict(Path(path))
        except FileNotFoundError as exc:
            messagebox.showerror("错误", str(exc))
            return
        result_var.set(f"预测类别：{out['label']}　置信度：{out['confidence']:.2%}")
        detail_var.set(
            "各类别概率：" + "　".join(f"{k}={v:.2%}" for k, v in out["probabilities"].items())
        )

    tk.Button(root, text="选择图片", command=choose, width=16).pack(pady=18)
    tk.Label(root, textvariable=result_var, font=("Microsoft YaHei", 12)).pack(pady=6)
    tk.Label(root, textvariable=detail_var, wraplength=480, fg="#555").pack(pady=6)
    tk.Label(root, text="提示：先运行 python -m app.cli gen && python -m app.cli train",
             fg="#999").pack(side="bottom", pady=10)
    root.mainloop()


if __name__ == "__main__":
    main()
