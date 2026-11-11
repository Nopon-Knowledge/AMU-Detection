# 验证记录

日期：2026-11-11，Asia/Shanghai。

环境：macOS Apple M4，CPU，Python 3.9.13，PyTorch 2.2.2，torchvision 0.17.2，Ultralytics 8.3.40。验证依赖安装在隔离虚拟环境中。

执行：

```text
python -m pip check
No broken requirements found.

python -m pytest -q --disable-warnings
10 passed, 14 warnings in 60.50s

python -m compileall -q amu tests
通过

python -m amu.cli --help
正常列出 7 个子命令
```

14 条 warning 来自第三方 matplotlib/pyparsing 的弃用提示，不是测试失败。

## 测试范围

- 额外浅层分支 g=0 时，任意改变 S 都不改变融合输入；g=1 的正对照会改变。
- 部分边界区域的平均池化和门控广播。
- 指数门控映射和探针二值/连续掩码的取值支持。
- 缓存前半网络与完整前向一致；保存和重新加载权重后的前向一致。
- 固定目标损失与锁定 Ultralytics 原版的单图损失数值一致，包含真实前景分配。
- 同一对预测的效用标签不随 batch 中其他图像改变。
- 相同 on/off 预测的效用为零；交换 on/off 使对称效用变号；固定目标评估不重算目标分配。
- 空 GT 图像损失有限；M7 反向传播只进入路由器，检测器梯度隔离。
- 合成的 4 张训练图片、2 张验证图片：1 epoch 探针训练 → 最终 FP32 EMA 导出 → 8 个区域标签 → M8/M9/M10 各 1 epoch → 权重冻结检查 → 验证 → 图片预测 → 延迟接口 → M7 1 epoch。合成数据只验证代码接口，不代表科研实验数据。
- 标签生成拒绝 train/val 中相同图片路径。

另以真实主配置的维度执行 640×640 前向：8 类和 10 类均通过，S/T/I 为 64/128/128×80×80，区域描述为 1×10×10×320，输出分别为 1×12×8400 和 1×14×8400。

## 尚未验证

未在真实 KITTI/BDD100K 上运行 300 epoch；未验证论文 AP、消融增益、GPU 耗时或跨设备一致性。没有 CUDA 测试、正式数据转换或原始 split ID。参数量差异及具体工程重建选择见 `IMPLEMENTATION.md`。
