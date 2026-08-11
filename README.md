# GRiTS-Resolver

独立的 S1-GRiTS / S2-GRiTS 虚拟数据立方体解析器。它读取 canonical Parquet，兼容 S1 历史瓦片相对路径与 S2 全局根目录相对路径，按窗口惰性地把 S2 原生网格投影到 S1 30 m 网格，并提供 `exact/nearest/backward/forward` 显式时间连接。

数据生产仍由 S1-GRiTS 和 S2-GRiTS 负责；应用仓库只依赖本包。

```powershell
conda env create -f environment.yml
conda activate py312_grits_resolver
python -m pip install -e . --no-build-isolation
```

## 核心语义

- `load_catalog()` 将 canonical v7 数据根目录相对路径和历史 S1 瓦片相对路径统一解析为可验证的本地绝对路径。
- catalog 只接受 `status=complete`，读取时确认 Parquet 没有发生代际替换，并在初始化时核对 catalog 与 Zarr 时间轴。
- `MultiSensorResolver.time_join()` 明确要求 `exact`、`nearest`、`backward` 或 `forward`，并可设置时间容差；不会隐式合并时间轴。
- `read_s2_on_s1()` 只读取目标 S1 窗口所需的 S2 源块，然后在内存中投影到 S1 30 m 网格，不生成新的事实数据文件。
- `read_s2_variables_on_s1()` 一次打开同一 Zarr group 并批量读取变量；连续变量使用 bilinear，`scl`、`cloud_probability` 和掩膜使用 nearest。
- S2/HLS 共同光学别名为 `blue`、`green`、`red`、`nir_narrow`、`swir_1`、`swir_2`。

## 17MPU 示例

```python
import numpy as np
from grits_resolver import MultiSensorResolver
from grits_resolver.spatial import TargetWindow

resolver = MultiSensorResolver(s1_root, s2_root)
matches = resolver.time_join(
    "17MPU", method="nearest", tolerance=np.timedelta64(20, "D")
)
nir = resolver.read_s2_on_s1(
    "17MPU", "nir_narrow", s2_time_index=0,
    window=TargetWindow(1300, 2200, 256, 256),
)
```

## 边界

resolver 是只读层。共享缓存、Zarr 和 catalog 的跨进程写入锁由 S2-GRiTS 生产端负责；resolver 不修改事实数据。
