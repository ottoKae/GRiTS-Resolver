# 数据来源说明、版本保护与 SCL 配对读取

2026-10-28，0.3.1rc1 候选版。所有接口只读；不会补下载、改写 Catalog、更新旧绑定或给影像增加缩放。可复现测试见 tests/test_provenance.py、test_provenance_resolver.py 与 test_s2_quality_pairing.py。

## 1. 最常用的入口

`S1GRiTSResolver` 和 `MultiSensorResolver` 自动识别数据根的 `provenance/current.json`：

- 有记录：验证发布文件、输出关联和已支持的结构见证，再缓存索引。
- 无记录：保留旧读取方式，来源查询明确返回 `metadata_status="missing"`。不会据当前代码推断历史处理。
- 有损坏、过期或未完成写入：报错，不能当作“只是缺少记录”放行。
- 数据追加、替换或元数据版本更新后：停止使用旧实例，生产完成后重新创建。没有自动热加载。

查看来源无需同时提供 S1 与 S2：

```python
from grits_resolver import ProvenanceView

view = ProvenanceView(r"D:/data/S2")
print(view.available, view.data_revision)
info = view.explain(
    store_path="17MPU/scenes_L2A_native/zarr/s2grits_scenes_17MPU_L2A_native.zarr",
    zarr_group="20m",
    time_index=0,
    variable="swir_1",
)
print(info["metadata_status"], info["matched"])
for match in info["outputs"]:
    print(match["output"])
    print(match["sources"])
```

查询可按 `output_id`，也可按 store／group／index／variable 筛选。返回的来源保留 `used`、`selected` 等原角色，且对应当前输出代次。变量解释、未知项和证据可一并查看。没有匹配输出不会捏造来源。

S1 新月度记录可查看 `info["numeric_processing"]`：每项绑定输出及代次，`record.input.sources` 区分源声明与实际读取，`record.steps` 说明执行步骤。缺旧记录返回 unknown，矛盾记录报错；源单位声明不等于已验证真实物理量。返回结果可供调用方修改，不会改变读取器下次查询的内部记录。

历史补录的单位和处理依据可以带 `historical_output_generations`；store 说明中对应 `historical_context.output_generations`。它们是 `output_id → generation_id` 映射，不表示整库或未来批次均适用。`explain` 按所选输出及当前代次筛选这些字段；新代次不会借用旧确认，未匹配项不返回该历史说明。原始 `declared_attributes` 保留，历史确认也不会变成执行 receipt。读取混合代次的新发布包应使用支持该范围约束的更新版 Resolver；直接读取 JSON 时也必须遵守这些映射。

0.3.1rc1 支持发布语义 v1/v2。v2 在加载时校验范围所属的 store、group、variable、output 和 generation 及证据引用；损坏的范围会拒绝加载。旧 v1 和早期 v1 范围试验包继续兼容。版本号指伴随发布语义，Parquet 表、Zarr 格式及 data_revision 算法并未改版；不额外换算像元。旧读取器拒绝 v2 是保护行为，不应删除 current 或关闭 provenance 绕过。请核对实际导入的 `grits_resolver.__file__` 和 `grits_resolver.__version__`；说明更新不会自动升级已安装软件。

`ProvenanceView(..., required=True)` 或 Resolver 的 `require_provenance=True` 要求存在可验证的元数据。它不表示历史记录全部是执行真值；`generation_id=null`、`partial_evidence` 和未知的上游校正仍按原记录解释。

## 2. 原有 S1 读取方式保持兼容

```python
from grits_resolver import S1GRiTSResolver

s1 = S1GRiTSResolver(r"D:/data/S1")
pair = s1.tracks("17MPU")[0]  # 多轨数据应按原规则选择具体轨道
values, valid, months = s1.read_monthly(pair, row=100, col=100, patch_size=3)
print(s1.explain_output(
    store_path=pair.monthly.zarr_path,
    time_index=0,
    variable="VV_dB",
))
```

从 Resolver 取得的 `S1TrackPair` 带有版本守卫，`S1GRiTSResolver.read_monthly(pair, ...)` 的静态调用方式仍然可用。月序列、static 和双轨共同读取均在实际读取前后检查；发现 pending 或变化时，不返回混合版本的数据。

手工构造的不带守卫的旧 pair 仍属旧接口用法，不代表已核验来源。读取结果继续沿用原有 dtype、有效性掩膜和缺测处理，元数据不会额外触发 dB 或反射率换算。

## 3. 显式使用新的独立 SCL 配对

先由 S2-GRiTS 的 `rebind_quality` 产生稳定配对版本。然后：

```python
from grits_resolver import MultiSensorResolver
from grits_resolver.spatial import TargetWindow

resolver = MultiSensorResolver(
    r"D:/data/S1", r"D:/data/S2",
    s2_quality_root=r"D:/data/S2_quality",
    s2_quality_pairing="current",
)
pair = resolver.s2_scene_quality("17MPU", "你的真实产品ID", allow_missing=True)
result = resolver.read_s2_quality_on_s1(
    pair, window=TargetWindow(row_off=100, col_off=100, height=64, width=64),
)
print(result["quality_available"])
print(resolver.explain_output(
    "sentinel-2", store_path=pair.spectral_20m.zarr_path,
    zarr_group="20m", time_index=pair.spectral_20m_index,
))
```

| 选项 | 使用的发布 |
| --- | --- |
| 默认 `s2_quality_pairing="legacy"` | 原 v1 `companion.json` 与原关联表 |
| `s2_quality_pairing="current"` | `provenance/pairings/current.json` 指向的独立配对版本，同时核验原接受证据 |

不会因为看见新文件就悄悄切换；不支持的版本、变化的 Catalog、错误索引或缺少必要证据会报错。三个组各自定位，不能用光谱 10 m 的下标替代光谱 20 m 或 SCL 下标。

配对中 `quality_available=false` 的场景可通过 `allow_missing=True` 查询；`pair.unavailable_reason` 保留原因。读取时返回无质量覆盖的状态，不能把零数组解释成晴空。存在 SCL 时保留 `uint8` 与原类别 0；`quality_valid` 表示非零质量覆盖，不等于云筛选结果。SCL 仍只允许最近邻空间采样。

质量读取继续核对原发布记录中的 **Zarr、xarray、numcodecs 精确版本**。按质量库附带的版本配置环境，不要修改环境锁或原 marker 绕过检查。通用候选包安装不会自动匹配任意历史质量库的精确依赖。

## 4. 搬到另一块磁盘或另一台电脑

新 provenance 表使用相对 store 路径。旧 Catalog 或 SCL binding 若含生产机的绝对根路径，显式提供映射；文件原件不修改：

```python
resolver = MultiSensorResolver(
    r"E:/delivery/S1", r"E:/delivery/S2",
    s2_quality_root=r"E:/delivery/S2_quality",
    s2_quality_pairing="current",
    root_map={
        r"D:/old/S1": r"E:/delivery/S1",
        r"D:/old/S2": r"E:/delivery/S2",
        r"D:/old/S2_quality": r"E:/delivery/S2_quality",
    },
)
```

映射按完整路径组件匹配，优先最具体的旧根；不按文件名猜位置。映射后的 Catalog 资产必须留在所绑定的数据根内。SCL 原绑定的 spectral／quality 根也需映射到相应目的根。原 `row` 和 static 的原始引用保留，便于核验原证据。

本轮实际验收的是 Windows 本地目录迁移。跨操作系统路径转换、网络文件系统及 junction／symlink 交付不在已验收范围；完整交付应包含实际文件。

如果只有独立的历史元数据目录，影像已在另一处：

```python
view = ProvenanceView(
    r"E:/data/S2",
    metadata_root=r"E:/metadata-only/S2",
    required=True,
)
# 对应构造参数也可用：
# S1GRiTSResolver(..., metadata_root=...)
# MultiSensorResolver(..., s1_metadata_root=..., s2_metadata_root=...)
```

仍须通过源结构与版本检查。独立元数据包不会因此变成包含影像的完整交付包。

## 5. 每次读取检查什么

加载时验证元数据清单及摘要、表结构与证据引用、输出代次、实际时间值、网格和变量布局；SCL 还验证独立配对发布、旧接受证据及两端版本。查询和窗口读取复用缓存。

读取前后检查 current／pending／Catalog 状态及所读 store 的小型结构文件。不会按每个窗口遍历全部 store、重新计算 Catalog 哈希或重跑源—SCL 全像元验收。带记录的 S2 场景身份索引加载后复用；没有对应记录的旧布局保留原身份读取检查。

SCL 的全像元摘要是生产者验收证据。P4 核验它属于所声明的发布和场景，不在每次打开 Resolver 时重新解码整套质量影像。结构见证与代次号不能识别所有绕过生产流程、只改影像块的篡改；需要这种核验时应单独执行数据完整性检查。

读取守卫适用于本轮接入的 S1/MultiSensorResolver 物化数组接口。低层 `open_dataset` 返回惰性 xarray，直接调用它不享有“未来读取”保护；独立 `OpticalDailyResolver` 日合成入口也未纳入本次 P4。生产与读取仍应分阶段进行，轻量守卫不是任意并发读写的事务保证。
