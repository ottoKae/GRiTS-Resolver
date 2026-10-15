# 跨瓦片候选与显式目标格网（2026-10-16）

S2源瓦片身份与目标S1格网必须分开。此增强不修改源库、不下载、不进行时间合成，也不自动把现有模型入口切换成新输入版本。

```python
target = resolver.s1_entry("17MPU")  # 多轨场景须由调用者明确选择实际目标条目
pairs = resolver.s2_quality_candidates_on_target(
    target, window=window, start="2018-01-01", end="2021-01-01"
)
for pair in pairs:
    quality = resolver.read_s2_quality_on_s1(
        pair, window=window, target_entry=target
    )
```

- 候选按照目标窗口与各源数组边界相交查询，跨tile，不按同名瓦片限制；日期为左闭右开区间。每个场景按collection/tile/item去重，10m/20m目录记录不会重复计景。
- 数组矩形只是保守候选范围；实际采集足迹、源nodata、SCL和十波段有效性仍须检查，不能把返回候选数当有效观测数。
- 场景与质量身份使用源tile，目标输出使用显式target_entry。没有传target_entry时保留原同名行为。
- `quality_valid`只表示非0覆盖，不表示晴空。缺质量、源范围外及SCL=0均不能充当晴空。
- 此函数用nearest返回SCL类别，不执行“30m完整支持内全部原生像元清晰”的研究质量规则。需要该规则的模型仍应在原生SCL上计算面积支持，并按对应光谱原生格网作面积加权汇总。
- 重叠tile可能来自同次采集。不同tile/item不能直接当独立观测追加到月中值；按采集身份去重、来源优先级与完整像元支持另行处理。本API只返回场景，不替调用者做这些决策。
- 新旧输入必须版本化。当前历史Tessera栅格入口仍是同名tile输入；仅调用本API的显式新入口才会使用邻景。

验收包括邻瓦片源身份保持、显式目标坐标、缺质量语义、时间范围、重复分辨率记录、源矩形不相交、同名兼容，以及源边缘部分相交但目标中心越界的0值。真实试点证据位于Tessera项目的`outputs/tile17MPU_cross_tile_probe_20261016_v1/`。

## 同次采集去重

`s2_acquisition.acquisition_key(properties)` 要求原始 STAC 的 `s2:datatake_id`，返回卫星、datatake 起始时间与绝对轨道号的组合。处理基线不属于采集身份；不同场景的瓦片中心观测时间也不用于分组。缺少身份或与平台信息冲突时抛出错误，不退回同日分组。

```python
from grits_resolver.s2_acquisition import acquisition_key, first_valid_mosaic

key = acquisition_key(stac_feature["properties"])
# 先按 key 分组。每组内的 observations 已按调用者冻结的优先级排列。
# values: [pixel, band]；valid: [pixel]，必须已检查全部波段和质量。
values, chosen_source, multiplicity = first_valid_mosaic(observations)
```

`first_valid_mosaic` 对每个像元选首个有效来源，所有波段来自同一个来源。无有效来源时 `chosen_source=-1`、值为 NaN；`multiplicity` 只是来源重叠数，不能作为时间样本权重。先合并同次采集，再进行月度统计。不会把不同过境合并，也不会在不同场景间拼凑有效波段。

来源优先级、完整像元面积支持和月度统计仍由应用明确规定。示范新入口：Tessera项目 `tooling/diagnostics/20261016_cross_tile_full/inputs.py`；独立输入版本 `native30_cross_tile_unique_datatake_monthly_v1`。该入口使用原17MPU优先、随后tile/item稳定排序、原生SCL完整30m支持、反射率面积加权及独立采集月中位数。仅支持已核对的同CRS平行格网，遇到其他格网明确失败，不静默错配。通用类别读取接口仍可使用显式目标重投影。

新版本与历史同名瓦片入口并存，不改变默认行为。真实整瓦片验收结果由应用项目 `outputs/tile17MPU_cross_tile2020_full_20261016_v1/` 保存；是否完成以该目录的 `COMPLETE.json` 为准，不以代码或文档存在代替验收。

## 17MPU真实数据验收摘要

2020-12结束的一年／三年输入，原30m核心区3333×3333，共11,108,889格。126条场景对应82次采集，其中33个采集组有多个瓦片。全图抑制1,922,111次重复场景像元贡献；独立有效像元采集次数为49,626,468，二者相加等于有效源场景像元观测次数51,548,579。

- 原场景路径与新入口限同名场景的输入逐值一致；四个8×8真实窗口通过点／块一致性和重复加入同次采集不变性检查。
- 196块完成并保存独立月度输入与校验值。融合一年覆盖65.0202%→75.2496%，三年79.9296%→89.6636%；旧覆盖未丢失，S1地图逐像元一致。
- 待发布索引的隔离副本：15项测试通过，3项可选真实数据测试因未设置环境变量跳过。上述应用层真实验收另行执行，不把跳过项计为通过。
- 新图保留冻结中心，覆盖增加不代表人工林分类精度已经提高。剩余三年无输出格位中86.31%位于候选景登记足迹之外；这是元数据诊断，不是逐像元云／nodata证明。

本地运行绑定：`RUN_SPEC.json` SHA256 `547a10e2850c93c8716892c059659f2422d29cefa6f83fdbde39640b229a7f6d`。此应用验收使用同CRS平行格网的完整面积支持策略，不宣称已经验证任意跨CRS面积合成。
