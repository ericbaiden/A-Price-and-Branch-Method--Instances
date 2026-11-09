# CTSP-LEO 实验实例

本仓库同时提供实例描述和可直接加载的冻结任务图。运行已有基准时，使用 `dataset/instances/` 中的冻结图可避免重新生成输入造成差异。

## 实例目录

| 目录 | 内容 | 数量 |
|---|---|---:|
| `dataset/instances/` | 三小时、单位体量冻结图；100–1000任务，每100任务一个规模，每规模10个种子 | 100 |
| `dataset/instances/h24/` | 24小时、单位体量冻结图；1000–10000任务，每1000任务一个规模，每规模3个种子 | 30 |
| `instances/validation/` | 缩小星座的历史实例描述；10个RS、50个CS，50/100任务标签，每规模10个种子 | 20 |
| `instances/large_scale/` | 完整星座的历史实例描述；136个RS、1321个CS，100/200/500/1000任务标签，每规模10个种子 | 40 |
| `instances/paper/` | 历史示例及任务描述，保留原文件内容 | 10 |
| `instances/revision_audit/` | 早期审计的两个完整压缩图导出及清单 | 2个图 |
| `instances/revision_audit_m200/` | M=200审计的两个完整压缩图导出及清单 | 2个图 |

历史描述文件记录配置、轨道或任务信息，字段和所含细节因版本而异。复现实验输入时应以冻结图和对应实验协议为准；不同历史目录保持独立。

冻结图总计约711MB；本次公开的全部实例输入约730MB。单个文件最大约19.3MB，使用普通Git保存，克隆后无需额外下载指针文件。大型原始求解结果另置于实验附件。

## 冻结图的内容

每个 `.npz` 文件包含：

- 图节点标识、弧起点和终点、弧类型、成本和容量；
- 任务标识、坐标、体量、优先级、释放和截止时隙；
- 每项任务的源节点和终点；
- 卫星标识，以及任务规模、种子、图大小和实例哈希。

每个规划时域目录中的 `manifest.json` 对应其本层冻结图。时隙长度为200秒；三小时包含54个时隙，24小时包含432个时隙。单位体量实例中每项任务为1GB。异质体量实验使用单独的生成器和协议，不混作这130个单位体量冻结图。

## 加载示例

在仓库根目录运行：

```python
from instance_store import load_instance

case = load_instance(tasks=100, seed=0, horizon_hours=3.0)
print(len(case.tasks), len(case.graph), case.graph.number_of_edges())
# 100 161436 554218

# 24小时示例；完整恢复数百万条弧需要相应的内存。
# case = load_instance(tasks=1000, seed=0, horizon_hours=24.0)
```

读取和检查冻结图需要NumPy与NetworkX；运行优化算法还需要相应依赖与Gurobi许可。加载器从可信实例文件中恢复NumPy对象数组。

## 完整性检查

在仓库根目录执行：

```bash
python scripts/verify_instances.py
```

`instances/file_checksums.json` 记录公开输入文件的字节数和SHA-256。检查脚本同时核验130个冻结图与各自清单的对应关系、压缩文件完整性和历史完整图导出的哈希。

输入文件保留已有原始字节，运行日志、求解结果和可重建的物理图缓存不属于本目录交付范围。

整理：ClawsGO Science Agent
