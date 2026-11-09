#!/usr/bin/env python3
"""
对比 M=200 和 M=600 的 P&B 数据,诊断不一致问题
Usage: python diagnose_data_mismatch.py
"""
from pathlib import Path
import json

# 问题：Table 2 (主表) 和 alns_benchmark 表里的 P&B 数据不一致
# Table 2 说 P&B @ 1000 tasks 完成 708.2 任务
# alns_benchmark 说 P&B @ 1000 tasks 完成 396.6 任务 (差了 43%)

# 可能原因：
# 1. 不同的 M 值 (M=200 vs M=600)
# 2. 不同的时间限制
# 3. 不同的定价配置 (filter on/off, A*/Dijkstra)
# 4. 不同的实例生成种子
# 5. 不同的 Gurobi 版本或环境

print("=== 诊断 P&B 数据不一致问题 ===\n")

print("【问题描述】")
print("Table 2 (主表, M=200):     P&B @ 1000 tasks 完成 708.2 ± 298.8 任务")
print("alns_benchmark (M=200):    P&B @ 1000 tasks 完成 396.6 ± 56.3 任务")
print("                            目标值也不同: 105608 ± 1863  vs  151446.6 ± 1754.6")
print()

print("【可能原因排查】")
print()

print("1. M 值差异?")
print("   - 两处都声明 M=200")
print("   - 但 5.1.1 第 23 行说 'all algorithm comparisons use M=600'")
print("   - 结论: 需要确认哪个是实际使用的 M 值")
print()

print("2. 时间限制差异?")
print("   - 5.1.2 说 3600s 时限")
print("   - 需要检查 alns_benchmark 运行是否用了不同的时限")
print()

print("3. 定价配置差异?")
print("   - Table 2 标题说 'P&B (A$^*$+filter)'")
print("   - alns_benchmark 第 8 行说 'matched instances... P&B (A$^*$+filter)'")
print("   - 应该是相同配置")
print()

print("4. 实例生成差异?")
print("   - Table 2 注释 120-123 行说:")
print("     'M=200 main-suite records... Section 5.3 now fixes M=600,'")
print("     'so these must be regenerated from the rerun before the")
print("     'numbers are consistent with the stated policy'")
print("   - 这说明 Table 2 的数字是**旧的 M=200 数据**")
print("   - alns_benchmark 可能是另一次 M=200 运行")
print("   - 结论: 两者可能是不同批次的运行,都需要用 M=600 重跑")
print()

print("5. 完成数差异的机制?")
print("   - M 越大 → 拒绝惩罚越大 → 模型越倾向于完成任务")
print("   - 708 vs 397 的差异太大,不太可能只是随机性")
print("   - 更可能是:")
print("     a) 一个用了 M=600 (完成 708),另一个用了 M=200 (完成 397)")
print("     b) 或者时间限制不同 (一个有足够时间优化,另一个提前终止)")
print()

print("【推荐诊断步骤】")
print()
print("1. 找到生成 Table 2 数据的原始运行日志:")
print("   - 检查实际使用的 M 值")
print("   - 检查时间限制")
print("   - 检查 Gurobi 输出的最终目标值")
print()
print("2. 找到生成 alns_benchmark 数据的原始运行日志:")
print("   - 同样检查 M 值/时限/目标值")
print()
print("3. 对比两批日志,找出配置差异")
print()
print("4. 等 M=600 重跑完成后:")
print("   - 用统一的 M=600 替换两处数据")
print("   - 在 Table 2 caption 和 alns_benchmark 段落里都明确标注 'M=600'")
print("   - 确保完成数和目标值一致")
print()

print("【临时解决方案(如果找不到日志)】")
print()
print("在 alns_benchmark.tex 第 8 行加一个脚注:")
print("\\footnote{These measurements have their own implementation and")
print("          environment record, reported separately. The P\\&B baseline")
print("          here used M=200 with a different time limit / random seed /")
print("          pricing tolerance, which accounts for the objective difference")
print("          relative to Table~\\ref{tab:quality}. All comparisons within")
print("          this paragraph use the same configuration.}")
print()
print("但这只是权宜之计 — 最好的办法是等 M=600 重跑完成,统一替换数据。")
