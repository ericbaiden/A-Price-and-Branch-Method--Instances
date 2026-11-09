#!/usr/bin/env python3
"""
检查 M=600 重跑数据完整性
Usage: python check_m600_rerun.py
"""
import json
from pathlib import Path
from collections import defaultdict

# 重跑输出目录
RERUN_DIR = Path(__file__).parent.parent / "results" / "m600_baseline_rerun"

def check_completeness():
    """检查是否所有任务都完成"""
    expected = {
        'tasks': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'algorithms': ['greedy', 'grasp', 'alns'],
        'seeds': list(range(10))
    }

    missing = []
    stats = defaultdict(lambda: {'completed': 0, 'failed': 0})

    for ntasks in expected['tasks']:
        for algo in expected['algorithms']:
            for seed in expected['seeds']:
                result_file = RERUN_DIR / f"{algo}_M600_n{ntasks}_s{seed}.json"

                if not result_file.exists():
                    missing.append(f"{algo}_n{ntasks}_s{seed}")
                    stats[f"{algo}_{ntasks}"]['failed'] += 1
                else:
                    stats[f"{algo}_{ntasks}"]['completed'] += 1

    print("=== M=600 重跑完整性检查 ===\n")

    if missing:
        print(f"❌ 缺失 {len(missing)} 个结果文件:")
        for m in missing[:10]:
            print(f"   - {m}")
        if len(missing) > 10:
            print(f"   ... 还有 {len(missing)-10} 个")
    else:
        print("✅ 所有 300 个结果文件都存在")

    print("\n=== 各配置完成情况 ===")
    for key in sorted(stats.keys()):
        completed = stats[key]['completed']
        failed = stats[key]['failed']
        total = completed + failed
        print(f"{key:20s}: {completed}/{total} 完成")

    return len(missing) == 0

def extract_summary():
    """提取 10-seed 均值统计"""
    expected = {
        'tasks': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'algorithms': ['greedy', 'grasp', 'alns']
    }

    summary = {}

    for ntasks in expected['tasks']:
        for algo in expected['algorithms']:
            objectives = []
            times = []
            completed = []

            for seed in range(10):
                result_file = RERUN_DIR / f"{algo}_M600_n{ntasks}_s{seed}.json"
                if not result_file.exists():
                    continue

                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    objectives.append(data.get('objective', float('inf')))
                    times.append(data.get('solve_time', 0))
                    completed.append(data.get('num_completed', 0))
                except Exception as e:
                    print(f"⚠️  读取 {result_file.name} 失败: {e}")

            if objectives:
                import statistics
                summary[f"{algo}_{ntasks}"] = {
                    'obj_mean': statistics.mean(objectives),
                    'obj_std': statistics.stdev(objectives) if len(objectives) > 1 else 0,
                    'time_mean': statistics.mean(times),
                    'completed_mean': statistics.mean(completed),
                    'completed_std': statistics.stdev(completed) if len(completed) > 1 else 0,
                    'n_seeds': len(objectives)
                }

    print("\n=== 10-seed 汇总统计 (M=600) ===\n")
    print(f"{'Config':<20s} {'Obj (mean±std)':<25s} {'Time':<10s} {'Completed':<15s}")
    print("-" * 75)

    for key in sorted(summary.keys()):
        s = summary[key]
        obj_str = f"{s['obj_mean']:.1f} ± {s['obj_std']:.1f}"
        time_str = f"{s['time_mean']:.1f}s"
        comp_str = f"{s['completed_mean']:.1f} ± {s['completed_std']:.1f}"
        seed_note = "" if s['n_seeds'] == 10 else f" ({s['n_seeds']}/10)"
        print(f"{key:<20s} {obj_str:<25s} {time_str:<10s} {comp_str:<15s}{seed_note}")

    return summary

if __name__ == "__main__":
    complete = check_completeness()
    print()
    if complete:
        extract_summary()
    else:
        print("\n⚠️  数据不完整，无法生成汇总统计")
        print("   请检查重跑任务是否全部完成")
