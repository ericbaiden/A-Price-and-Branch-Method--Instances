#!/usr/bin/env python3
"""
从 M=600 重跑数据生成 LaTeX 表格
Usage: python generate_m600_tables.py > output.tex
"""
import json
import statistics
from pathlib import Path

RERUN_DIR = Path(__file__).parent.parent / "results" / "m600_baseline_rerun"

def load_rerun_data():
    """加载所有重跑数据"""
    data = {}
    for ntasks in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        data[ntasks] = {}
        for algo in ['greedy', 'grasp', 'alns']:
            runs = []
            for seed in range(10):
                result_file = RERUN_DIR / f"{algo}_M600_n{ntasks}_s{seed}.json"
                if result_file.exists():
                    with open(result_file) as f:
                        runs.append(json.load(f))
            data[ntasks][algo] = runs
    return data

def compute_stats(runs, field):
    """计算均值和标准差"""
    values = [r[field] for r in runs if field in r]
    if not values:
        return None, None
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0
    return mean_val, std_val

def generate_main_table(data, pb_data):
    """生成主表 (Table 2): P&B vs Greedy"""
    print("% === Table 2: P&B vs Greedy @ M=600 ===")
    print("\\begin{table}[!htbp]")
    print("\\ExpTableStyle")
    print("\\caption{Solution quality of price-and-branch (P\\&B) and the Greedy baseline at $M=600$ (10 seeds per scale). The objective column reports 10-seed mean $\\pm$ standard deviation; the Greedy gap is the mean relative excess over the P\\&B objective; the certified-bound gap \\eqref{eq:gap-def} is the upper bound on the distance to the LP optimum.}")
    print("\\label{tab:quality}")
    print("\\begin{center}")
    print("\\begin{tabular}{@{}lrrrrrr@{}}")
    print("\\toprule")
    print(" & \\multicolumn{2}{c}{Objective} & \\multicolumn{2}{c}{Solve time (s)} & \\multicolumn{2}{c}{P\\&B detail} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    print("Tasks & \\multicolumn{1}{c}{P\\&B} & \\multicolumn{1}{c}{Greedy} & \\multicolumn{1}{c}{P\\&B} & \\multicolumn{1}{c}{Greedy} & Greedy gap & Certified gap \\\\")
    print("\\midrule")

    for ntasks in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        # P&B 数据（从已有的 pb_data 读取，或者如果有 P&B 重跑数据也可以用）
        pb_obj_mean, pb_obj_std = pb_data[ntasks]['obj']
        pb_time_mean, _ = pb_data[ntasks]['time']
        pb_gap = pb_data[ntasks]['gap']

        # Greedy 数据
        greedy_runs = data[ntasks]['greedy']
        if not greedy_runs:
            continue

        gr_obj_mean, gr_obj_std = compute_stats(greedy_runs, 'objective')
        gr_time_mean, _ = compute_stats(greedy_runs, 'solve_time')

        # 计算 Greedy gap
        greedy_gap = (gr_obj_mean - pb_obj_mean) / pb_obj_mean * 100

        print(f"{ntasks:4d} & ${pb_obj_mean:.0f} \\pm {pb_obj_std:.0f}$ & ${gr_obj_mean:.0f} \\pm {gr_obj_std:.0f}$ & "
              f"{pb_time_mean:.1f} & {gr_time_mean:.1f} & +{greedy_gap:.1f}\\% & {pb_gap:.4f}\\% \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")
    print()

def generate_alns_table(data, pb_data):
    """生成 ALNS 对比表: P&B vs GRASP vs ALNS"""
    print("% === ALNS benchmark table @ M=600 ===")
    print("\\begin{table}[!htbp]")
    print("\\ExpSmallTableStyle")
    print("\\caption{Matched three-algorithm comparison at $M=600$ (10 seeds per scale). The objective column reports 10-seed mean $\\pm$ standard deviation; the heuristic gap is the mean relative excess over the P\\&B objective.}")
    print("\\label{tab:alns-m600}")
    print("\\begin{center}")
    print("\\begin{tabular}{@{}lrrrrr@{}}")
    print("\\toprule")
    print(" & \\multicolumn{3}{c}{Objective} & \\\\")
    print("\\cmidrule(lr){2-4}")
    print("Tasks & \\multicolumn{1}{c}{P\\&B} & \\multicolumn{1}{c}{GRASP} & \\multicolumn{1}{c}{ALNS} & Heuristic gap & Completed \\\\")
    print("\\midrule")

    for ntasks in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        pb_obj_mean, pb_obj_std = pb_data[ntasks]['obj']
        pb_completed = pb_data[ntasks]['completed']

        grasp_runs = data[ntasks]['grasp']
        alns_runs = data[ntasks]['alns']

        if not (grasp_runs and alns_runs):
            continue

        gr_obj_mean, gr_obj_std = compute_stats(grasp_runs, 'objective')
        al_obj_mean, al_obj_std = compute_stats(alns_runs, 'objective')

        grasp_gap = (gr_obj_mean - pb_obj_mean) / pb_obj_mean * 100
        alns_gap = (al_obj_mean - pb_obj_mean) / pb_obj_mean * 100

        print(f"{ntasks:4d} & ${pb_obj_mean:.0f} \\pm {pb_obj_std:.0f}$ & "
              f"${gr_obj_mean:.0f} \\pm {gr_obj_std:.0f}$ & "
              f"${al_obj_mean:.0f} \\pm {al_obj_std:.0f}$ & "
              f"+{grasp_gap:.1f}\\% / +{alns_gap:.1f}\\% & ${pb_completed:.1f} \\pm {pb_data[ntasks]['completed_std']:.1f}$ \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")

if __name__ == "__main__":
    # 加载 M=600 重跑数据
    data = load_rerun_data()

    # P&B 数据占位符（需要从实际 P&B @ M=600 运行中获取）
    # 如果你还没有 P&B @ M=600 的数据，这里先用占位符
    pb_placeholder = {
        ntasks: {
            'obj': (0, 0),  # 需要替换
            'time': (0, 0),
            'gap': 0.0000,
            'completed': 0,
            'completed_std': 0
        }
        for ntasks in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    }

    print("% ========================================")
    print("% 自动生成于 M=600 重跑数据")
    print("% ========================================")
    print()

    generate_main_table(data, pb_placeholder)
    generate_alns_table(data, pb_placeholder)

    print("% ========================================")
    print("% 注意: P&B 数据使用占位符,需要手动替换")
    print("% ========================================")
