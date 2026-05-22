import os, json, pygame
import matplotlib.pyplot as plt
from simulation import evaluate
try:
    from PSO import run_pso
    from GA import run_ga
    from graphs import plot_pso_comparison, plot_ga_comparison, plot_optimizer_comparison_runs,plot_final_comparison
except ImportError:
    pass

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
    
def save_results(method_name, timings, score, history=None, extra=None):
    filename = os.path.join("results", f"{method_name.lower()}.json")

    data = {
        "method": method_name,
        "timings": timings,
        "score": float(score)
    }

    # 🔥 add extra metrics (low/medium/high etc.)
    if extra is not None:
        data.update(extra)

    if history is not None:
        data["history"] = history

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅  {method_name} results saved → {filename}")


def robust_evaluate(green_times, sim_duration=180, headless=True, traffic_mode="medium"):
    low = evaluate(green_times, sim_duration, headless, "low")["fitness"]
    medium = evaluate(green_times, sim_duration, headless, "medium")["fitness"]
    high = evaluate(green_times, sim_duration, headless, "high")["fitness"]

    return (low + medium + high) / 3

def run_and_save_optimizer(
    name,
    run_fn,
    evaluate_fn,
    steps,
    population_size,
    sim_time,
    mode="pso",   # 👈 ADD THIS
    bounds=None
):
    print(f"\n▶  Running {name} …")

    if mode == "pso":
        times, score, history = run_fn(
            evaluate_fn,
            iterations=steps,
            swarm_size=population_size,
            sim_time=sim_time
        )

    elif mode == "ga":
        times, score, history = run_fn(
            evaluate_fn,
            bounds,  # 
            generations=steps,
            population_size=population_size,
            sim_time=sim_time
        )

    low_res = evaluate(times, 180, True, "low")
    med_res = evaluate(times, 180, True, "medium")
    high_res = evaluate(times, 180, True, "high")

    save_results(
        name,
        times,
        score,
        history,
        extra={
            "low": low_res,
            "medium": med_res,
            "high": high_res
        }
    )

    return times, score, history

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    pygame.init()

    print("=" * 55)
    print("  TRAFFIC OPTIMISATION SYSTEM  (PSO + GA, tick-based)")
    print("=" * 55)

    baseline_times = [10, 10, 10, 10]
    baseline = os.path.join(RESULTS_DIR, "baseline.json")

    # ── BASELINE ──────────────────────────────────────────────────────
    if os.path.exists(baseline):
        print("\n📂  Loading baseline …")
        with open(baseline) as f:
            _d = json.load(f)
        baseline_times = _d["timings"]
        baseline_score = _d["fitness"]
        print(f"    Score: {baseline_score:.4f}")
    else:
        print("\n▶  Running baseline (headless, 180 s) …")
        print("\n▶  Running baseline (headless, 180 s each traffic mode) …")

        low_res = evaluate(baseline_times, sim_duration=180, headless=True, traffic_mode="low")
        med_res = evaluate(baseline_times, sim_duration=180, headless=True, traffic_mode="medium")
        high_res = evaluate(baseline_times, sim_duration=180, headless=True, traffic_mode="high")

        baseline_score = (
            low_res["fitness"] +
            med_res["fitness"] +
            high_res["fitness"]
        ) / 3

        print(f"    Score: {baseline_score:.4f}")

        with open(baseline, "w") as f:
            json.dump({
                "timings": baseline_times,
                "fitness": baseline_score,

                "low": low_res,
                "medium": med_res,
                "high": high_res
            }, f, indent=4)
        print("✅  baseline.json saved")

    print("\n▶  Visualising baseline (60 s) …")
    evaluate(baseline_times, sim_duration=60, headless=False, traffic_mode="low")
    evaluate(baseline_times, sim_duration=60, headless=False, traffic_mode="medium")
    evaluate(baseline_times, sim_duration=60, headless=False, traffic_mode="high")

    # ── PSO ───────────────────────────────────────────────────────────
    RUN_PSO = False

    if RUN_PSO:
        print("\n▶  Running PSO …")
        pso_small_times, pso_small_score, pso_small_history = run_and_save_optimizer(
            "PSO_SMALL",
            run_pso,
            robust_evaluate,
            steps=8,
            population_size=6,
            sim_time=200,
            mode="pso"
        )

        pso_big_times, pso_big_score, pso_big_history = run_and_save_optimizer(
            "PSO_BIG",
            run_pso,
            robust_evaluate,
            steps=15,
            population_size=12,
            sim_time=300,
            mode="pso"
        )
    else:
        print("\n📂  Loading PSO results …")
        pso_big = os.path.join(RESULTS_DIR, "pso_big.json")
        with open(pso_big) as f:
            _d = json.load(f)
        pso_big_history = _d.get("history", [])

        pso_small = os.path.join(RESULTS_DIR, "pso_small.json")
        with open(pso_small) as f:
            _d = json.load(f)
        pso_small_times   = _d["timings"]
        pso_small_history = _d.get("history", [])

    print("\n▶ Visualising PSO BIG …")
    evaluate(pso_small_times, sim_duration=60, headless=False, traffic_mode="medium")
    plot_optimizer_comparison_runs(
        "PSO SMALL", pso_small_history,
        "PSO BIG", pso_big_history
    )
    plot_pso_comparison([baseline_score] * len(pso_small_history), pso_small_history)

    # ── GA ────────────────────────────────────────────────────────────
    RUN_GA = False

    if RUN_GA:
        print("\n▶  Running GA …")

        ga_small_times, ga_small_score, ga_small_history = run_and_save_optimizer(
            "GA_SMALL",
            run_ga,
            robust_evaluate,
            bounds = [(5, 60)] * 4,
            steps=8,
            population_size=6,
            sim_time=200,
            mode="ga"
        )

        ga_big_times, ga_big_score, ga_big_history = run_and_save_optimizer(
            "GA_BIG",
            run_ga,
            robust_evaluate,
            bounds = [(5, 60)] * 4,
            steps=15,
            population_size=12,
            sim_time=300,
            mode="ga"
        )
    else:
        print("\n📂  Loading GA results …")
        ga_big = os.path.join(RESULTS_DIR, "ga_big.json")
        with open(ga_big) as f:
            _d = json.load(f)

        ga_big_times = _d["timings"]
        ga_big_history = _d.get("history", [])

        ga_small = os.path.join(RESULTS_DIR, "ga_small.json")
        with open(ga_small) as f:
            _d = json.load(f)
        ga_small_times = _d["timings"]
        ga_small_history = _d.get("history", [])

    print("\n▶ Visualising GA BIG …")
    evaluate(ga_small_times, sim_duration=60, headless=False, traffic_mode="medium")
    plot_optimizer_comparison_runs(
        "GA SMALL", ga_small_history,
        "GA BIG", ga_big_history
    )
    plot_ga_comparison([baseline_score] * len(ga_small_history),ga_small_history)
    plot_final_comparison(
        baseline_score,
        [h["global_best"] for h in pso_big_history],
        [h["global_best"] for h in ga_big_history]
    )
    pygame.quit()
