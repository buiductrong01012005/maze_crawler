"""So sánh main.py vs improved_agent.py"""

from kaggle_environments import make

def compare_agents():
    """Chạy 2 games: main.py vs main.py, và improved_agent.py vs random"""

    print("🎮 Game 1: main.py vs random")
    print("="*60)
    env1 = make("crawl", configuration={"randomSeed": 42}, debug=False)
    env1.run(["main.py", "random"])

    result1 = env1.steps[-1]
    score1_main = result1[0].reward
    score1_random = result1[1].reward
    print(f"Kết quả: main.py = {score1_main:.1f}, random = {score1_random:.1f}")

    print("\n🎮 Game 2: improved_agent.py vs random")
    print("="*60)
    env2 = make("crawl", configuration={"randomSeed": 42}, debug=False)
    env2.run(["improved_agent.py", "random"])

    result2 = env2.steps[-1]
    score2_improved = result2[0].reward
    score2_random = result2[1].reward
    print(f"Kết quả: improved_agent.py = {score2_improved:.1f}, random = {score2_random:.1f}")

    print("\n📊 TỔNG HỢP")
    print("="*60)
    print(f"main.py vs random: {'THẮNG' if score1_main > score1_random else 'THUA' if score1_main < score1_random else 'HÒA'} ({score1_main:.1f} vs {score1_random:.1f})")
    print(f"improved_agent.py vs random: {'THẮNG' if score2_improved > score2_random else 'THUA' if score2_improved < score2_random else 'HÒA'} ({score2_improved:.1f} vs {score2_random:.1f})")

if __name__ == "__main__":
    compare_agents()
