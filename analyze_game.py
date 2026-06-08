"""Script để phân tích chi tiết game state."""

from kaggle_environments import make
import json

def analyze_game():
    """Chạy game và hiển thị chi tiết mỗi step."""

    print("🎮 Khởi tạo game Crawl...")
    env = make("crawl", configuration={"randomSeed": 42}, debug=True)

    # Chỉ lấy vài bước đầu tiên
    env.run(["main.py", "random"])

    # Hiển thị thông tin config
    config = env.configuration
    print("\n" + "="*60)
    print("⚙️  CẤU HÌNH GAME")
    print("="*60)
    print(f"Width (rộng): {config.width} ô")
    print(f"Height (cao): {config.height} ô")

    # Hiển thị chi tiết config
    print("\nCác chi phí xây dựng:")
    print(f"  scoutCost: {config.scoutCost} năng lượng")
    print(f"  workerCost: {config.workerCost} năng lượng")
    print(f"  minerCost: {config.minerCost} năng lượng")
    print(f"  wallRemoveCost: {config.wallRemoveCost} năng lượng")

    # Hiển thị vài bước đầu
    print("\n" + "="*60)
    print("📋 CHI TIẾT MỘT SỐ BƯỚC ĐẦU")
    print("="*60)

    for step_idx in range(min(5, len(env.steps))):
        step = env.steps[step_idx]
        print(f"\n--- BƯỚC {step_idx + 1} ---")

        for player_idx, state in enumerate(step):
            if state.observation is None:
                continue

            obs = state.observation
            print(f"\n👤 Player {player_idx}:")
            print(f"   Robots của mình: {len(obs['robots'])} cái")

            for uid, robot_data in obs["robots"].items():
                rtype, col, row, energy, owner, *_ = robot_data
                type_name = {0: "Factory", 1: "Scout", 2: "Worker", 3: "Miner"}[rtype]
                if owner == player_idx:
                    print(f"   - {type_name} tại ({col}, {row}) - Energy: {energy}")

            # Hiển thị mines
            if obs["mines"]:
                print(f"   Mines nhớ được: {len(obs['mines'])} cái")

            # Hiển thị miningNodes
            if obs["miningNodes"]:
                print(f"   Mining nodes nhìn thấy: {len(obs['miningNodes'])} cái")

if __name__ == "__main__":
    analyze_game()
    print("\n✨ Phân tích hoàn tất!")
