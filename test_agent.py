"""Test script để chơi game Crawl cục bộ và render."""

from kaggle_environments import make
import json

def test_game():
    """Chạy một ván game và hiển thị kết quả."""

    # Tạo environment
    print("🎮 Khởi tạo game Crawl...")
    env = make("crawl", configuration={"randomSeed": 42}, debug=True)

    # Chạy game: agent của bạn vs random agent
    print("⚔️  Chạy game: main.py vs random...")
    env.run(["main.py", "random"])

    # Hiển thị kết quả
    print("\n" + "="*50)
    print("📊 KỂT QUẢ GAME")
    print("="*50)
    final_state = env.steps[-1]

    for i, state in enumerate(final_state):
        player_name = "main.py" if i == 0 else "random"
        print(f"\n🤖 Player {i} ({player_name}):")
        print(f"   Reward: {state.reward}")
        print(f"   Status: {state.status}")

    # Xác định người thắng
    rewards = [state.reward for state in final_state]
    if rewards[0] > rewards[1]:
        print("\n✅ main.py THẮNG!")
    elif rewards[1] > rewards[0]:
        print("\n❌ main.py THUA!")
    else:
        print("\n🤝 HÒA!")

    # Hiển thị số bước
    print(f"\n⏱️  Tổng bước: {len(env.steps)}")

    # Render trong notebook (nếu có)
    try:
        print("\n📺 Rendering game replay...")
        env.render(mode="ipython", width=800, height=800)
        print("✅ Replay đã render!")
    except Exception as e:
        print(f"⚠️  Không thể render (bình thường nếu không phải Jupyter): {e}")

if __name__ == "__main__":
    test_game()
    print("\n✨ Test hoàn tất!")
