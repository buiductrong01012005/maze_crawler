"""Improved agent: Scout rush strategy"""

from random import choice


def agent(obs, config):
    """
    Chiến lược:
    1. Xây Scout sớm để khám phá
    2. Tìm mining nodes để xây Miner
    3. Xây Worker để phá tường khi cần
    """
    actions = {}
    width = config.width

    # Lấy robots của mình
    my_robots = {
        uid: data for uid, data in obs.robots.items()
        if data[4] == obs.player
    }

    for uid, data in my_robots.items():
        rtype, col, row, energy, owner, move_cd, jump_cd, build_cd = data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7] if len(data) > 7 else 0

        idx = (row - obs.southBound) * width + col
        w = obs.walls[idx] if 0 <= idx < len(obs.walls) and obs.walls[idx] != -1 else 0

        if rtype == 0:  # Factory
            # Ưu tiên: Scouts (khám phá) > Workers (phá tường) > Miners (khai thác)

            # 1. Nếu có tường phía bắc, nhảy qua nó
            if w & 1 and jump_cd == 0:  # Tường phía bắc và jump ready
                actions[uid] = "JUMP_NORTH"

            # 2. Xây Scouts đầu tiên (rẻ, nhanh, tầm nhìn tốt)
            elif energy >= config.scoutCost and build_cd == 0:
                actions[uid] = "BUILD_SCOUT"

            # 3. Không thể xây, đi bắc
            else:
                actions[uid] = "NORTH"

        elif rtype == 1:  # Scout - khám phá tìm mining nodes
            # Ưu tiên: Bắc > Tây > Đông > Nam
            directions = []
            if not (w & 1): directions.append("NORTH")
            if not (w & 8): directions.append("WEST")
            if not (w & 2): directions.append("EAST")
            if not (w & 4): directions.append("SOUTH")

            actions[uid] = directions[0] if directions else "IDLE"

        elif rtype == 2:  # Worker - phá tường
            # Tìm tường phía bắc và phá
            if (w & 1) and energy >= config.wallRemoveCost:
                actions[uid] = "REMOVE_NORTH"
            else:
                # Đi theo hướng có sẵn
                directions = []
                if not (w & 1): directions.append("NORTH")
                if not (w & 2): directions.append("EAST")
                if not (w & 4): directions.append("SOUTH")
                if not (w & 8): directions.append("WEST")

                actions[uid] = directions[0] if directions else "IDLE"

        elif rtype == 3:  # Miner - khai thác
            # Kiểm tra xem có phải trên mining node không
            mining_key = f"{col},{row}"
            if mining_key in obs.miningNodes:
                actions[uid] = "TRANSFORM"
            else:
                # Tìm mining node gần nhất
                directions = []
                if not (w & 1): directions.append("NORTH")
                if not (w & 2): directions.append("EAST")
                if not (w & 4): directions.append("SOUTH")
                if not (w & 8): directions.append("WEST")

                actions[uid] = directions[0] if directions else "IDLE"

    return actions
