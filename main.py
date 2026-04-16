import heapq

class CampusGraph:

    def __init__(self):
        self.graph = {}       # 邻接表：{建筑: {相邻建筑: 距离}}
        self.coords = {}      # 建筑坐标：{建筑: (x, y)}

    # ==================== 基础：添加/删除 建筑 ====================
    def add_building(self, name, x, y):
        """添加一栋建筑"""
        if name in self.graph:
            print(f"建筑 {name} 已存在！")
            return
        self.graph[name] = {}
        self.coords[name] = (x, y)
        print(f"已添加建筑：{name}")

    def delete_building(self, name):
        """删除一栋建筑（同时删除所有关联道路）"""
        if name not in self.graph:
            print(f"建筑 {name} 不存在！")
            return

        # 删除所有与它相连的边
        for neighbor in list(self.graph[name].keys()):
            del self.graph[neighbor][name]

        # 删除自身
        del self.graph[name]
        del self.coords[name]
        print(f"已删除建筑：{name}")

    # ==================== 基础：添加/删除 道路 ====================
    # 曼哈顿距离
    def add_road(self, a, b):
        """在两栋楼之间添加一条道路（双向）"""
        if a not in self.graph or b not in self.graph:
            print("建筑不存在，无法添加道路！")
            return
        x1, y1 = self.coords[a]
        x2, y2 = self.coords[b]
        dist = abs(x1 - x2) + abs(y1 - y2)
        self.graph[a][b] = dist
        self.graph[b][a] = dist
        # print(f"已添加道路：{a} ↔ {b}")

    def delete_road(self, a, b):
        """删除两栋楼之间的道路（双向）"""
        if a not in self.graph or b not in self.graph:
            print("建筑不存在！")
            return
        if b in self.graph[a]:
            del self.graph[a][b]
            del self.graph[b][a]
            print(f"已删除道路：{a} ↔ {b}")
        else:
            print(f"不存在道路 {a} ↔ {b}")

    # ==================== 生成最小生成树（原有功能） ====================
    def build_full_connection(self):
        """自动构建连通图（最小生成树）"""
        nodes = list(self.coords.keys())
        if not nodes:
            return
        visited = set()
        visited.add(nodes[0])

        while len(visited) < len(nodes):
            min_edge = None
            min_dist = float('inf')

            for u in visited:
                for v in nodes:
                    if v in visited:
                        continue
                    x1, y1 = self.coords[u]
                    x2, y2 = self.coords[v]
                    dist = abs(x1 - x2) + abs(y1 - y2)
                    if dist < min_dist:
                        min_dist = dist
                        min_edge = (u, v)

            u, v = min_edge
            self.add_road(u, v)
            visited.add(v)
        print("校园连通图构建完成！")

    # ==================== 最短路径：Dijkstra（原有功能 + 修复） ====================
    def dijkstra(self, start, end):
        """查询两点之间最短路径与距离"""
        if start not in self.graph or end not in self.graph:
            return [], float('inf')

        dist = {node: float('inf') for node in self.graph}
        prev = {}
        dist[start] = 0
        pq = [(0, start)]

        while pq:
            d, node = heapq.heappop(pq)
            if node == end:
                break
            if d > dist[node]:
                continue
            for neighbor, weight in self.graph[node].items():
                new_d = d + weight
                if new_d < dist[neighbor]:
                    dist[neighbor] = new_d
                    prev[neighbor] = node
                    heapq.heappush(pq, (new_d, neighbor))

        # 不可达
        if dist[end] == float('inf'):
            return [], float('inf')

        # 还原路径
        path = []
        cur = end
        while cur in prev:
            path.append(cur)
            cur = prev[cur]
        path.append(start)
        path.reverse()
        return path, dist[end]

    # ==================== 遍历：深度优先 DFS ====================
    def dfs(self, start):
        """从起点开始深度优先遍历所有可达建筑"""
        if start not in self.graph:
            print("起点不存在！")
            return []
        visited = []
        def dfs_recur(node):
            visited.append(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    dfs_recur(neighbor)
        dfs_recur(start)
        return visited

    # ==================== 遍历：广度优先 BFS ====================
    def bfs(self, start):
        """从起点开始广度优先遍历所有可达建筑"""
        if start not in self.graph:
            print("起点不存在！")
            return []
        visited = [start]
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.append(neighbor)
                    queue.append(neighbor)
        return visited

    # ==================== 导航输出（原有功能） ====================
    def navigation(self, path):
        print("\n========== 导航路线 ==========")
        total = 0
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            x1, y1 = self.coords[a]
            x2, y2 = self.coords[b]
            dx = x2 - x1
            dy = y2 - y1
            d = abs(dx) + abs(dy)
            total += d

            if x1 == x2:
                direction = "向北" if dy > 0 else "向南"
                print(f"{a} → {direction} {abs(dy)}米 → {b}")
            elif y1 == y2:
                direction = "向东" if dx > 0 else "向西"
                print(f"{a} → {direction} {abs(dx)}米 → {b}")
            else:
                dir_y = "向北" if dy > 0 else "向南"
                print(f"{a} → {dir_y} {abs(dy)}米")
                dir_x = "向东" if dx > 0 else "向西"
                print(f"然后 → {dir_x} {abs(dx)}米 → {b}")
        print(f"\n总距离：{int(total)} 米")

    # ==================== 工具：查看所有建筑/道路 ====================
    def show_all_buildings(self):
        """显示所有建筑"""
        print("\n========== 所有建筑 ==========")
        for b in self.graph.keys():
            print(f"- {b}")

    def show_all_roads(self):
        """显示所有道路"""
        print("\n========== 所有道路 ==========")
        printed = set()
        for a in self.graph:
            for b in self.graph[a]:
                if (b, a) not in printed:
                    print(f"{a} ↔ {b} （距离：{self.graph[a][b]}米）")
                    printed.add((a, b))

# ==================== 主程序：菜单交互 ====================
if __name__ == "__main__":

    # 初始化校园
    campus = CampusGraph()

    buildings = {
        "主楼": (285, 325),
        "教一楼": (235, 325),
        "教二楼": (235, 220),
        "教三楼": (135, 220),
        "教四楼": (135, 325),
        "科学会堂": (325, 270),
        "创新楼": (325, 220),
        "小松林": (215, 455),
        "时光广场": (215, 395),
        "行政办公楼": (235, 375),
        "图书馆": (285, 425),
        "学生发展中心": (285, 450),
        "学生活动中心": (250, 580),
        "经管楼": (250, 550),
        "学一公寓": (85, 395),
        "学二公寓": (185, 395),
        "学三公寓": (85, 455),
        "学四公寓": (185, 455),
        "学五公寓": (85, 500),
        "学六公寓": (325, 580),
        "学八公寓": (185, 500),
        "留学生公寓": (85, 550),
        "学九公寓": (85, 565),
        "学十公寓": (185, 580),
        "学十一公寓": (85, 595),
        "学十三公寓": (25, 470),
        "学29公寓": (425, 425),
        "学苑风味餐厅": (325, 450),
        "学生食堂": (355, 450),
        "综合食堂": (155, 550),
        "科研楼": (355, 580),
        "体育馆": (375, 325),
        "游泳馆": (425, 325),
        "体育场": (400, 270),
        "篮球场": (355, 425),
    }

    # 批量添加建筑
    for name, (x, y) in buildings.items():
        campus.add_building(name, x, y)

    # 构建校园连通图
    campus.build_full_connection()

    # 菜单循环
    while True:
        print("\n====== 校园导航系统 ======")
        print("1. 查询两栋楼最短路径")
        print("2. 深度优先遍历（DFS）")
        print("3. 广度优先遍历（BFS）")
        print("4. 添加建筑")
        print("5. 删除建筑")
        print("6. 添加道路")
        print("7. 删除道路")
        print("8. 查看所有建筑")
        print("9. 查看所有道路")
        print("0. 退出")

        choice = input("请输入功能编号：")

        if choice == "1":
            start = input("请输入起点：")
            end = input("请输入终点：")
            path, dist = campus.dijkstra(start, end)
            if not path:
                print("无法到达！")
            else:
                print("\n最短路径：", " → ".join(path))
                campus.navigation(path)

        elif choice == "2":
            start = input("请输入遍历起点：")
            res = campus.dfs(start)
            print("DFS 遍历顺序：", " → ".join(res))

        elif choice == "3":
            start = input("请输入遍历起点：")
            res = campus.bfs(start)
            print("BFS 遍历顺序：", " → ".join(res))

        elif choice == "4":
            name = input("建筑名称：")
            x = int(input("x 坐标："))
            y = int(input("y 坐标："))
            campus.add_building(name, x, y)

        elif choice == "5":
            name = input("要删除的建筑名称：")
            campus.delete_building(name)

        elif choice == "6":
            a = input("建筑A：")
            b = input("建筑B：")
            campus.add_road(a, b)

        elif choice == "7":
            a = input("建筑A：")
            b = input("建筑B：")
            campus.delete_road(a, b)

        elif choice == "8":
            campus.show_all_buildings()

        elif choice == "9":
            campus.show_all_roads()

        elif choice == "0":
            print("退出系统！")
            break

        else:
            print("输入无效，请重新输入！")