import heapq

class CampusGraph:

    def __init__(self):
        self.graph = {}
        self.coords = {}

    def add_building(self, name, x, y):
        self.graph[name] = {}
        self.coords[name] = (x, y)

    # 曼哈顿距离
    def add_road(self, a, b):
        x1, y1 = self.coords[a]
        x2, y2 = self.coords[b]

        dist = abs(x1 - x2) + abs(y1 - y2)

        self.graph[a][b] = dist
        self.graph[b][a] = dist

    # 构建连通图
    def build_full_connection(self):
        nodes = list(self.coords.keys())
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

    # Dijkstra
    def dijkstra(self, start, end):
        pq = [(0, start)]
        dist = {node: float('inf') for node in self.graph}
        prev = {}

        dist[start] = 0

        while pq:
            d, node = heapq.heappop(pq)

            if node == end:
                break

            for n, w in self.graph[node].items():
                new_d = d + w

                if new_d < dist[n]:
                    dist[n] = new_d
                    prev[n] = node
                    heapq.heappush(pq, (new_d, n))

        # 不可达判断
        if dist[end] == float('inf'):
            return [], float('inf')

        path = []
        cur = end

        while cur in prev:
            path.append(cur)
            cur = prev[cur]

        path.append(start)
        path.reverse()

        return path, dist[end]

    # 导航输出
    def navigation(self, path):
        print("\n导航路线：")

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
                print(f"{a} -> {direction} {abs(dy)}米 -> {b}")

            elif y1 == y2:
                direction = "向东" if dx > 0 else "向西"
                print(f"{a} -> {direction} {abs(dx)}米 -> {b}")

            else:
                # 先南北
                dir_y = "向北" if dy > 0 else "向南"
                print(f"{a} -> {dir_y} {abs(dy)}米")

                # 再东西
                dir_x = "向东" if dx > 0 else "向西"
                print(f"然后 -> {dir_x} {abs(dx)}米 -> {b}")

        print(f"\n总距离：{int(total)}米")


# 初始化
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

for name, (x, y) in buildings.items():
    campus.add_building(name, x, y)

campus.build_full_connection()

# 用户输入
start = input("请输入起点：")
end = input("请输入终点：")

if start not in campus.graph or end not in campus.graph:
    print("输入错误！可选地点：")
    print(list(campus.graph.keys()))
    exit()

path, dist = campus.dijkstra(start, end)

if not path:
    print("无法到达")
else:
    print("\n最短路径:", " -> ".join(path))
    campus.navigation(path)