import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import json
import os

class SimpleGraphBase:
    """简化版的GraphBase类"""
    def __init__(self):
        self.edges = {}  # 存储边的信息
        self.nodes = {}  # 存储节点的信息
        
    def add_edge(self, start_layer, start_node, end_layer, end_node, spline_coeff=None, cost=None):
        edge_key = (start_layer, start_node, end_layer, end_node)
        self.edges[edge_key] = {
            'spline_coeff': spline_coeff,
            'cost': cost
        }
    
    def get_edges(self):
        return list(self.edges.keys())

def create_open_reference_line():
    # 创建一个S形的开环参考线
    t = np.linspace(0, 2*np.pi, 100)
    x = t * 10
    y = 30 * np.sin(t/2)
    return np.column_stack((x, y))

def generate_layer_nodes(refline, layer_indices, track_width):
    """在指定的s位置（layer_indices）生成横向采样点"""
    nodes_all = []
    headings_all = []  # 存储每个点的朝向
    
    # 在每个layer位置生成横向采样点
    for idx in layer_indices:
        # 获取该位置的参考线点
        ref_point = refline[idx]
        
        # 计算该位置的切向量和法向量
        if idx < len(refline) - 1:
            dx = refline[idx + 1, 0] - refline[idx, 0]
            dy = refline[idx + 1, 1] - refline[idx, 1]
        else:
            dx = refline[idx, 0] - refline[idx - 1, 0]
            dy = refline[idx, 1] - refline[idx - 1, 1]
        
        # 计算朝向角
        heading = np.arctan2(dy, dx)
        
        # 计算单位法向量
        norm = np.sqrt(dx*dx + dy*dy)
        normal = np.array([-dy/norm, dx/norm])
        
        # 在横向生成采样点
        lateral_positions = np.linspace(-track_width/2, track_width/2, 5)  # 每个横截面5个点
        layer_nodes = []
        layer_headings = []
        
        for lat_pos in lateral_positions:
            node = ref_point + lat_pos * normal
            layer_nodes.append(node)
            layer_headings.append(heading)
        
        nodes_all.append(np.array(layer_nodes))
        headings_all.append(np.array(layer_headings))
    
    return nodes_all, headings_all

def generate_edges(nodes_all, headings_all, max_lateral_step=2):
    """生成图的边"""
    graph = SimpleGraphBase()
    
    # 遍历所有层（除了最后一层）
    for i in range(len(nodes_all) - 1):
        current_layer = i
        next_layer = i + 1
        
        # 当前层的所有节点
        current_nodes = nodes_all[i]
        next_nodes = nodes_all[i + 1]
        
        # 为每个当前层的节点生成到下一层的边
        for start_idx in range(len(current_nodes)):
            # 允许连接到下一层的节点范围（限制横向跨度）
            min_next_idx = max(0, start_idx - max_lateral_step)
            max_next_idx = min(len(next_nodes), start_idx + max_lateral_step + 1)
            
            # 生成到下一层允许范围内节点的边
            for end_idx in range(min_next_idx, max_next_idx):
                # 获取起点和终点
                start_point = current_nodes[start_idx]
                end_point = next_nodes[end_idx]
                start_heading = headings_all[i][start_idx]
                end_heading = headings_all[i+1][end_idx]
                
                # 使用scipy的CubicSpline，但提供更多控制点
                t_control = np.array([0, 0.3, 0.7, 1.0])
                
                # 计算中间控制点
                dist = np.linalg.norm(end_point - start_point)
                start_tangent = np.array([np.cos(start_heading), np.sin(start_heading)])
                end_tangent = np.array([np.cos(end_heading), np.sin(end_heading)])
                
                p1 = start_point + start_tangent * (dist * 0.3)
                p2 = end_point - end_tangent * (dist * 0.3)
                
                # 构建控制点序列
                x_points = np.array([start_point[0], p1[0], p2[0], end_point[0]])
                y_points = np.array([start_point[1], p1[1], p2[1], end_point[1]])
                
                # 创建样条
                spline_x = CubicSpline(t_control, x_points)
                spline_y = CubicSpline(t_control, y_points)
                
                # 存储样条系数
                graph.add_edge(current_layer, start_idx, next_layer, end_idx, 
                             spline_coeff=[spline_x.c, spline_y.c])
    
    return graph

def calculate_edge_cost(edge, nodes_all, headings_all, graph):
    """计算边的代价"""
    start_layer, start_node, end_layer, end_node = edge
    start_point = nodes_all[start_layer][start_node]
    end_point = nodes_all[end_layer][end_node]
    start_heading = headings_all[start_layer][start_node]
    end_heading = headings_all[end_layer][end_node]
    
    # 计算路径长度代价
    path_length = np.linalg.norm(end_point - start_point)
    
    # 计算横向偏移代价
    lateral_cost = abs(start_node - len(nodes_all[start_layer])//2)
    
    # 计算曲率代价（使用起点和终点的朝向差异作为简化的曲率估计）
    heading_diff = abs(end_heading - start_heading)
    if heading_diff > np.pi:
        heading_diff = 2 * np.pi - heading_diff
    curvature_cost = heading_diff
    
    # 组合代价（可以调整权重）
    total_cost = path_length + 2.0 * lateral_cost + 5.0 * curvature_cost
    
    return float(total_cost)  # 确保返回标量值

def save_graph_data(refline, nodes_all, headings_all, graph, output_dir):
    """保存图数据为JSON格式
    
    Args:
        refline: 参考线点集 numpy.ndarray (N, 2)
        nodes_all: 所有层的节点列表 [numpy.ndarray]
        headings_all: 所有层的航向角列表 [numpy.ndarray]
        graph: 图对象
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建数据字典
    data = {
        "reference_line": {
            "points": refline.tolist(),
        },
        "layers": [],
        "edges": []
    }
    
    # 保存每层的节点和航向角
    for layer_idx, (nodes, headings) in enumerate(zip(nodes_all, headings_all)):
        layer_data = {
            "layer_index": layer_idx,
            "nodes": nodes.tolist(),
            "headings": headings.tolist()
        }
        data["layers"].append(layer_data)
    
    # 保存边和对应的样条系数
    for edge in graph.get_edges():
        start_layer, start_node, end_layer, end_node = edge
        
        # 计算样条系数
        start_point = nodes_all[start_layer][start_node]
        end_point = nodes_all[end_layer][end_node]
        start_heading = headings_all[start_layer][start_node]
        end_heading = headings_all[end_layer][end_node]
        
        # 计算起点和终点之间的距离
        dist = np.linalg.norm(end_point - start_point)
        
        # 计算起点和终点的导数（切向量）
        start_derivative = dist * np.array([np.cos(start_heading), np.sin(start_heading)])
        end_derivative = dist * np.array([np.cos(end_heading), np.sin(end_heading)])
        
        # 创建参数化的三次样条，显式指定边界导数
        t = np.array([0, 1])
        x = np.array([start_point[0], end_point[0]])
        y = np.array([start_point[1], end_point[1]])
        
        # 使用导数约束创建样条
        spline_x = CubicSpline(t, x, bc_type=((1, start_derivative[0]), (1, end_derivative[0])))
        spline_y = CubicSpline(t, y, bc_type=((1, start_derivative[1]), (1, end_derivative[1])))
        
        edge_data = {
            "start": {"layer": int(start_layer), "node": int(start_node)},
            "end": {"layer": int(end_layer), "node": int(end_node)},
            "spline": {
                "x_coeffs": spline_x.c.tolist(),
                "y_coeffs": spline_y.c.tolist()
            },
            "cost": float(graph.edges[edge]['cost'])
        }
        data["edges"].append(edge_data)
    
    # 保存到文件
    output_file = os.path.join(output_dir, "graph_data.json")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Graph data saved to {output_file}")

def main():
    # 1. 创建基础数据
    # 生成参考线（S形）
    t = np.linspace(0, 2*np.pi, 100)
    refline = np.zeros((len(t), 2))
    refline[:, 0] = 3 * t
    refline[:, 1] = 5 * np.sin(t)
    
    # 2. 计算参考线的切向量（用于确定航向角）
    # 使用中心差分来计算更准确的切向量
    dx = np.zeros_like(refline[:, 0])
    dy = np.zeros_like(refline[:, 1])
    
    # 对于内部点，使用中心差分
    dx[1:-1] = (refline[2:, 0] - refline[:-2, 0]) / 2
    dy[1:-1] = (refline[2:, 1] - refline[:-2, 1]) / 2
    
    # 对于端点，使用前向/后向差分
    dx[0] = refline[1, 0] - refline[0, 0]
    dy[0] = refline[1, 1] - refline[0, 1]
    dx[-1] = refline[-1, 0] - refline[-2, 0]
    dy[-1] = refline[-1, 1] - refline[-2, 1]
    
    # 归一化切向量
    norms = np.sqrt(dx**2 + dy**2)
    dx = dx / norms
    dy = dy / norms
    
    headings_ref = np.arctan2(dy, dx)
    
    # 3. 在参考线上均匀选择4个点作为层
    num_layers = 4
    layer_indices = np.linspace(0, len(t)-1, num_layers, dtype=int)
    layer_points = refline[layer_indices]
    layer_headings = headings_ref[layer_indices]
    
    # 4. 在每个层上横向采样5个点
    num_points = 5
    lateral_offsets = np.linspace(-2, 2, num_points)
    
    nodes_all = []
    headings_all = []
    
    for i, (point, heading) in enumerate(zip(layer_points, layer_headings)):
        # 计算横向偏移向量（垂直于参考线方向）
        lateral_vector = np.array([-np.sin(heading), np.cos(heading)])
        
        # 在当前层生成节点
        layer_nodes = []
        layer_headings = []
        for offset in lateral_offsets:
            # 节点位置 = 参考点 + 横向偏移
            node = point + offset * lateral_vector
            layer_nodes.append(node)
            # 所有节点的航向角都与参考线在该处的航向角相同
            layer_headings.append(heading)
        
        nodes_all.append(np.array(layer_nodes))
        headings_all.append(np.array(layer_headings))
    
    # 5. 生成边
    graph = generate_edges(nodes_all, headings_all)
    
    # 6. 计算每条边的代价
    for edge in graph.get_edges():
        cost = calculate_edge_cost(edge, nodes_all, headings_all, graph)
        graph.edges[edge]['cost'] = cost
    
    # 7. 可视化结果
    plt.figure(figsize=(15, 8))
    
    # 绘制参考线
    plt.plot(refline[:, 0], refline[:, 1], 'k--', label='Reference Line')
    
    # 绘制每个层的采样点和航向
    colors = ['r', 'g', 'b', 'y']
    for i, (layer_nodes, layer_headings) in enumerate(zip(nodes_all, headings_all)):
        plt.scatter(layer_nodes[:, 0], layer_nodes[:, 1], color=colors[i], s=50, label=f'Layer {i+1} nodes')
        
        # 绘制航向箭头
        for node, heading in zip(layer_nodes, layer_headings):
            arrow_length = 0.5
            dx = arrow_length * np.cos(heading)
            dy = arrow_length * np.sin(heading)
            plt.arrow(node[0], node[1], dx, dy, head_width=0.1, head_length=0.2, fc=colors[i], ec=colors[i])
    
    # 绘制边（使用代价值来决定颜色深浅）
    costs = [calculate_edge_cost(edge, nodes_all, headings_all, graph) for edge in graph.get_edges()]
    min_cost, max_cost = min(costs), max(costs)
    
    # 为每条边生成更细致的样条曲线点
    t_fine = np.linspace(0, 1, 50)
    
    for i, edge in enumerate(graph.get_edges()):
        start_layer, start_node, end_layer, end_node = edge
        start_point = nodes_all[start_layer][start_node]
        end_point = nodes_all[end_layer][end_node]
        start_heading = headings_all[start_layer][start_node]
        end_heading = headings_all[end_layer][end_node]
        
        # 计算起点和终点之间的距离
        dist = np.linalg.norm(end_point - start_point)
        
        # 计算起点和终点的导数（切向量）
        start_derivative = dist * np.array([np.cos(start_heading), np.sin(start_heading)])
        end_derivative = dist * np.array([np.cos(end_heading), np.sin(end_heading)])
        
        # 创建参数化的三次样条，显式指定边界导数
        t = np.array([0, 1])
        x = np.array([start_point[0], end_point[0]])
        y = np.array([start_point[1], end_point[1]])
        
        # 使用导数约束创建样条
        spline_x = CubicSpline(t, x, bc_type=((1, start_derivative[0]), (1, end_derivative[0])))
        spline_y = CubicSpline(t, y, bc_type=((1, start_derivative[1]), (1, end_derivative[1])))
        
        # 计算样条曲线上的点
        curve_x = spline_x(t_fine)
        curve_y = spline_y(t_fine)
        
        # 根据代价值计算颜色深浅
        cost = costs[i]
        alpha = float(0.3 + 0.7 * (cost - min_cost) / (max_cost - min_cost))
        
        # 绘制样条曲线
        plt.plot(curve_x, curve_y, 'gray', alpha=alpha, linewidth=1)
    
    # 添加起点和终点标记
    plt.plot(refline[0, 0], refline[0, 1], 'go', markersize=15, label='Start')
    plt.plot(refline[-1, 0], refline[-1, 1], 'ro', markersize=15, label='End')
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('4-Layer Offline Planning Demo with Edges and Costs')
    
    # 保存数据
    output_dir = "graph_output"
    save_graph_data(refline, nodes_all, headings_all, graph, output_dir)
    
    plt.show()

if __name__ == "__main__":
    main()
