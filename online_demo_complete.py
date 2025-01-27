import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from queue import PriorityQueue

"""
这是在线轨迹规划的demo，使用预先生成的图数据进行路径规划。
该demo将：
1. 读取预生成的图数据（来自graph_output文件夹）
2. 在图上进行最小曲率路径搜索
3. 使用考虑朝向约束的三次样条曲线连接节点并可视化结果
"""

def load_graph_data(file_path):
    """加载图数据"""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_node_position(graph_data, layer, node):
    """获取指定层和节点的位置"""
    for layer_data in graph_data['layers']:
        if layer_data['layer_index'] == layer:
            return layer_data['nodes'][node]
    return None

def get_node_heading(graph_data, layer, node):
    """获取指定层和节点的朝向"""
    for layer_data in graph_data['layers']:
        if layer_data['layer_index'] == layer:
            return layer_data['headings'][node]
    return None

def calculate_path_curvature(points):
    """计算路径的总曲率"""
    if len(points) < 3:
        return 0
    
    total_curvature = 0
    for i in range(1, len(points)-1):
        # 计算相邻三点形成的角度
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        
        # 计算角度变化（曲率的近似）
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 避免数值误差
        theta = np.arccos(cos_theta)
        
        total_curvature += abs(theta)
    
    return total_curvature

def check_point_in_rotated_rectangle(point, rect_center, rect_length, rect_width, rect_angle_deg):
    """检查点是否在旋转矩形内"""
    # 将角度转换为弧度
    angle_rad = np.deg2rad(rect_angle_deg)
    
    # 创建旋转矩阵的逆矩阵（用于将点转换回未旋转的坐标系）
    rot_matrix_inv = np.array([
        [np.cos(angle_rad), np.sin(angle_rad)],
        [-np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # 将点平移到以矩形中心为原点的坐标系
    translated_point = np.array(point) - np.array(rect_center)
    
    # 将点旋转回未旋转的坐标系
    rotated_point = translated_point @ rot_matrix_inv
    
    # 在未旋转的坐标系中检查点是否在矩形内
    half_length = rect_length / 2
    half_width = rect_width / 2
    
    return (abs(rotated_point[0]) <= half_length and 
            abs(rotated_point[1]) <= half_width)

def check_spline_collision(x_curve, y_curve, rect_center, rect_length, rect_width, rect_angle_deg, check_points=20):
    """检查样条曲线是否与旋转矩形碰撞"""
    # 在样条曲线上均匀采样点进行碰撞检测
    for i in range(check_points):
        point = [x_curve[i], y_curve[i]]
        if check_point_in_rotated_rectangle(point, rect_center, rect_length, rect_width, rect_angle_deg):
            return True  # 发生碰撞
    return False  # 无碰撞

def find_min_curvature_path(graph_data, start_layer=0, start_node=0):
    """使用Dijkstra算法找到总曲率最小的无碰撞路径"""
    # 初始化
    pq = PriorityQueue()
    visited = set()
    costs = {}
    previous = {}
    num_layers = len(graph_data['layers'])
    
    # 障碍物参数
    obstacle_center = [10, 1]
    obstacle_length = 3.8
    obstacle_width = 1.8
    obstacle_angle = -45
    
    # 起始节点
    start_state = (start_layer, start_node)
    pq.put((0, start_state))
    costs[start_state] = 0
    
    while not pq.empty():
        current_cost, current_state = pq.get()
        current_layer, current_node = current_state
        
        if current_state in visited:
            continue
            
        visited.add(current_state)
        
        # 如果到达最后一层，结束搜索
        if current_layer == num_layers - 1:
            break
        
        # 检查所有可能的下一个节点
        for edge in graph_data['edges']:
            if edge['start']['layer'] == current_layer and edge['start']['node'] == current_node:
                next_layer = edge['end']['layer']
                next_node = edge['end']['node']
                next_state = (next_layer, next_node)
                
                if next_state in visited:
                    continue
                
                # 计算这条边的曲率代价
                current_pos = np.array(get_node_position(graph_data, current_layer, current_node))
                next_pos = np.array(get_node_position(graph_data, next_layer, next_node))
                current_heading = get_node_heading(graph_data, current_layer, current_node)
                next_heading = get_node_heading(graph_data, next_layer, next_node)
                
                # 生成样条曲线点
                x_curve, y_curve = create_spline_with_heading(
                    current_pos, next_pos, current_heading, next_heading)
                
                # 检查碰撞
                if check_spline_collision(x_curve, y_curve, 
                                       obstacle_center, obstacle_length, 
                                       obstacle_width, obstacle_angle):
                    continue  # 如果发生碰撞，跳过这条边
                
                curve_points = np.column_stack((x_curve, y_curve))
                
                # 计算曲率
                edge_curvature = calculate_path_curvature(curve_points)
                new_cost = costs[current_state] + edge_curvature
                
                if next_state not in costs or new_cost < costs[next_state]:
                    costs[next_state] = new_cost
                    previous[next_state] = current_state
                    pq.put((new_cost, next_state))
    
    # 重建路径
    path = []
    end_nodes = [node for layer, node in visited if layer == num_layers-1]
    if not end_nodes:  # 如果没有找到可行路径
        return None
        
    current_state = (num_layers-1, min(
        end_nodes,
        key=lambda n: costs.get((num_layers-1, n), float('inf'))
    ))
    
    while current_state in previous:
        path.append(current_state)
        current_state = previous[current_state]
    path.append(start_state)
    
    return path[::-1]

def create_spline_with_heading(start_pos, end_pos, start_heading, end_heading, num_points=20):
    """创建考虑朝向约束的三次样条曲线"""
    # 根据朝向创建控制点
    heading_scale = np.linalg.norm(np.array(end_pos) - np.array(start_pos)) / 3
    
    # 使用朝向创建切向量
    start_tangent = np.array([np.cos(start_heading), np.sin(start_heading)]) * heading_scale
    end_tangent = np.array([np.cos(end_heading), np.sin(end_heading)]) * heading_scale
    
    # 创建控制点
    p1 = np.array(start_pos)
    p2 = p1 + start_tangent
    p3 = np.array(end_pos) - end_tangent
    p4 = np.array(end_pos)
    
    # 使用参数化的点
    t = np.linspace(0, 1, num_points)
    
    # 计算贝塞尔曲线点
    points = np.zeros((num_points, 2))
    for i, ti in enumerate(t):
        points[i] = (1-ti)**3 * p1 + \
                   3*(1-ti)**2*ti * p2 + \
                   3*(1-ti)*ti**2 * p3 + \
                   ti**3 * p4
    
    return points[:, 0], points[:, 1]

def create_rotated_rectangle(center_x, center_y, length, width, angle_deg):
    """创建旋转矩形的顶点"""
    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle_deg)
    
    # 创建旋转矩阵
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # 未旋转时的矩形顶点（相对于中心点）
    half_length = length / 2
    half_width = width / 2
    points = np.array([
        [-half_length, -half_width],  # 左下
        [half_length, -half_width],   # 右下
        [half_length, half_width],    # 右上
        [-half_length, half_width]    # 左上
    ])
    
    # 应用旋转
    rotated_points = points @ rot_matrix.T
    
    # 移动到指定中心位置
    rotated_points += np.array([center_x, center_y])
    
    return rotated_points

def visualize_graph_and_path(graph_data, path=None):
    """可视化图和路径"""
    plt.figure(figsize=(12, 8))
    
    # 绘制参考线
    ref_points = np.array(graph_data['reference_line']['points'])
    plt.plot(ref_points[:, 0], ref_points[:, 1], 'k--', alpha=0.5, label='Reference Line')
    
    # 绘制所有节点
    for layer in graph_data['layers']:
        layer_idx = layer['layer_index']
        for node_idx, node_pos in enumerate(layer['nodes']):
            x, y = node_pos
            plt.plot(x, y, 'bo', markersize=4)
            
            # 绘制朝向箭头
            heading = layer['headings'][node_idx]
            arrow_length = 0.5
            dx = arrow_length * np.cos(heading)
            dy = arrow_length * np.sin(heading)
            plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='b', ec='b', alpha=0.5)
    
    # 绘制所有边（使用考虑朝向的样条曲线）
    for edge in graph_data['edges']:
        start_pos = get_node_position(graph_data, edge['start']['layer'], edge['start']['node'])
        end_pos = get_node_position(graph_data, edge['end']['layer'], edge['end']['node'])
        start_heading = get_node_heading(graph_data, edge['start']['layer'], edge['start']['node'])
        end_heading = get_node_heading(graph_data, edge['end']['layer'], edge['end']['node'])
        
        if start_pos and end_pos and start_heading is not None and end_heading is not None:
            x_curve, y_curve = create_spline_with_heading(start_pos, end_pos, start_heading, end_heading)
            plt.plot(x_curve, y_curve, 'gray', alpha=0.3)
    
    # 绘制障碍物（旋转矩形）
    obstacle_points = create_rotated_rectangle(10, 1, 3.8, 1.8, -45)
    # 闭合多边形
    obstacle_points = np.vstack([obstacle_points, obstacle_points[0]])
    plt.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'r-', linewidth=2, label='Obstacle')
    plt.fill(obstacle_points[:, 0], obstacle_points[:, 1], 'r', alpha=0.2)
    
    # 如果有路径，用红色高亮显示（使用考虑朝向的样条曲线）
    if path:
        total_curvature = 0
        for i in range(len(path)-1):
            start_pos = get_node_position(graph_data, path[i][0], path[i][1])
            end_pos = get_node_position(graph_data, path[i+1][0], path[i+1][1])
            start_heading = get_node_heading(graph_data, path[i][0], path[i][1])
            end_heading = get_node_heading(graph_data, path[i+1][0], path[i+1][1])
            
            if start_pos and end_pos and start_heading is not None and end_heading is not None:
                x_curve, y_curve = create_spline_with_heading(start_pos, end_pos, start_heading, end_heading)
                curve_points = np.column_stack((x_curve, y_curve))
                total_curvature += calculate_path_curvature(curve_points)
                plt.plot(x_curve, y_curve, 'r-', linewidth=2)
        
        plt.title(f'Graph Visualization with Min Curvature Path (Total Curvature: {total_curvature:.2f})')
    else:
        plt.title('Graph Visualization')
    
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

def main():
    # 加载图数据
    graph_data = load_graph_data('graph_output/graph_data.json')
    
    # 找到总曲率最小的路径（从第一层的第二个节点开始）
    start_layer = 0
    start_node = 3  # 从右往左数第二个节点
    path = find_min_curvature_path(graph_data, start_layer, start_node)
    
    if path is None:
        print("未找到无碰撞路径！")
        # 仍然可视化图和障碍物
        visualize_graph_and_path(graph_data)
    else:
        print(f"找到的最小曲率无碰撞路径: {path}")
        # 可视化结果
        visualize_graph_and_path(graph_data, path)

if __name__ == "__main__":
    main()
