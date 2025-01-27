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

def calculate_segment_curvature(points, num_points=50):
    """计算一段路径的曲率
    返回：平均曲率，所有采样点的曲率列表
    """
    if len(points) < 3:
        return 0.0, []
    
    # 使用更多的点进行插值
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, num_points)
    x = np.interp(t_new, t, points[:, 0])
    y = np.interp(t_new, t, points[:, 1])
    
    # 使用更大的窗口计算导数
    dx = np.gradient(x, t_new)
    dy = np.gradient(y, t_new)
    
    # 计算二阶导数
    ddx = np.gradient(dx, t_new)
    ddy = np.gradient(dy, t_new)
    
    # 计算曲率：k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    
    # 添加数值稳定性
    mask = denominator > 1e-6
    curvatures = np.zeros_like(x)
    curvatures[mask] = numerator[mask] / denominator[mask]
    
    # 过滤掉异常值
    curvatures = np.clip(curvatures, 0, 5.0)  # 限制最大曲率为5
    
    # 打印调试信息
    valid_curvatures = curvatures[curvatures > 0]
    if len(valid_curvatures) > 0:
        print(f"Debug - Segment curvatures: min={np.min(valid_curvatures):.4f}, max={np.max(valid_curvatures):.4f}, mean={np.mean(valid_curvatures):.4f}")
    
    return np.mean(curvatures), curvatures

def calculate_path_curvature(points, num_points=50):
    """计算路径的曲率相关指标
    返回：总曲率，平均曲率，曲率范围，每段的曲率信息
    """
    if len(points) < 3:
        return 0.0, 0.0, 0.0, []
    
    # 计算每一段的曲率
    segment_curvatures = []
    all_curvatures = []
    total_curvature = 0
    
    # 直接使用所有点计算曲率
    avg_curvature, curvatures = calculate_segment_curvature(points, num_points)
    if not np.isnan(avg_curvature):  # 确保曲率值有效
        segment_curvatures.append(avg_curvature)
        all_curvatures.extend(curvatures)
        total_curvature = avg_curvature
    
    # 计算总体指标
    if all_curvatures:
        mean_curvature = np.mean(all_curvatures)
        valid_curvatures = np.array(all_curvatures)[np.array(all_curvatures) > 0]
        if len(valid_curvatures) > 0:
            curvature_range = np.max(valid_curvatures) - np.min(valid_curvatures)
        else:
            curvature_range = 0
        # 打印调试信息
        print(f"Debug - Path curvature: total={total_curvature:.4f}, mean={mean_curvature:.4f}, range={curvature_range:.4f}")
    else:
        mean_curvature = 0
        curvature_range = 0
    
    return total_curvature, mean_curvature, curvature_range, segment_curvatures

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

def check_line_segment_collision(p1, p2, rect_center, rect_length, rect_width, rect_angle_deg, num_checks=20):
    """检查线段是否与旋转矩形碰撞
    使用更密集的采样点进行检查
    """
    # 在线段上均匀采样点进行检测
    t = np.linspace(0, 1, num_checks)
    for ti in t:
        point = p1 * (1-ti) + p2 * ti
        if check_point_in_rotated_rectangle(point, rect_center, rect_length, rect_width, rect_angle_deg):
            return True
    return False

def check_spline_collision(x_curve, y_curve, rect_center, rect_length, rect_width, rect_angle_deg, check_points=50):
    """检查样条曲线是否与旋转矩形碰撞
    1. 使用更多的检查点
    2. 检查相邻点之间的线段
    """
    points = np.column_stack((x_curve, y_curve))
    
    # 检查所有点
    for point in points:
        if check_point_in_rotated_rectangle(point, rect_center, rect_length, rect_width, rect_angle_deg):
            return True
            
    # 检查相邻点之间的线段
    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        if check_line_segment_collision(p1, p2, rect_center, rect_length, rect_width, rect_angle_deg):
            return True
            
    return False

def calculate_distance_to_reference(point, ref_points):
    """计算点到参考线的最小距离"""
    # 计算点到所有参考线段的距离
    min_dist = float('inf')
    
    for i in range(len(ref_points) - 1):
        p1 = ref_points[i]
        p2 = ref_points[i + 1]
        
        # 计算点到线段的距离
        segment = p2 - p1
        segment_length = np.linalg.norm(segment)
        if segment_length == 0:
            continue
            
        # 计算投影点的参数 t
        t = np.dot(point - p1, segment) / (segment_length * segment_length)
        
        # 如果投影点在线段上
        if 0 <= t <= 1:
            # 计算投影点
            projection = p1 + t * segment
            dist = np.linalg.norm(point - projection)
        else:
            # 如果投影点不在线段上，计算到端点的距离
            dist = min(np.linalg.norm(point - p1), np.linalg.norm(point - p2))
        
        min_dist = min(min_dist, dist)
    
    return min_dist

def calculate_path_deviation(points, ref_points):
    """计算路径点到参考线的平均偏离距离"""
    total_deviation = 0
    for point in points:
        total_deviation += calculate_distance_to_reference(point, ref_points)
    return total_deviation / len(points)

def calculate_path_length(points):
    """计算路径长度"""
    length = 0
    for i in range(len(points)-1):
        length += np.linalg.norm(points[i+1] - points[i])
    return length

def find_min_curvature_path(graph_data, start_layer=0, start_node=0):
    """使用Dijkstra算法找到最优路径"""
    # 初始化
    pq = PriorityQueue()
    visited = set()
    costs = {}
    previous = {}
    curvatures = {}  # 存储每条边的曲率信息
    num_layers = len(graph_data['layers'])
    
    # 参考线点
    ref_points = np.array(graph_data['reference_line']['points'])
    
    # 代价权重
    total_curvature_weight = 1.0    # 总曲率权重
    mean_curvature_weight = 0.5     # 平均曲率权重
    curvature_range_weight = 0.3    # 曲率范围权重
    deviation_weight = 0.5          # 参考线偏离权重
    length_weight = 0.01           # 路径长度权重
    
    # 障碍物参数
    obstacle_center = [10, 1]
    obstacle_length = 3.8
    obstacle_width = 1.8
    obstacle_angle = -45
    
    # 曲率计算参数
    num_curvature_points = 50  # 计算曲率时的采样点数
    
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
                
                # 计算曲率相关代价
                total_curvature, mean_curvature, curvature_range, segment_curvatures = calculate_path_curvature(
                    curve_points, num_curvature_points)
                
                curvature_cost = (total_curvature_weight * total_curvature + 
                                mean_curvature_weight * mean_curvature +
                                curvature_range_weight * curvature_range)
                
                # 存储曲率信息
                edge_id = (current_state, next_state)
                curvatures[edge_id] = {
                    'total': total_curvature,
                    'mean': mean_curvature,
                    'range': curvature_range,
                    'segments': segment_curvatures
                }
                
                # 计算参考线偏离代价
                deviation_cost = calculate_path_deviation(curve_points, ref_points)
                
                # 计算路径长度代价
                length_cost = calculate_path_length(curve_points)
                
                # 计算总代价
                edge_cost = (curvature_cost + 
                           deviation_weight * deviation_cost +
                           length_weight * length_cost)
                new_cost = costs[current_state] + edge_cost
                
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
    
    # 打印路径的曲率信息
    path = path[::-1]
    print("\n路径曲率信息:")
    for i in range(len(path)-1):
        edge_id = (path[i], path[i+1])
        if edge_id in curvatures:
            info = curvatures[edge_id]
            print(f"路径段 {i}:")
            print(f"  总曲率 = {info['total']:.4f}")
            print(f"  平均曲率 = {info['mean']:.4f}")
            print(f"  曲率范围 = {info['range']:.4f}")
    
    return path

def create_spline_with_heading(start_pos, end_pos, start_heading, end_heading, num_points=50):
    """创建考虑朝向约束的三次样条曲线"""
    # 计算控制点的距离（可以调整这个值来控制曲线形状）
    control_distance = 2.0
    
    # 计算控制点
    start_control = start_pos + control_distance * np.array([np.cos(start_heading), np.sin(start_heading)])
    end_control = end_pos - control_distance * np.array([np.cos(end_heading), np.sin(end_heading)])
    
    # 创建参数点
    t = np.linspace(0, 1, num_points)
    
    # 计算三次贝塞尔曲线的点
    x = (1-t)**3 * start_pos[0] + \
        3*(1-t)**2 * t * start_control[0] + \
        3*(1-t) * t**2 * end_control[0] + \
        t**3 * end_pos[0]
        
    y = (1-t)**3 * start_pos[1] + \
        3*(1-t)**2 * t * start_control[1] + \
        3*(1-t) * t**2 * end_control[1] + \
        t**3 * end_pos[1]
    
    return x, y

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
    
    # 绘制所有节点和朝向
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
    
    # 绘制所有边
    total_curvature = 0
    for edge in graph_data['edges']:
        start_layer = edge['start']['layer']
        start_node = edge['start']['node']
        end_layer = edge['end']['layer']
        end_node = edge['end']['node']
        
        start_pos = get_node_position(graph_data, start_layer, start_node)
        end_pos = get_node_position(graph_data, end_layer, end_node)
        
        if start_pos and end_pos:
            start_heading = get_node_heading(graph_data, start_layer, start_node)
            end_heading = get_node_heading(graph_data, end_layer, end_node)
            
            if start_heading is not None and end_heading is not None:
                # 生成样条曲线
                x_curve, y_curve = create_spline_with_heading(
                    np.array(start_pos), np.array(end_pos), 
                    start_heading, end_heading)
                
                # 检查碰撞
                if not check_spline_collision(x_curve, y_curve, 
                                           [10, 1], 3.8, 1.8, -45):
                    # 只绘制没有碰撞的边
                    plt.plot(x_curve, y_curve, 'b-', alpha=0.2, linewidth=1)
    
    # 绘制最优路径
    if path:
        for i in range(len(path)-1):
            current_layer, current_node = path[i]
            next_layer, next_node = path[i+1]
            
            start_pos = get_node_position(graph_data, current_layer, current_node)
            end_pos = get_node_position(graph_data, next_layer, next_node)
            start_heading = get_node_heading(graph_data, current_layer, current_node)
            end_heading = get_node_heading(graph_data, next_layer, next_node)
            
            if start_pos and end_pos and start_heading is not None and end_heading is not None:
                x_curve, y_curve = create_spline_with_heading(
                    np.array(start_pos), np.array(end_pos), 
                    start_heading, end_heading)
                curve_points = np.column_stack((x_curve, y_curve))
                total_curvature += calculate_path_curvature(curve_points)[0]
                plt.plot(x_curve, y_curve, 'r-', linewidth=2, label='Optimal Path' if i == 0 else "")
                
                # 高亮显示路径上的节点
                plt.plot(start_pos[0], start_pos[1], 'ro', markersize=6)
                if i == len(path)-2:  # 最后一个节点
                    plt.plot(end_pos[0], end_pos[1], 'ro', markersize=6)
    
    # 绘制障碍物
    rect_center = [10, 1]
    rect_length = 3.8
    rect_width = 1.8
    rect_angle = -45
    vertices = create_rotated_rectangle(rect_center[0], rect_center[1], 
                                      rect_length, rect_width, rect_angle)
    vertices = np.array(vertices)
    plt.fill(vertices[:, 0], vertices[:, 1], 'gray', alpha=0.5, label='Obstacle')
    
    plt.title(f'Graph Visualization with Min Curvature Path (Total Curvature: {total_curvature:.2f})')
    plt.axis('equal')
    plt.grid(True)
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
