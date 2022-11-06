from io import BytesIO
import math
from math import pi
import random
import scipy.stats as ss
import cairo
import json
 
STOP_PROB = 0.3  # Вероятность остановиться
BRANCH_PROB = 0.3  # Вероятность разветвиться
STRAIGHT_ANGLES = (75, 85)  # Угол поворота для продолжения ствола
BRANCH_ANGLES = (30, 60)  # Угол поворота для разветвления
 
H = 128  # Размеры картинки
REAL_IMAGE_COEF = 1
REAL_IMAGE_SIZE = H * REAL_IMAGE_COEF
C = 5  # Паддинг
 
MAX_TREE_HEIGHT_COEF = 0.6 # Def: 0.6
MAX_TREE_HEIGHT = (H - C) * MAX_TREE_HEIGHT_COEF
 
MIN_VERTEXES_NUMBER = 30  # Минимальное количество вершин в дереве
MAX_VERTEXES_NUMBER = 1000 # Максимальное количество вершин в дереве
 
MAX_NUMBER_OF_BRANCH = 4  # Максимальное количество ветвлений на одной ветке
MAX_BRANCH_TRY = 20  # Сколько раз пытаемся сгенерировать ветку пока не сдаемся
MAX_DEPTH_WITHOUT_BRANCH = 1  # Cколько вершин можно пройти не наткнувся на ветку
MIN_TREE_HEIGHT_IN_VERTEXES = 6 # Минимальная высота дерева в вершинах
MAX_TREE_HEIGHT_IN_VERTEXES = 35 # Максимальная высота дерева в вершинах
MAX_SYMMETRY_DIFF = 5
 
STARTING_LENGTH = 15  # Изначальная длина ствола дерева, def: w // 4
LENGTH_COEF = 7
cst = 0.7  # Коэффиент для коэффициента Больше => длиннее ветки
q = 1 - LENGTH_COEF / (cst * (H - C))  # Коэффициент уменьшения длины ветви
 
MIN_BRANCH_LENGTH = H // 10 # Минимальная длина ветви
 
STARTING_RADIUS = 15  # Изначальная толщина ствола, def: W // 10
MINIMUM_TRUNK_RADIUS = 4     # Минимальная толщина ствола 128: 5, 64: 3
MINIMUM_BRANCH_RADIUS = 2  # Минимальная толщина ветки 128: 3, 64: 1
r_branch = 0.4 * (STARTING_RADIUS - MINIMUM_TRUNK_RADIUS) / H  # Коэффициент радиуса ствола
 
BORDER_SIZE = MINIMUM_BRANCH_RADIUS // 2 # Размер границы дерева
SHADE_SIZE = MINIMUM_BRANCH_RADIUS // 1.25 # Размер тени дерева

bg_color = (1, 1, 1)
 
trunk_light = (161/255,110/255,81/255) 
trunk_shade = (122/255, 68/255, 34/255)
trunk_dark = (82/255, 61/255, 44/255)
        
MIN_DISTANCE_FOR_LEAVES = 2
leaves_prob = 0.3
 
leaf4 = (90/255, 42/255, 44/255)
angle4 = 0.40
leaf3 = (129/255, 60/255, 64/ 255)
angle3 = 0.50
leaf2 = (162/255, 75/255, 81/255)
angle2 = 0.65
leaf1 = (189/255, 129/255, 133/255)
 
MIN_NUMBER_OF_INTER_VERTEXES = 1 # Минимальное количество вершин, на которые разбивается ветка
MAX_NUMBER_OF_INTER_VERTEXES = 2 # Максимальное количество вершин, на которые разбивается ветка
 
def smooth_min(dstA, dstB, k):
    h = max(k - abs(dstA - dstB), 0) / k
    return min(dstA, dstB) - (h ** 3) * k / 6
 
def segments_distance(x11, y11, x12, y12, x21, y21, x22, y22):
    """ distance between two segments in the plane:
      one segment is (x11, y11) to (x12, y12)
      the other is   (x21, y21) to (x22, y22)
  """
    if segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22): return 0
    # try each of the 4 vertices w/the other segment
    distances = []
    distances.append(point_segment_distance(x11, y11, x21, y21, x22, y22))
    distances.append(point_segment_distance(x12, y12, x21, y21, x22, y22))
    distances.append(point_segment_distance(x21, y21, x11, y11, x12, y12))
    distances.append(point_segment_distance(x22, y22, x11, y11, x12, y12))
    return min(distances)
 
 
def segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22):
    """ whether two segments in the plane intersect:
      one segment is (x11, y11) to (x12, y12)
      the other is   (x21, y21) to (x22, y22)
  """
    dx1 = x12 - x11
    dy1 = y12 - y11
    dx2 = x22 - x21
    dy2 = y22 - y21
    delta = dx2 * dy1 - dy2 * dx1
    if delta == 0: return False  # parallel segments
    s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta
    t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / (-delta)
    return (0 <= s <= 1) and (0 <= t <= 1)
 
 
def point_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:  # the segment's just a point
        return math.hypot(px - x1, py - y1)
 
    # Calculate the t that minimizes the distance.
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
 
    # See if this represents one of the segment's
    # end points or a point in the middle.
    if t < 0:
        dx = px - x1
        dy = py - y1
    elif t > 1:
        dx = px - x2
        dy = py - y2
    else:
        near_x = x1 + t * dx
        near_y = y1 + t * dy
        dx = px - near_x
        dy = py - near_y
 
    return math.hypot(dx, dy)
 
def current_number_of_vertexes():
    return len(graph)
 
BRANCH_LENGTH_L = 0.9
BRANCH_LENGTH_M = 1
BRANCH_LENGTH_R = 1.02
def generate_branch_length(expected_length):
    scale = ss.norm.rvs(loc=BRANCH_LENGTH_M, scale=0.2)
    while scale < BRANCH_LENGTH_L or scale > BRANCH_LENGTH_R:
        scale = ss.norm.rvs(loc=BRANCH_LENGTH_M, scale=0.2)
    return expected_length * scale
 
RADIUS_SCALE_L = 0.6
RADIUS_SCALE_M = 0.7
RADIUS_SCAE_R = 0.8
def generate_branch_radius_scale():
    scale = ss.norm.rvs(loc=RADIUS_SCALE_M, scale=0.2)
    while scale < RADIUS_SCALE_L or scale > RADIUS_SCALE_R:
        scale = ss.norm.rvs(loc=RADIUS_SCALE_M, scale=0.2)
    return scale
 
def generate_branch_radius(expected_radius):
    expected_radius = max(expected_radius, MINIMUM_BRANCH_RADIUS)
    return expected_radius * generate_branch_radius_scale()
 
def generate_branch(i, angles, is_branch=False, l=0, r=0):
    expected_branch_length = STARTING_LENGTH * pow(q, vertex_dists[i])
    expected_branch_length = max(expected_branch_length, MIN_BRANCH_LENGTH)
 
    angle = random.randint(*angles)
    branch_length = generate_branch_length(expected_branch_length)
    dw = int(math.cos(math.radians(angle)) * branch_length)
    dh = int(math.sin(math.radians(angle)) * branch_length)
    if l < r:
        dw = -dw
    elif l == r and random.random() < 0.5:
        dw = -dw
    y, x = graph[i]
 
    if not is_branch:
        current_rad_scale = 1
    else:
        current_rad_scale = generate_branch_radius_scale()
    branch_rad = generate_branch_radius(current_rad_scale * vertex_rads[i] - branch_length * r_branch)
 
    return y + dh, x + dw, current_rad_scale, branch_rad
 
 
def add_branch(i, new_y, new_x, rad_coef=1, branch_rad=MINIMUM_BRANCH_RADIUS, is_branch=False):
    graph.append((new_y, new_x))
    edges[i].append((len(graph) - 1, rad_coef))
    edges.append([])
    if is_branch:
        number_of_branches_down.append(number_of_branches_down[i] + 1)
    else:
        number_of_branches_down.append(number_of_branches_down[i])
    vertex_dists.append(vertex_dists[i] + 1)
    vertex_rads.append(branch_rad)
 
 
BRANCH_INTERSECTION_COEF = 1
def check_branch(i, new_y, new_x, branch_rad = MINIMUM_BRANCH_RADIUS):
    if new_y <= C or new_y >= H - C or new_x <= C or new_x >= H - C:
        return False
    for key in range(len(edges)):
        edge = edges[key]
        if key == i:
            continue
        for j, _ in edge:
            if j == i:
                continue
            y, x = graph[j]
            if segments_distance(graph[i][0], graph[i][1], new_y, new_x, graph[key][0], graph[key][1], y, x) < (branch_rad + vertex_rads[j]) * BRANCH_INTERSECTION_COEF:
                return False
    return True
 
 
def build_graph(i=0, total_branch=0, l=0, r=0, last_branch=1):
    max_depth.append(0)
    if(len(graph) == MAX_VERTEXES_NUMBER):
        return
    if graph[i][0] >= MAX_TREE_HEIGHT or graph[i][1] >= H - C or graph[i][1] <= C:
        return
    if vertex_dists[i] >= MIN_TREE_HEIGHT_IN_VERTEXES and random.random() < STOP_PROB:
        return
    if vertex_dists[i] >= MAX_TREE_HEIGHT_IN_VERTEXES:
        return
    y, x = graph[i]
    if  total_branch < MAX_NUMBER_OF_BRANCH and i != 0 and (
            last_branch > MAX_DEPTH_WITHOUT_BRANCH or random.random() < BRANCH_PROB):
        number_of_tries = 0
        while number_of_tries < MAX_BRANCH_TRY:
            new_y, new_x, current_rad_scale, branch_rad = generate_branch(i, BRANCH_ANGLES, True, l, r)
            if not check_branch(i, new_y, new_x, branch_rad):
                number_of_tries += 1
                continue
            add_branch(i, new_y, new_x, current_rad_scale, branch_rad, True)
            old_n = len(graph)
            build_graph(len(graph) - 1, total_branch + 1, 0, 0)
            max_depth[i] = max(max_depth[i], 1 + max_depth[old_n - 1])
            new_n = len(graph)
            if new_x < x:
                l += new_n - old_n
            else:
                r += new_n - old_n
            last_branch = 0
            break
        if number_of_tries == MAX_BRANCH_TRY:
            return
 
    number_of_tries = 0
    while number_of_tries < MAX_BRANCH_TRY:
        if i:
            new_y, new_x, current_rad_scale, branch_rad = generate_branch(i, STRAIGHT_ANGLES, False, l, r)
        else:
            new_y, new_x, current_rad_scale, branch_rad = generate_branch(i, [90, 90], False, l, r)
        if not check_branch(i, new_y, new_x, branch_rad):
            number_of_tries += 1
            continue
        add_branch(i, new_y, new_x, current_rad_scale, branch_rad, False)
        j = len(graph) - 1
        build_graph(len(graph) - 1, total_branch, l, r, last_branch + 1)
        max_depth[i] = max(max_depth[i], 1 + max_depth[j])
        break
 
 
def make_tree():
    global vertex_dists
    global graph
    global edges
    global vertex_rads
    global number_of_branches_down
    global max_depth
    number_of_branches_down = [0]
    max_depth = []
    vertex_rads = [STARTING_RADIUS]
    vertex_dists = [0]
    graph = [(0, H // 2)]
    edges = [[]]
    build_graph()
 
def check_tree():
    if len(graph) < MIN_VERTEXES_NUMBER or len(graph) > MAX_VERTEXES_NUMBER:
        return False
    l = 0
    r = 0
    for v in graph:
        if v[1] < (H // 2):
            l += 1
        if v[1] > (H // 2):
            r += 1
    if abs(r - l) > MAX_SYMMETRY_DIFF:
        return False
    return True
 
def destruct_edge(y_start, x_start, y_end, x_end, rad_start, rad_end):
    number_of_new_ver = ss.randint.rvs(MIN_NUMBER_OF_INTER_VERTEXES, MAX_NUMBER_OF_INTER_VERTEXES)
    delta_rad = (rad_start - rad_end) / (number_of_new_ver + 1)
    dy = (y_end - y_start) / (number_of_new_ver + 1)
    dx = (x_end - x_start) / (number_of_new_ver + 1)
    vertexes = []
    radiuses = []
    for i in range(1, number_of_new_ver + 1):
        vertexes.append((y_start + dy * i, x_start + dx * i))
        radiuses.append(rad_start - delta_rad * i)
    return vertexes, radiuses
 
 
dist_coef = 1
dists = [[1e9 for i in range(H)] for j in range(H)] # Расстояния от точки до ближайшего листа
def get_dist():
    global dists
    dists = [[1e9 for i in range(H)] for j in range(H)] 
    queue = []
    for i in range(len(graph)):
        v = graph[i]
        if max_depth[i] == 0 or is_leaves[i]:
            queue.append((v[0], v[1]))
            dists[v[0]][v[1]] = 0
    if H == 128:
        dobavka = ss.norm(50, 40).rvs(H * H) # 128: (50, 40), 64: (25, 20)
    else:
        dobavka = ss.norm(4, 4).rvs(H * H) # 128: (50, 40), 64: (25, 20)
    for i in range(H):
        for j in range(H):
            for v in queue:
                ln = math.sqrt((i - v[0]) ** 2 + (j - v[1]) ** 2 / (dist_coef ** 2)) # Def: 1.75
                fun = abs(H - v[0]) * int(ln)
                dists[i][j] = min(dists[i][j], fun)
            if H == 128:
                dists[i][j] -= min(max(0, dobavka[i * H + j]), 80) # 128: 80, 64: 40
            else:
                dists[i][j] -= min(max(0, dobavka[i * H + j]), 13) # 128: 80, 64: 40
 
pixel_colors = [[(-1, -1, -1) for i in range(H)] for j in range(H)]
def color_pixels():
    for i in range(H):
        for j in range(H):
            if heights[i][j] == 0:
                continue
            cc = (0, 0, 0)
            if angles[i][j] <= angle4:
                cc = leaf4
            elif angles[i][j] <= angle3:
                cc = leaf3
            elif angles[i][j] <= angle2:
                cc = leaf2
            else:
                cc = leaf1
            pixel_colors[i][j] = cc
 
 
heights = [[0 for i in range(H)] for j in range(H)]
angles = [[0 for i in range(H)] for j in range(H)]
HEIGHT_CONST = 400 # 128: 750, 64: 175
def build_heights():
    for i in range(H):
        for j in range(H):
            df = min(dists[i][j], HEIGHT_CONST) / HEIGHT_CONST
            cc = math.sqrt(max(0, 1 - df ** 2))
            heights[i][j] = cc
 
    camera_vec = (0, -1, 0)
    ln = math.sqrt(camera_vec[0] ** 2 + camera_vec[1] ** 2 + camera_vec[2] ** 2)
    camera_vec = (camera_vec[0] / ln, camera_vec[1] / ln, camera_vec[2] / ln)
    for i in range(1, H - 1):
        for j in range(1, H - 1):
            if heights[i][j] == 0:
                continue
 
            vec1 = [2 * (heights[i + 1][j] - heights[i - 1][j]), 2 * (heights[i][j + 1] - heights[i][j - 1]), 4]
            vec1[0], vec1[1] = vec1[1], vec1[0]
            angles[i][j] = vec1[0] * camera_vec[0] + vec1[1] * camera_vec[1] + vec1[2] * camera_vec[2]
            angles[i][j] = (angles[i][j] + 1) / 2 
 
 
def build_leaves(i):
   if i == 0:
       is_leaves[i] = False
   for j, _ in edges[i]:
       if is_leaves[i] or (vertex_dists[j] >= MIN_DISTANCE_FOR_LEAVES and random.random() < leaves_prob):
           is_leaves[j] = True
       else:
           is_leaves[j] = False
       build_leaves(j)
 
def add_more_tree_structure():
    global is_leaves
    is_leaves = [0 for i in range(len(graph))]
 
    build_leaves(0)
    get_dist()
    build_heights()
    color_pixels()

def make_new_graph(is_birch=False):
    global new_graph
    global new_edges
    global new_rads
    global number_of_old_vertexes
    new_graph = []
    new_edges = []
    new_rads = []
 
    n = len(graph)
    number_of_old_vertexes = n
    for i in range(n):
        new_graph.append(graph[i])
        new_rads.append(vertex_rads[i])
        new_edges.append([])
 
    
    for i in range(n):
        for j, rad_coef in edges[i]:
            new_vert, new_rad = destruct_edge(graph[i][0], graph[i][1], graph[j][0], graph[j][1], vertex_rads[i] * rad_coef, vertex_rads[j])
            for v in range(len(new_vert)):
                new_graph.append(new_vert[v])
                new_rads.append(new_rad[v])
                new_edges.append([])
                if v == 0:
                    new_edges[i].append(len(new_graph) - 1)
                else:
                    new_edges[len(new_graph) - 2].append(len(new_graph) - 1)
            new_edges[len(new_graph) - 1].append(j)
    
    if not is_birch:
        return
    global tiles
    tiles = [(-1, -1) for _ in range(len(new_graph))]
    for i in range(1, len(new_graph)):
        y, x = new_graph[i]
        rad = new_rads[i] / 2 - 1
        if new_rads[i] < MINIMUM_BRANCH_RADIUS + 2:
            continue
        if x - rad >= x + rad:
            continue
        if i % 2 == 1:
            continue
        if random.random() < 0.5:
            l = x - rad + 1
            r = l + max(rad / 1.5, int(min(rad / 2, max(4, ss.expon(scale=3).rvs(1)))))
        else:
            r = x + rad - 1
            l = r - max(rad / 1.5, int(min(rad / 2, max(4, ss.expon(scale=3).rvs(1)))))
        tiles[i] = (l, r)
 
def disp(draw_func, file_name, is_birch=False):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, REAL_IMAGE_SIZE, REAL_IMAGE_SIZE)
    ctx = cairo.Context(surface)
    draw_func(ctx, is_birch)

    images_bytes = list(surface.get_data().tobytes())
    image = []
    cur = 0
    for i in range(REAL_IMAGE_SIZE):
        image.append([])
        for j in range(REAL_IMAGE_SIZE):
            image[-1].append((images_bytes[cur], images_bytes[cur + 1], images_bytes[cur + 2]))
            cur += 4

    with BytesIO() as fileobj:
        surface.write_to_png(fileobj)
        with open(f"{file_name}.png", "wb") as f:
            f.write(fileobj.getvalue())
    
    return image
 
def draw_leaves(cr: cairo.Context, is_birch=False):
    # cr.set_source_rgb(*bg_color)
    # cr.fill()
 
    cr.set_line_cap(cairo.LINE_CAP_ROUND)
    cr.set_antialias(cairo.ANTIALIAS_NONE)
    
    delta = H * (REAL_IMAGE_COEF // 2)
    if H == 128:
        M = (delta + H // 2, delta + H - 5)
        L = (delta + H // 2 - 15, delta + H + 2)
        R = (delta + H // 2 + 15, delta + H + 2)
    elif H == 48:
        M = (delta + H // 2, delta + H - 2)
        L = (delta + H // 2 - 5, delta + H + 1)
        R = (delta + H // 2 + 5, delta + H + 1)
    if not is_birch:
        cr.set_source_rgb(*trunk_dark)
        for key in range(len(new_edges)):
            value = new_edges[key]
            for id in value:
                cr.set_line_width(new_rads[id] + 1)
                cr.move_to(delta + new_graph[key][1], delta + H - new_graph[key][0])
                cr.line_to(delta + new_graph[id][1], delta + H - new_graph[id][0])
                cr.stroke()
        
        cr.set_line_width(5)
        cr.move_to(*M)
        cr.line_to(*L)
        cr.stroke()
        cr.move_to(*M)
        cr.line_to(*R)
        cr.stroke()
        cr.move_to(*L)
        cr.line_to(*R)
        cr.stroke()
 
    cr.set_source_rgb(*trunk_shade)
    for key in range(len(new_edges)):
        value = new_edges[key]
        for id in value:
            cr.set_line_width(new_rads[id] - BORDER_SIZE )
            cr.move_to(delta + new_graph[key][1], delta + H - new_graph[key][0])
            cr.line_to(delta + new_graph[id][1], delta + H - new_graph[id][0])
            cr.stroke()
        
        cr.set_line_width(4)
        cr.move_to(*M)
        cr.line_to(*L)
        cr.stroke()
        cr.move_to(*M)
        cr.line_to(*R)
        cr.stroke()
        cr.move_to(*L)
        cr.line_to(*R)
        cr.stroke()

 
    cr.set_source_rgb(*trunk_light)
    for key in range(len(new_edges)):
        value = new_edges[key]
        for id in value:
            shade_size = (new_rads[id]) / 4
            cr.set_line_width(max(1, new_rads[id] - BORDER_SIZE - shade_size / 2 - 1))
            cr.move_to(delta + new_graph[key][1] - shade_size / 2, delta + H - new_graph[key][0])
            cr.line_to(delta + new_graph[id][1] - shade_size / 2, delta + H - new_graph[id][0])
            cr.stroke()
        
        cr.set_line_width(2)
        cr.move_to(*M)
        cr.line_to(*L)
        cr.stroke()
        cr.move_to(*M)
        cr.line_to(*R)
        cr.stroke()
        cr.move_to(*L)
        cr.line_to(*R)
        cr.stroke()
    
    if is_birch:
        cr.set_line_width(1)
        for i in range(1, len(new_graph)):
            l, r = tiles[i]
            y, x = new_graph[i]
            if l == -1:
                continue
            cr.set_source_rgb(0, 0, 0)
            cr.move_to(delta + l, delta + H - y)
            cr.line_to(delta + r, delta + H - y)
            cr.stroke()
 
    for i in range(H):
        for j in range(H):
            if heights[i][j] == 0:
                continue
            cr.set_source_rgb(*pixel_colors[i][j])
            cr.arc(delta + j, delta + H - i, 1, 0, 2 * math.pi)
            cr.fill()
 
def build_healthy_maple():
    global leaf1, leaf2, leaf3, leaf4, leaves_prob, HEIGHT_CONST, MIN_DISTANCE_FOR_LEAVES
    leaf4 = (90/255, 42/255, 44/255)
    leaf3 = (129/255, 60/255, 64/ 255)
    leaf2 = (162/255, 75/255, 81/255)
    leaf1 = (189/255, 129/255, 133/255)
    leaves_prob = 0.4
    HEIGHT_CONST = 750
    MIN_DISTANCE_FOR_LEAVES = 2
    add_more_tree_structure()

def build_unhealthy_maple():
    global leaf1, leaf2, leaf3, leaf4, leaves_prob, HEIGHT_CONST, MIN_DISTANCE_FOR_LEAVES
    leaf4 = (96/255, 32/255, 11/255)
    leaf3 = (120/255, 40/255, 14/ 255)
    leaf2 = (160/255, 104/255, 86/255)
    leaf1 = (188/255, 149/255, 136/255)
    leaves_prob = 0.35
    HEIGHT_CONST = 450
    MIN_DISTANCE_FOR_LEAVES = 3
    add_more_tree_structure()

def build_dead_maple():
    global leaf1, leaf2, leaf3, leaf4, leaves_prob, HEIGHT_CONST, MIN_DISTANCE_FOR_LEAVES
    leaf4 = (96/255, 32/255, 11/255)
    leaf3 = (120/255, 40/255, 14/ 255)
    leaf2 = (160/255, 104/255, 86/255)
    leaf1 = (188/255, 149/255, 136/255)
    leaves_prob = 0.35
    HEIGHT_CONST = 1
    MIN_DISTANCE_FOR_LEAVES = 3
    add_more_tree_structure()
 

def build_maple():
    global STOP_PROB, BRANCH_PROB, STRAIGHT_ANGLES, BRANCH_ANGLES
    STOP_PROB = 0.8
    BRANCH_PROB = 0.4
    STRAIGHT_ANGLES = (60, 85)
    BRANCH_ANGLES = (30, 55)
    
    global C
    C = 5  # Паддинг
    
    global MAX_TREE_HEIGHT_COEF, MAX_TREE_HEIGHT
    MAX_TREE_HEIGHT_COEF = 0.7 
    MAX_TREE_HEIGHT = (H - C) * MAX_TREE_HEIGHT_COEF
    
    global MIN_VERTEXES_NUMBER
    MIN_VERTEXES_NUMBER = 24  # Минимальное количество вершин в дереве
    
    global MAX_NUMBER_OF_BRANCH, MAX_DEPTH_WITHOUT_BRANCH, MIN_TREE_HEIGHT_IN_VERTEXES, MAX_TREE_HEIGHT_IN_VERTEXES, MAX_SYMMETRY_DIFF
    MAX_NUMBER_OF_BRANCH = 4  # Максимальное количество ветвлений на одной ветке
    MAX_DEPTH_WITHOUT_BRANCH = 4  # Cколько вершин можно пройти не наткнувся на ветку
    MIN_TREE_HEIGHT_IN_VERTEXES = 6 # Минимальная высота дерева в вершинах
    MAX_TREE_HEIGHT_IN_VERTEXES = 35 # Максимальная высота дерева в вершинах
    MAX_SYMMETRY_DIFF = 3
    
    global STARTING_LENGTH, LENGTH_COEF, cst, q
    STARTING_LENGTH = 15  # Изначальная длина ствола дерева, def: w // 4
    LENGTH_COEF = 7
    cst = 0.7  # Коэффиент для коэффициента Больше => длиннее ветки
    q = 1 - LENGTH_COEF / (cst * (H - C))  # Коэффициент уменьшения длины ветви
    
    global STARTING_RADIUS, MINIMUM_TRUNK_RADIUS, MINIMUM_BRANCH_RADIUS, r_branch
    STARTING_RADIUS = 15  # Изначальная толщина ствола, def: W // 10
    MINIMUM_TRUNK_RADIUS = 4     # Минимальная толщина ствола 128: 5, 64: 3
    MINIMUM_BRANCH_RADIUS = 2  # Минимальная толщина ветки 128: 3, 64: 1
    r_branch = 0.4 * (STARTING_RADIUS - MINIMUM_TRUNK_RADIUS) / H  # Коэффициент радиуса ствола

    global bg_color
    bg_color = (216/255, 194/255, 247/255)
    
    global trunk_light, trunk_dark, trunk_shade
    trunk_light = (161/255,110/255,81/255) 
    trunk_shade = (122/255, 68/255, 34/255)
    trunk_dark = (82/255, 61/255, 44/255)

    global MIN_DISTANCE_FOR_LEAVES, leaves_prob
    MIN_DISTANCE_FOR_LEAVES = 2
    leaves_prob = 0.3
    
    global leaf1, leaf2, leaf3, leaf4
    leaf4 = (90/255, 42/255, 44/255)
    leaf3 = (129/255, 60/255, 64/ 255)
    leaf2 = (162/255, 75/255, 81/255)
    leaf1 = (189/255, 129/255, 133/255)
    
    global MIN_NUMBER_OF_INTER_VERTEXES, MAX_NUMBER_OF_INTER_VERTEXES
    MIN_NUMBER_OF_INTER_VERTEXES = 1 # Минимальное количество вершин, на которые разбивается ветка
    MAX_NUMBER_OF_INTER_VERTEXES = 2 # Максимальное количество вершин, на которые разбивается ветка

    global HEIGHT_CONST
    HEIGHT_CONST = 400

    global RADIUS_SCALE_L, RADIUS_SCALE_M, RADIUS_SCALE_R
    RADIUS_SCALE_L = 0.6
    RADIUS_SCALE_M = 0.7
    RADIUS_SCALE_R = 0.8

    global BRANCH_LENGTH_L, BRANCH_LENGTH_M, BRANCH_LENGTH_R
    BRANCH_LENGTH_L = 0.9
    BRANCH_LENGTH_M = 1
    BRANCH_LENGTH_R = 1.02

    global dist_coef
    dist_coef = 1

    make_tree()
    while not check_tree():
        make_tree()
    make_new_graph()

    build_healthy_maple()
    healthy = disp(draw_leaves, "healthy")

    build_unhealthy_maple()
    unhealthy = disp(draw_leaves, "unhealthy")

    build_dead_maple()
    dead = disp(draw_leaves, "dead")
    return healthy, unhealthy, dead

def build_healthy_oak(is_gray = False):
    global leaf1, leaf2, leaf3, leaf4, leaves_prob, HEIGHT_CONST, MIN_DISTANCE_FOR_LEAVES
    leaf4 = (0/255, 65/255, 0/255)
    leaf3 = (0/255, 97/255, 0/255)
    leaf2 = (0/255, 158/255, 0/255)
    leaf1 = (0/255, 200/255, 0/255)
    if is_gray:
        leaf4 = (78/255, 78/255, 78/255)
        leaf3 = (131/255, 131/255, 131/255)
        leaf2 = (188/255, 188/255, 188/255)
        leaf1 = (201/255, 201/255, 201/255)
        global trunk_dark, trunk_shade, trunk_light
        trunk_light = (0.75, 0.75, 0.75)
        trunk_dark = (0, 0, 0)
        trunk_shade = (80/255, 80/255, 80/255)
        global bg_color
        bg_color = (0.6, 0.6, 0.6)
    leaves_prob = 0.7
    if H == 128:
        HEIGHT_CONST = 750
    elif H == 48:
        HEIGHT_CONST = 110
    MIN_DISTANCE_FOR_LEAVES = 2
    add_more_tree_structure()
 
def build_unhealthy_oak():
    global leaf1, leaf2, leaf3, leaf4, leaves_prob, HEIGHT_CONST, MIN_DISTANCE_FOR_LEAVES
    leaf4 = (94/255, 74/255, 51/255)
    leaf3 = (135/255, 107/255, 74/255)
    leaf2 = (193/255, 154/255, 107/255)
    leaf1 = (205/255, 174/255, 136/255)
    leaves_prob = 0.5
    if H == 128:
        HEIGHT_CONST = 400
    elif H == 48:
        HEIGHT_CONST = 60
    MIN_DISTANCE_FOR_LEAVES = 3
    add_more_tree_structure()
 
def build_dead_oak():
    global leaf1, leaf2, leaf3, leaf4, leaves_prob, HEIGHT_CONST, MIN_DISTANCE_FOR_LEAVES
    leaf4 = (94/255, 74/255, 51/255)
    leaf3 = (135/255, 107/255, 74/255)
    leaf2 = (193/255, 154/255, 107/255)
    leaf1 = (205/255, 174/255, 136/255)
    leaves_prob = 0.1
    HEIGHT_CONST = 1
    MIN_DISTANCE_FOR_LEAVES = 3
    add_more_tree_structure()

def build_oak():
    global STOP_PROB, BRANCH_PROB, STRAIGHT_ANGLES, BRANCH_ANGLES
    STOP_PROB = 0.9
    BRANCH_PROB = 0.9
    STRAIGHT_ANGLES = (30, 85)
    BRANCH_ANGLES = (10, 50)
    
    global C
    if H == 128:
        C = 25 
    elif H == 48:
        C = 8
    
    global MAX_TREE_HEIGHT_COEF, MAX_TREE_HEIGHT
    MAX_TREE_HEIGHT_COEF = 0.7 
    MAX_TREE_HEIGHT = (H - C) * MAX_TREE_HEIGHT_COEF
    
    global MIN_VERTEXES_NUMBER
    MIN_VERTEXES_NUMBER = 18  # Минимальное количество вершин в дереве
    
    global MAX_NUMBER_OF_BRANCH, MAX_DEPTH_WITHOUT_BRANCH, MIN_TREE_HEIGHT_IN_VERTEXES, MAX_TREE_HEIGHT_IN_VERTEXES, MAX_SYMMETRY_DIFF
    MAX_NUMBER_OF_BRANCH = 4  # Максимальное количество ветвлений на одной ветке
    MAX_DEPTH_WITHOUT_BRANCH = 1  # Cколько вершин можно пройти не наткнувся на ветку
    MIN_TREE_HEIGHT_IN_VERTEXES = 6 # Минимальная высота дерева в вершинах
    MAX_TREE_HEIGHT_IN_VERTEXES = 35 # Максимальная высота дерева в вершинах
    MAX_SYMMETRY_DIFF = 3
    
    global STARTING_LENGTH, LENGTH_COEF, cst, q
    if H == 128:
        STARTING_LENGTH = 30  # Изначальная длина ствола дерева, def: w // 4
        LENGTH_COEF = 10
    elif H == 48:
        STARTING_LENGTH = 10
        LENGTH_COEF = 3
    cst = 0.4  # Коэффиент для коэффициента Больше => длиннее ветки
    q = 1 - LENGTH_COEF / (cst * (H - C))  # Коэффициент уменьшения длины ветви
    
    global STARTING_RADIUS, MINIMUM_TRUNK_RADIUS, MINIMUM_BRANCH_RADIUS, r_branch
    if H == 128:
        STARTING_RADIUS = 20  # Изначальная толщина ствола, def: W // 10
        MINIMUM_TRUNK_RADIUS = 7     # Минимальная толщина ствола 128: 5, 64: 3
        MINIMUM_BRANCH_RADIUS = 4  # Минимальная толщина ветки 128: 3, 64: 1
    elif H == 48:
        STARTING_RADIUS = 7
        MINIMUM_TRUNK_RADIUS = 3
        MINIMUM_BRANCH_RADIUS = 1
    r_branch = 1.25 * (STARTING_RADIUS - MINIMUM_TRUNK_RADIUS) / H  # Коэффициент радиуса ствола

    global bg_color
    bg_color = (122/255, 237/255, 157/255)
    
    global trunk_light, trunk_dark, trunk_shade
    trunk_dark = (50/255, 32/255, 9/255)
    trunk_light = (140/255,89/255,26/255) 
    trunk_shade = (84/255, 53/255, 15/255)

    global MIN_DISTANCE_FOR_LEAVES, leaves_prob
    MIN_DISTANCE_FOR_LEAVES = 2
    leaves_prob = 0.3
    
    global leaf1, leaf2, leaf3, leaf4
    leaf4 = (94/255, 74/255, 51/255)
    leaf3 = (135/255, 107/255, 74/255)
    leaf2 = (193/255, 154/255, 107/255)
    leaf1 = (205/255, 174/255, 136/255)
    
    global MIN_NUMBER_OF_INTER_VERTEXES, MAX_NUMBER_OF_INTER_VERTEXES
    MIN_NUMBER_OF_INTER_VERTEXES = 1 # Минимальное количество вершин, на которые разбивается ветка
    MAX_NUMBER_OF_INTER_VERTEXES = 2 # Максимальное количество вершин, на которые разбивается ветка

    global HEIGHT_CONST
    if H == 128:
        HEIGHT_CONST = 400
    elif H == 48:
        HEIGHT_CONST = 110

    global RADIUS_SCALE_L, RADIUS_SCALE_M, RADIUS_SCALE_R
    RADIUS_SCALE_L = 0.6
    RADIUS_SCALE_M = 0.7
    RADIUS_SCALE_R = 0.8

    global BRANCH_LENGTH_L, BRANCH_LENGTH_M, BRANCH_LENGTH_R
    BRANCH_LENGTH_L = 0.9
    BRANCH_LENGTH_M = 1
    BRANCH_LENGTH_R = 1.02

    global dist_coef
    dist_coef = 1.40

    make_tree()
    while not check_tree():
        make_tree()
    make_new_graph()

    build_healthy_oak()
    healthy = disp(draw_leaves, "healthy")

    build_unhealthy_oak()
    unhealthy = disp(draw_leaves, "unhealthy")

    build_dead_oak()
    dead = disp(draw_leaves, "dead")
    return healthy, unhealthy, dead

def build_healthy_birch():
    global leaf1, leaf2, leaf3, leaf4, leaves_prob, HEIGHT_CONST, MIN_DISTANCE_FOR_LEAVES
    leaf4 = (107/255, 40/255, 16/255)
    leaf3 = (154/255, 58/255, 24/255)
    leaf2 = (220/255, 84/255, 35/255)
    leaf1 = (227/255, 118/255, 78/255)
    leaves_prob = 0.7
    if H == 128:
        HEIGHT_CONST = 500
    elif H == 48:
        HEIGHT_CONST = 90
    MIN_DISTANCE_FOR_LEAVES = 2
    add_more_tree_structure()
 
def build_unhealthy_birch():
    global leaf1, leaf2, leaf3, leaf4, leaves_prob, HEIGHT_CONST, MIN_DISTANCE_FOR_LEAVES
    leaf4 = (70/255, 41/255, 2/255)
    leaf3 = (100/255, 59/255, 3/255)
    leaf2 = (131/255, 98/255, 53/255)
    leaf1 = (168/255, 145/255, 113/255)
    leaves_prob = 0.4
    if H == 128:
        HEIGHT_CONST = 300
    elif H == 48:
        HEIGHT_CONST = 50
    MIN_DISTANCE_FOR_LEAVES = 3
    add_more_tree_structure()
 
def build_dead_birch():
    global leaf1, leaf2, leaf3, leaf4, leaves_prob, HEIGHT_CONST, MIN_DISTANCE_FOR_LEAVES
    leaf4 = (70/255, 41/255, 2/255)
    leaf3 = (100/255, 59/255, 3/255)
    leaf2 = (131/255, 98/255, 53/255)
    leaf1 = (168/255, 145/255, 113/255)
    leaves_prob = 0.4
    HEIGHT_CONST = 1
    MIN_DISTANCE_FOR_LEAVES = 3
    add_more_tree_structure()

def build_birch():
    global STOP_PROB, BRANCH_PROB, STRAIGHT_ANGLES, BRANCH_ANGLES
    STOP_PROB = 0.4
    BRANCH_PROB = 0.4
    STRAIGHT_ANGLES = (50, 85)
    BRANCH_ANGLES = (30, 60)
    
    global C
    if H == 128:
        C = 25 
    elif H == 48:
        C = 9
    
    global MAX_TREE_HEIGHT_COEF, MAX_TREE_HEIGHT
    MAX_TREE_HEIGHT_COEF = 0.8 
    MAX_TREE_HEIGHT = (H - C) * MAX_TREE_HEIGHT_COEF
    
    global MIN_VERTEXES_NUMBER
    MIN_VERTEXES_NUMBER = 18  # Минимальное количество вершин в дереве
    
    global MAX_NUMBER_OF_BRANCH, MAX_DEPTH_WITHOUT_BRANCH, MIN_TREE_HEIGHT_IN_VERTEXES, MAX_TREE_HEIGHT_IN_VERTEXES, MAX_SYMMETRY_DIFF
    MAX_NUMBER_OF_BRANCH = 4  # Максимальное количество ветвлений на одной ветке
    MAX_DEPTH_WITHOUT_BRANCH = 1  # Cколько вершин можно пройти не наткнувся на ветку
    MIN_TREE_HEIGHT_IN_VERTEXES = 6 # Минимальная высота дерева в вершинах
    MAX_TREE_HEIGHT_IN_VERTEXES = 35 # Максимальная высота дерева в вершинах
    MAX_SYMMETRY_DIFF = 3
    
    global STARTING_LENGTH, LENGTH_COEF, cst, q
    if H == 128:
        STARTING_LENGTH = 30  # Изначальная длина ствола дерева, def: w // 4
        LENGTH_COEF = 10
    elif H == 48:
        STARTING_LENGTH = 10
        LENGTH_COEF = 3
    cst = 0.4  # Коэффиент для коэффициента Больше => длиннее ветки
    q = 1 - LENGTH_COEF / (cst * (H - C))  # Коэффициент уменьшения длины ветви
    
    global STARTING_RADIUS, MINIMUM_TRUNK_RADIUS, MINIMUM_BRANCH_RADIUS, r_branch
    if H == 128:
        STARTING_RADIUS = 15  # Изначальная толщина ствола, def: W // 10
        MINIMUM_TRUNK_RADIUS = 7     # Минимальная толщина ствола 128: 5, 64: 3
        MINIMUM_BRANCH_RADIUS = 4  # Минимальная толщина ветки 128: 3, 64: 1
    elif H == 48:
        STARTING_RADIUS = 5
        MINIMUM_TRUNK_RADIUS = 3
        MINIMUM_BRANCH_RADIUS = 1
    r_branch = 0.5 * (STARTING_RADIUS - MINIMUM_TRUNK_RADIUS) / H  # Коэффициент радиуса ствола

    global bg_color
    bg_color = (122/255, 201/255, 237/255)
    
    global trunk_light, trunk_dark, trunk_shade
    trunk_dark = (0.5, 0.5, 0.5)
    trunk_light = (0.7, 0.7, 0.7) 
    trunk_shade = (0.5, 0.5, 0.5)

    global MIN_DISTANCE_FOR_LEAVES, leaves_prob
    MIN_DISTANCE_FOR_LEAVES = 3
    leaves_prob = 0.7
    
    global leaf1, leaf2, leaf3, leaf4
    leaf4 = (107/255, 40/255, 16/255)
    leaf3 = (154/255, 58/255, 24/255)
    leaf2 = (220/255, 84/255, 35/255)
    leaf1 = (227/255, 118/255, 78/255)
    
    global MIN_NUMBER_OF_INTER_VERTEXES, MAX_NUMBER_OF_INTER_VERTEXES
    MIN_NUMBER_OF_INTER_VERTEXES = 4 # Минимальное количество вершин, на которые разбивается ветка
    MAX_NUMBER_OF_INTER_VERTEXES = 5 # Максимальное количество вершин, на которые разбивается ветка

    global HEIGHT_CONST
    if H == 128:
        HEIGHT_CONST = 500
    elif H == 48:
        HEIGHT_CONST = 100

    global RADIUS_SCALE_L, RADIUS_SCALE_M, RADIUS_SCALE_R
    RADIUS_SCALE_L = 0.8
    RADIUS_SCALE_M = 0.7
    RADIUS_SCALE_R = 1

    global BRANCH_LENGTH_L, BRANCH_LENGTH_M, BRANCH_LENGTH_R
    BRANCH_LENGTH_L = 0.9
    BRANCH_LENGTH_M = 1
    BRANCH_LENGTH_R = 1.02

    global dist_coef
    dist_coef = 1.40

    make_tree()
    while not check_tree():
        make_tree()
    make_new_graph(is_birch=True)

    build_healthy_birch()
    healthy = disp(draw_leaves, "healthy", is_birch=True)

    build_unhealthy_birch()
    unhealthy = disp(draw_leaves, "unhealthy", is_birch=True)

    build_dead_birch()
    dead = disp(draw_leaves, "dead", is_birch=True)
    return healthy, unhealthy, dead

def generate_tree(tree_name, file_name='tree.json'):
    if tree_name == 'birch':
        images = build_birch()
    elif tree_name == 'oak':
        images = build_oak()
    else:
        images = build_maple()

    json_images = json.dumps(images)
    with open(file_name, 'w') as f:
        f.write(json_images)

generate_tree('oak')