import os
import math
import time
import numpy as np
import shutil
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from functools import cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Manager, shared_memory
import multiprocessing

# 获取当前文件所在的目录
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

DATA_DIR = os.path.join(CUR_DIR, "data")
RESULT_DIR = os.path.join(CUR_DIR, "result")
TMP_DIR = os.path.join(CUR_DIR, "tmp")





np.seterr(divide='ignore', invalid='ignore')

def log_error(*args):
    print("error : ", *args)

def log_info(*args):
    print("log : ", *args)

def file_line_generator(file_path):
    if not os.path.exists(file_path):
        log_error("{} 不存在！".format(file_path))
        return 
    with open(file_path,"r") as f:
        for line in f:
            yield line.strip()

class Pos():
    def __init__(self,x = 0, y = 0, z = 0):
        self.x = x
        self.y = y
        self.z = z

    def dis(self,other):
        dis_x = other.x - self.x
        dis_y = other.y - self.y
        dis_z = other.z - self.z
        dis_list = [dis_x, dis_y, dis_z]

        return math.sqrt(sum(d * d for d in dis_list))

class Box():
    def __init__(self, **config):
        self.x_min = config["x_min"]
        self.x_max = config["x_max"]

        self.y_min = config["y_min"]
        self.y_max = config["y_max"]

        self.z_min = config["z_min"]
        self.z_max = config["z_max"]

    def mod_in_box(self, pos):
        len_x = self.x_max - self.x_min
        len_y = self.y_max - self.y_min
        len_z = self.z_max - self.z_min

        x = (pos.x - self.x_min) % len_x + self.x_min
        y = (pos.y - self.y_min) % len_y + self.y_min
        z = (pos.z - self.z_min) % len_z + self.z_min

        new_pos = Pos(x, y, z)

        return new_pos

class Atom():
    def __init__(self,atom_id, atom_type, x, y, z):
        self.atom_id = atom_id
        self.atom_type = atom_type
        self.pos = Pos(x, y, z)
        self.cache_dis_dict = {}

    def dis(self, other):
        # cache一下加快避免重复计算
        if self.atom_id in other.cache_dis_dict:
            return other.cache_dis_dict[self.atom_id]
        if other.atom_id in self.cache_dis_dict:
            return self.cache_dis_dict[other.atom_id]

        dis = self.pos.dis(other.pos)
        self.cache_dis_dict[other.atom_id] = dis
        other.cache_dis_dict[self.atom_id] = dis

        return dis

    def set_pos(self, pos):
        self.pos = pos

    def get_pos(self):
        return self.pos

    def get_atom_id():
        return self.atom_id

    def __eq__(self, other):
        return self.atom_id == other.atom_id

    def __hash__(self):
        return hash(self.atom_id)

class Frame():
    def __init__(self, QS):
        self.atom_list = []
        self.atom_count = 0
        self.box = None
        self.total_dis = 0
        self.frame_id = -1
        self.QS = QS

    def set_frame_id(self, frame_id):
        self.frame_id = frame_id

    def set_atom_count(self, atom_count):
        self.atom_count = atom_count

    def get_atom_count(self):
        return self.atom_count
    
    def add_atom(self, atom):
        # 重定位
        old_pos = atom.get_pos()
        new_pos = self.box.mod_in_box(old_pos)
        atom.set_pos(new_pos)
        self.atom_list.append([new_pos.x, new_pos.y, new_pos.z])

    def set_box(self, box):
        self.box = box

    def cal_with_Qs(self):
        D_max = L / 6 
        # 确保输入是 numpy 数组
        positions = np.array(self.atom_list)  # (N, 3)

        # 使用 scipy 计算距离矩阵并稀疏化
        distance_matrix = cdist(positions, positions)  # (N, N)
        np.fill_diagonal(distance_matrix, np.nan)  # 排除自身距离

        # 稀疏化距离矩阵（仅保留距离 <= D_max 的点对）
        mask = distance_matrix <= D_max
        row, col = np.where(mask)
        data = distance_matrix[mask]
        sparse_distance_matrix = coo_matrix((data, (row, col)), shape=distance_matrix.shape)

        N = len(self.atom_list)
        K = 1 / ((N + 1) ** 2)  # 常数因子

        # 将稀疏矩阵的值直接计算成结果
        results = []
        for Q in self.QS:
            # 直接计算 sin(Q * distance) / distance
            scaled_data = sparse_distance_matrix.data * Q
            sin_scaled_data = np.sin(scaled_data)
            ratio_data = sin_scaled_data / sparse_distance_matrix.data

            # 计算总和
            total_sum = np.sum(ratio_data)
            total_dis = K * total_sum / Q
            results.append(total_dis)

        return results

class AverageCalculator():
    def __init__(self, file_name, max_workers, QS):
        file_path = os.path.join(DATA_DIR,file_name)
        self.file_name = file_name
        self.file_lines = file_line_generator(file_path)
        self.max_workers = max_workers
        self.frames = []
        self.QS = QS
        self.parse_frames()

    def parse_box_pos(self, line):
        offsets = line.split(" ")
        offset_1,offset_2 = float(offsets[0]), float(offsets[1])
        return offset_1,offset_2

    def parse_atom_pos(self, line):
        info = line.split(" ")
        atom_id     = int(info[0])
        atom_type   = int(info[1])
        x           = float(info[2])
        y           = float(info[3])
        z           = float(info[4])
        return atom_id, atom_type, x, y, z

    def parse_frames(self):
        line_index = 0
        box_config = {}
        sklp_line = set((2, 4, 8))
        for line in self.file_lines:
            # 新建一个frame
            if line_index == 0:
                self.frames.append(Frame(self.QS[:]))
            elif line_index == 1:
                frame_id = int(line)
                self.frames[-1].set_frame_id(frame_id)
            #跳过无用行
            elif line_index in sklp_line:
                line_index += 1
                continue
            #解析原子数
            elif line_index == 3:
                total_atom = int(line)
                self.frames[-1].set_atom_count(total_atom)
            #解析box边界
            elif line_index == 5:
                box_config['x_min'], box_config['x_max'] = self.parse_box_pos(line)
            elif line_index == 6:
                box_config['y_min'], box_config['y_max'] = self.parse_box_pos(line)
            elif line_index == 7:
                box_config['z_min'], box_config['z_max'] = self.parse_box_pos(line)  
                box = Box(**box_config)
                self.frames[-1].set_box(box)
            # 解析原子数据
            else:
                total_atom = self.frames[-1].get_atom_count()
                # print(line_index,total_atom == line_index, total_atom, type(total_atom), line)
                atom_id, atom_type, x, y, z = self.parse_atom_pos(line)
                atom = Atom(atom_id, atom_type, x, y, z)
                self.frames[-1].add_atom(atom)
                # print(atom.atom_id,"add__________")

            #下一行
            line_index += 1

            #重新添加下一个frame
            if line_index == self.frames[-1].get_atom_count() + 9:
                line_index = 0

    def cal_frames(self):
        if self.max_workers > 1:
            return self.cal_arvage_multprogress()
        else:
            return self.cal_arvage()

    def save_cal_result(self,frame_id,frame_QS):
        file_path = os.path.join(TMP_DIR,'{}_{}'.format(self.file_name,frame_id))
        with open(file_path,"w") as f:
            f.write(" ".join([str(q) for q in frame_QS]))

    def read_cal_result(self,frame_id):
        file_path = os.path.join(TMP_DIR,'{}_{}'.format(self.file_name,frame_id))
        if not os.path.exists(file_path):
            return False

        with open(file_path,"r") as f:
            result_str_list = f.read().split(" ")
            return [float(r) for r in result_str_list]

    def cal_arvage_multprogress(self):
        print("cal_arvage_multprogress")
        Q_count = len(self.QS)
        Q_result = np.zeros((Q_count,))

        need_cal_frame = []
        for frame in self.frames:
            result = self.read_cal_result(frame.frame_id)
            if not result:
                need_cal_frame.append(frame)
            else:
                log_info("skip : ",frame.frame_id)
                Q_result += result

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:  # 创建线程池
            future_to_frame = {executor.submit(frame.cal_with_Qs): frame for frame in need_cal_frame}
            for future in as_completed(future_to_frame):
                frame = future_to_frame[future]
                frame_QS = future.result()
                Q_result += frame_QS
                log_info("finish : ",frame.frame_id)
                self.save_cal_result(frame.frame_id,frame_QS)

        return Q_result / len(self.frames)

    def cal_arvage(self):
        Q_count = len(self.QS)
        Q_result = np.zeros((Q_count,))
        for frame in self.frames:
            frame_QS = frame.cal_with_Qs()
            Q_result += frame_QS

        return Q_result / len(self.frames)

    def format_print_cal_result(self):
        Q_result = self.cal_frames()

        content = []
        for i in range(len(self.QS)):
            Q,P = self.QS[i], Q_result[i]
            Q,P = format(Q, ".15e"),format(P, ".15e")
            result_line = "{}      {}".format(Q,P)
            # print(Q,P)
            content.append(result_line)

        return "\n".join(content)




def save_result(file_name, result):
    file_path = os.path.join(RESULT_DIR, file_name.replace("atom","dat"))
    with open(file_path,"w") as f:
        f.write("# Total structure factor for atom types: \n")
        f.write("# { 1(5) }in system. \n")
        f.write("# Q-values       P(Q)  \n")
        f.write(result)

L = 300
begin, end = 1, 100
q0, qmax, M = begin * 2 * math.pi / L, end * 2 * math.pi / L, end
max_workers = 5

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)



    file_names = []
    file_names.append("test.atom")
    # file_names.append("md100.atom")
    # file_names.append("md125.atom")
    # file_names.append("md150.atom")
    # file_names.append("md175.atom")

    for file_name in file_names:
        begin_time = int(time.time())
        print("begin", begin_time)
        QS = list(np.linspace(q0, qmax, M))
        file_path = os.path.join(DATA_DIR,file_name)
        ac = AverageCalculator(file_name,max_workers,QS)
        QP = ac.format_print_cal_result()
        save_result(file_name,QP)
        end_time = int(time.time())
        print("end", end_time)
        print(end_time - begin_time)

def spilit2test():
    contentList = []
    with open("data/md100.atom","r") as f:
        for i in range(5009 * 20 - 1):
            contentList.append(f.readline())
    with open("data/test.atom","w") as f:
        f.write("".join(contentList))

def clear():
    shutil.rmtree(TMP_DIR) 

def print_help():
    print("1.文件放入data目录")
    print("2.结果产出在result目录")


if __name__ == '__main__':
    # spilit2test()
    clear()
    main()













