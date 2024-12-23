import os
import math
import time
import numpy as np
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
    def __init__(self, QS, chain_count):
        self.atom_count = 0
        self.box = None
        self.total_dis = 0
        self.frame_id = -1
        self.QS = QS
        self.chain_count = chain_count
        self.chain_atom_list = [[] for i in range(chain_count)]

    def set_frame_id(self, frame_id):
        self.frame_id = frame_id

    def set_atom_count(self, atom_count):
        self.atom_count = atom_count
        self.chain_atom_count = self.atom_count // self.chain_count
        N = self.chain_atom_count
        self.K = 1 / (N * N)

        # print(self.atom_count,self.chain_atom_count)

    def get_atom_count(self):
        return self.atom_count
    
    def add_atom(self, atom, idx):
        # 重定位
        idx = int(idx)
        old_pos = atom.get_pos()
        new_pos = self.box.mod_in_box(old_pos)
        atom.set_pos(new_pos)

        chain_index = (idx - 1) // self.chain_atom_count
        atom_index = (idx - 1) % self.chain_atom_count
        # print(chain_index,atom_index)
        self.chain_atom_list[chain_index].append([new_pos.x, new_pos.y, new_pos.z])

    def set_box(self, box):
        self.box = box

    def cal_with_Qs(self):
        # # 确保输入是 numpy 数组
        # positions = np.array(self.atom_list)
        # # 使用广播机制计算两两点之间的向量差
        # diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 3)
        # # 计算欧几里得距离矩阵
        # distance_matrix = np.linalg.norm(diff, axis=-1)  # (N, N)
        # # 排除自身距离（对角线元素为 0）
        # np.fill_diagonal(distance_matrix, 0)

        Q_count = len(self.QS)
        result = [0] * Q_count
        for chain in self.chain_atom_list:
            # 确保输入是 numpy 数组
            positions = np.array(chain)
            # 使用广播机制计算两两点之间的向量差
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 3)
            # 计算欧几里得距离矩阵
            distance_matrix = np.linalg.norm(diff, axis=-1)  # (N, N)
            # 排除自身距离（对角线元素为 0）
            np.fill_diagonal(distance_matrix, 0)
            for i in range(Q_count):
                Q = self.QS[i]
                new_distance_matrix = distance_matrix * Q
                sin_distance_matrix = np.sin(new_distance_matrix)
                ratio_matrix = sin_distance_matrix / distance_matrix 
                sum_ratio = np.nansum(ratio_matrix) 
                total_dis = self.K * sum_ratio / Q
                # result.append(total_dis)
                result[i] += total_dis
        new_result = [r / self.chain_count for r in result]
        return new_result

class AverageCalculator():
    def __init__(self, file_path, max_workers, QS, chain_count):
        self.file_lines = file_line_generator(file_path)
        self.max_workers = max_workers
        self.frames = []
        self.QS = QS
        self.chain_count = chain_count
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
                self.frames.append(Frame(self.QS[:],self.chain_count))
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
                self.frames[-1].add_atom(atom,atom_id)
                # print(atom.atom_id,"add__________")

            #下一行
            line_index += 1

            #重新添加下一个frame
            if line_index == self.frames[-1].get_atom_count() + 9:
                line_index = 0
        # print(self.frames[-1].chain_atom_list)

    def cal_frames(self):
        if self.max_workers > 1:
            return self.cal_arvage_multprogress()
        else:
            return self.cal_arvage()

    def cal_arvage_multprogress(self):
        print("cal_arvage_multprogress")
        Q_count = len(self.QS)
        Q_result = np.zeros((Q_count,))
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:  # 创建线程池
            future_to_frame = {executor.submit(frame.cal_with_Qs): frame for frame in self.frames}
            for future in as_completed(future_to_frame):
                frame_QS = future.result()
                Q_result += frame_QS
                log_info("finish : ",future_to_frame[future].frame_id)
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

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


    file_names = []
    file_names.append("test.atom")
    # file_names.append("md100.atom")
    # file_names.append("md125.atom")
    # file_names.append("md150.atom")
    # file_names.append("md175.atom")
    L = 300
    begin, end = 1, 100
    q0, qmax, M = begin * 2 * math.pi / L, end * 2 * math.pi / L, end
    max_workers = 4
    chain_count = 50
    for file_name in file_names:
        begin_time = int(time.time())
        print("begin", begin_time)
        QS = list(np.linspace(q0, qmax, M))
        file_path = os.path.join(DATA_DIR,file_name)
        ac = AverageCalculator(file_path, max_workers, QS, chain_count)
        QP = ac.format_print_cal_result()
        save_result(file_name,QP)
        end_time = int(time.time())
        print("end", end_time)
        print(end_time - begin_time)

def spilit2test():
    contentList = []
    with open("data/md100.atom","r") as f:
        for i in range(5009 * 1000 - 1):
            contentList.append(f.readline())
    with open("data/test.atom","w") as f:
        f.write("".join(contentList))

def print_help():
    print("1.文件放入data目录")
    print("2.结果产出在result目录")


if __name__ == '__main__':
    # spilit2test()
    main()













