import os
import math
import time
import numpy as np
import multiprocessing
from functools import cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory, Manager
from multiprocessing import Array

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

def Q_generator(q0, qmax, M):
    step = (qmax - q0) / M
    cur = q0
    while abs(cur - qmax) > (10 ** (-16)):
        # print(cur,qmax)
        yield cur
        cur += step

def test():
    return 0

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

    def dis(self, other):
        # cache一下加快避免重复计算
        # if self.atom_id in other.cache_dis_dict:
        #     return other.cache_dis_dict[self.atom_id]
        # if other.atom_id in self.cache_dis_dict:
        #     return self.cache_dis_dict[other.atom_id]

        dis = self.pos.dis(other.pos)
        # self.cache_dis_dict[other.atom_id] = dis
        # other.cache_dis_dict[self.atom_id] = dis

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
    def __init__(self,multithread):
        self.atom_list = []
        self.atom_count = 0
        self.box = None
        self.total_dis = 0
        self.frame_id = -1
        self.multithread = multithread
        self.K = 0

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

    def cal_init(self):
        # 确保输入是 numpy 数组
        positions = np.array(self.atom_list)

        # 使用广播机制计算两两点之间的向量差
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 3)

        # 计算欧几里得距离矩阵
        distance_matrix = np.linalg.norm(diff, axis=-1)  # (N, N)
        np.fill_diagonal(distance_matrix, 0)
        N = len(self.atom_list)
        K = 1 / ((N + 1) * (N + 1)) 
        self.K = K

        if not self.multithread:
            self.distance_matrix = distance_matrix
        else:
            self.distance_matrix = distance_matrix
        self.atom_list = []


    def cal_total_dis_worker(self, Q):
        # print("xxxx",Q)
        # 使用共享内存中的数据
        # np_array = np.frombuffer(frame.shared_array.get_obj(), dtype=np.float64).reshape(shape)
        # np_array = np.frombuffer(frame.shared_array.get_obj(), dtype=np.float64).reshape(shape)
        
        # 执行计算
        # distance_matrix = self.distance_matrix 
        total_dis = 0
        distance_matrix = self.distance_matrix * Q
        sin_distance_matrix = np.sin(distance_matrix)
        ratio_matrix = sin_distance_matrix / distance_matrix
        sum_ratio = np.nansum(ratio_matrix)
        total_dis = self.K * sum_ratio

        return total_dis

    def cal_total_dis(self, Q, shape = None):
        distance_matrix = self.distance_matrix * Q
        sin_distance_matrix = np.sin(distance_matrix)
        ratio_matrix = sin_distance_matrix / distance_matrix
        sum_ratio = np.nansum(ratio_matrix)
        
        total_dis = self.K * sum_ratio

        return total_dis



class AverageCalculator():
    def __init__(self, file_path, multithread):
        self.file_lines = file_line_generator(file_path)
        self.multithread = multithread
        self.frames = []
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
                self.frames.append(Frame(self.multithread))
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
        for frame in self.frames:
            frame.cal_init()

    # 单线程
    def cal_arvage(self, Q):
        frame_total = 0
        for frame in self.frames:
            frame_total = frame.cal_total_dis(Q)
        return frame_total / len(self.frames)

    def format_print_cal_result(self, Q , multithread, max_workers):
        # P = self.cal_arvage(Q)
        P = 0
        if multithread:
            P = self.cal_arvage_multithread(Q, max_workers)
        else:
            P = self.cal_arvage(Q) 
        return format(P, ".15e"),format(Q, ".15e")

    # 修改多线程的计算方法，使用共享内存
    def cal_arvage_multithread(self, Q, max_workers):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            features = {}
            futures = {executor.submit(frame.cal_total_dis_worker, Q): frame for frame in self.frames}
            # futures = {executor.submit(test): i for i in range(10)}
            frame_total = 0
            # 等待所有进程完成并更新结果
            for future in as_completed(futures):
                try:
                    frame_one = future.result()  # 捕获结果
                    frame_total += frame_one
                except Exception as e:
                    log_error("Error occurred: ", e)  # 处理可能的异常

            return frame_total 




# 获取当前文件所在的目录
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

DATA_DIR = os.path.join(CUR_DIR, "data")
RESULT_DIR = os.path.join(CUR_DIR, "result")

def save_result(file_name, result_line):
    file_path = os.path.join(RESULT_DIR, file_name.replace("atom","dat"))
    with open(file_path,"a") as f:
        f.write(result_line)

def save_result_pre(file_name):
    file_path = os.path.join(RESULT_DIR, file_name.replace("atom","dat"))
    with open(file_path,"w") as f:
        f.write("# Total structure factor for atom types: \n")
        f.write("# { 1(5) }in system. \n")
        f.write("# Q-values       P(Q)  \n")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


    file_names = []
    file_names.append("test.atom")
    # file_names.append("md100.atom")
    # file_names.append("md125.atom")
    # file_names.append("md150.atom")
    # file_names.append("md175.atom")

    # 参数定义
    begin_n, end_n = 1, 10
    L = 300
    q0, qmax, M = begin_n * 2 * math.pi / L,end_n * 2 * math.pi / L, end_n
    multithread = True
    max_workers = 5
    # 参数定义


    for file_name in file_names:
        file_path = os.path.join(DATA_DIR, file_name)
        ac = AverageCalculator(file_path, multithread)
        result = []
        count = 0
        save_result_pre(file_name)
        for Q in Q_generator(q0, qmax, M):
            count += 1
            log_info("cal_begin : " + str(count), int(time.time()))
            P,Q = ac.format_print_cal_result(Q, multithread, max_workers)
            result_line = "{}      {}\n".format(Q,P)
            # log_info(result_line)
            result.append(result_line)
            log_info("cal_end : " + str(count), int(time.time()))
            save_result(file_name, result_line)

                # 释放所有 frame 的共享内存
            # ac.cal_arvage_multithread(Q, max_workers)


def spilit2test():
    contentList = []
    with open("data/md100.atom","r") as f:
        for i in range(5009 * 5 - 1):
            contentList.append(f.readline())
    with open("data/test.atom","w") as f:
        f.write("".join(contentList))

def print_help():
    print("1.文件放入data目录")
    print("2.结果产出在result目录")

if __name__ == '__main__':
    print_help()
    main()
    # spilit2test()














