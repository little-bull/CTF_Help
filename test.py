import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

class Frame:
    def __init__(self, large_data):
        self.large_data = large_data  # 假设这里是一个大的 numpy 数组
        
    def cal_total_dis_worker(self, Q):
        # 假设你需要对 large_data 进行计算
        result = np.sum(self.large_data * Q)
        print(result)

def get_large_data():
    atom_list = [[1,1,1],[2,2,2]]
    # 确保输入是 numpy 数组
    positions = np.array(atom_list)

    # 使用广播机制计算两两点之间的向量差
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 3)

    # 计算欧几里得距离矩阵
    distance_matrix = np.linalg.norm(diff, axis=-1)  # (N, N)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

def process_frame(frame, Q):
    frame.cal_total_dis_worker(Q)

class AC:
    def __init__(self):
        self.frames = []
        for i in range(5):
            manager = multiprocessing.Manager()
            large_data = get_large_data()
            shared_data = manager.list(large_data)
            print(shared_data)
            self.frames.append(Frame(shared_data))

    def cal_all(self):
        # 使用 ProcessPoolExecutor 提交任务
        for Q in range(1,5):
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_frame, frame, Q): frame for frame in self.frames}



def main2():
    ac = AC()
    ac.cal_all()

def main1():
    # 创建一个大的数据（假设是一个 numpy 数组）
    large_data = np.random.rand(1000000)  # 例如，包含 100 万个元素
    
    # 使用 multiprocessing.Manager 来共享数据
    manager = multiprocessing.Manager()
    shared_data = manager.list(large_data)
    
    # 创建 Frame 实例
    frame = Frame(shared_data)
    
    # 使用 ProcessPoolExecutor 提交任务
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_frame, frame, Q): frame for Q in [1, 2, 3, 4]}
    
    # 等待所有任务完成
    for future in futures:
        future.result()

if __name__ == "__main__":
    # main1()
    main2()
