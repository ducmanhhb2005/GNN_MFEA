"""
#Skill Factor (hệ số chuyên môn) là chỉ số của task mà một cá thể có thứ hạng tốt nhất.
#Scalar Fitness (độ thích nghi vô hướng) là một con số thể hiện chất lượng của cá thể, được tính từ thứ hạng tốt nhất của nó. (1/ thứ hạng từ task tốt nhất của nó)

Main goal: 
Xếp hạng cá thể trên từng tác vụ, xác định cá thể mạnh ở tác vụ nào và tính điểm để chọn cá thể sống sang thế hệ tiếp theo
"""
from __future__ import annotations
import random
from .chromosome import Chromosome

#Chromosome ví dụ: Chromosome(priority=[3, 1, 2], edge_index=[1, 2, 1], skill=None, scores={0: 10.0, 1: 25.0}, scalar_fitness=0.0)
def assign_ranks(
    pop: list[Chromosome], #Population: danh sách các cá thể trong quần thể MFEA
    task_count: int,
    rng: random.Random | None = None, #Bộ sinh số ngẫu nhiên (random number generator) để xáo trộn thứ tự các cá thể khi có cùng chi phí trên một task. Nếu không cần xáo trộn, có thể để là None.
    initial: bool = False 
) -> None:
    """
    Xếp hạng các cá thể trong quần thể MFEA.

    Kết quả:
    - Gán skill factor cho từng cá thể.
    - Gán scalar fitness cho từng cá thể.

    Hàm thay đổi trực tiếp các đối tượng trong pop,
    nên không cần return danh sách mới.
    """

    # BƯỚC 1: Tạo nơi lưu thứ hạng của từng task.
    #
    # Ví dụ có 2 task:
    # ranks = [{}, {}]
    #
    # ranks[0]: lưu thứ hạng các cá thể trên Task 0
    # ranks[1]: lưu thứ hạng các cá thể trên Task 1
    #
    # Mỗi dict sẽ có dạng:
    # {chỉ_số_cá_thể: thứ_hạng}
    ranks = [dict() for _ in range(task_count)]

    # BƯỚC 2: Xếp hạng lần lượt trên từng task.
    #
    # range(2) tạo ra 0, 1.
    # Nghĩa là xét Task 0 trước, rồi Task 1.
    for task in range(task_count):

        # Lấy chỉ số các cá thể ĐÃ được đánh giá
        # trên task hiện tại.
        #
        # enumerate(pop) cho ra:
        # (0, cá_thể_0), (1, cá_thể_1), ...
        #
        # c.scores là dict lưu chi phí theo task.
        # Ví dụ: {0: 10.0, 1: 25.0}
        #
        # "task in c.scores" kiểm tra xem cá thể
        # có kết quả đánh giá trên task đó không.
        #
        # Ví dụ Task 0 có 3 cá thể đã đánh giá:
        # order = [0, 1, 2] #Vị trí index của các cá thể trong danh sách pop nếu cá thể đó có điểm trên task hiện tại
        order = [
            i
            for i, c in enumerate(pop)
            if c.scores is not None
            and task in c.scores
        ]

        # Nếu có bộ sinh số ngẫu nhiên rng:
        # xáo trộn order trước khi sắp xếp.
        #
        # Mục đích: nếu hai cá thể có cùng chi phí,
        # thứ tự giữa chúng có thể được phá hòa
        # một cách ngẫu nhiên.
        if rng is not None:
            rng.shuffle(order)

        # Sắp xếp các cá thể theo chi phí TĂNG DẦN.
        #
        # pop[i].scores[task] là chi phí cá thể i
        # trên task đang xét.
        #
        # Vì IDPC-DU cần CỰC TIỂU chi phí:
        # chi phí thấp hơn => xếp trước.
        #
        # Ví dụ:
        # cá thể 0: cost = 10
        # cá thể 1: cost = 15
        # cá thể 2: cost = 12
        #
        # Sau sort: order = [0, 2, 1]
        order.sort(key=lambda i: pop[i].scores[task])

        # Gán thứ hạng cho các cá thể.
        #
        # enumerate(order, 1):
        # Bắt đầu đếm rank từ 1, không phải 0.
        #
        # Nếu order = [0, 2, 1]:
        # ranks[task][0] = 1
        # ranks[task][2] = 2
        # ranks[task][1] = 3
        for rank, idx in enumerate(order, 1):
            ranks[task][idx] = rank

    # BƯỚC 3: Tìm skill factor và scalar fitness
    # cho TỪNG cá thể.
    for i, c in enumerate(pop):

        # Tạo danh sách các cặp:
        # (thứ_hạng, task)
        #
        # Chỉ lấy task mà cá thể i đã được đánh giá.
        #
        # Ví dụ cá thể 0:
        # Task 0: rank = 1
        # Task 1: rank = 3
        #
        # candidates = [(1, 0), (3, 1)]
        candidates = [
            (ranks[t][i], t)
            for t in range(task_count)
            if i in ranks[t]
        ]

        # Nếu cá thể không có kết quả ở bất kỳ task nào,
        # không thể tính thứ hạng cho nó.
        #
        # raise ValueError: báo lỗi và dừng hàm.
        if not candidates:
            raise ValueError(
                "Individual was not evaluated on any task"
            )

        # Tìm thứ hạng NHỎ NHẤT của cá thể.
        #
        # Ví dụ:
        # candidates = [(1, 0), (3, 1)]
        #
        # min_rank = 1
        #
        # Rank nhỏ nghĩa là xếp hạng tốt.
        min_rank = min(
            rank for rank, _ in candidates
        )

        # Tìm các task mà cá thể đạt min_rank.
        #
        # Ví dụ:
        # candidates = [(1, 0), (3, 1)]
        # min_rank = 1
        # tied = [0]
        #
        # Nếu candidates = [(2, 0), (2, 1)]
        # thì tied = [0, 1]
        #
        # Dấu "_" nghĩa là biến không cần dùng.
        tied = [
            t
            for rank, t in candidates
            if rank == min_rank
        ]

        # Gán skill factor: task có rank tốt nhất.
        #
        # Nếu có nhiều task cùng rank tốt nhất
        # và có rng, chọn ngẫu nhiên một task.
        #
        # Nếu không, lấy task đầu tiên trong tied.
        c.skill = (
            rng.choice(tied)
            if rng is not None and len(tied) > 1
            else tied[0]
        )

        # Tính Scalar Fitness.
        #
        # Công thức:
        # scalar_fitness = 1 / min_rank
        #
        # Rank 1 => fitness = 1.0
        # Rank 2 => fitness = 0.5
        # Rank 3 => fitness = 0.333...
        #
        # Fitness CÀNG LỚN => cá thể càng tốt.
        c.scalar_fitness = 1.0 / min_rank

def assign_ranks(pop: list[Chromosome], task_count: int,
                 rng: random.Random | None = None, initial: bool = False) -> None:
    """MFEA scalar fitness = 1 / min factorial rank among EVALUATED tasks.

    Offspring are evaluated only on inherited skill, while original individuals
    are evaluated on all tasks. Missing task costs are NOT ranked as valid
    entries; assigning them a finite rank produces fictitious fitness.
    """
    ranks = [dict() for _ in range(task_count)]
    for task in range(task_count):
        order = [i for i, c in enumerate(pop) if c.scores is not None and task in c.scores]
        if rng is not None:
            rng.shuffle(order)  # randomly break equal fitness ties (incl. +inf)
        order.sort(key=lambda i: pop[i].scores[task])
        for rank, idx in enumerate(order, 1):
            ranks[task][idx] = rank
    for i, c in enumerate(pop):
        candidates = [(ranks[t][i], t) for t in range(task_count) if i in ranks[t]]
        if not candidates:
            raise ValueError("Individual was not evaluated on any task")
        # In the original MFEA, skill factor is argmin of factorial ranks.
        min_rank = min(rank for rank, _ in candidates)
        tied = [t for rank, t in candidates if rank == min_rank]
        c.skill = rng.choice(tied) if rng is not None and len(tied) > 1 else tied[0]
        c.scalar_fitness = 1.0 / min_rank
