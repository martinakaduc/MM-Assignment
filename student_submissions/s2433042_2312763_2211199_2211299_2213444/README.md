# 2D Cutting Stock Problem

## Mô tả bài toán
Bài toán cắt 2D (2D Cutting Stock Problem) là một bài toán tối ưu trong đó mục tiêu là cắt các sản phẩm có kích thước và số lượng cụ thể từ các tấm nguyên liệu (phôi) sao cho giảm thiểu lãng phí và tối ưu hóa việc sử dụng vật liệu.

Chương trình sử dụng thư viện `gym_cutting_stock` để mô phỏng môi trường và áp dụng thuật toán dựa trên chính sách để thực hiện bài toán.

---

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường
Để chạy chương trình, bạn cần cài đặt các thư viện sau:

```bash
pip install gymnasium
pip install gym_cutting_stock
pip install numpy
```

Ngoài ra, đảm bảo rằng file chính sách (`Policy2433042_2312763_2211199_2211299_2213444`) được đặt trong thư mục `student_submissions` như cấu trúc sau:

```
project_directory/
├── student_submissions/
│   └── policy2433042_2312763_2211199_2211299_2213444.py
├── main.py  # File chính của chương trình
```

---

### 2. Chạy chương trình

Để chạy chương trình, thực hiện lệnh sau trong terminal hoặc command line:

```bash
python main.py
```

### 3. Cấu trúc chương trình

#### Tạo môi trường
Môi trường được tạo từ thư viện `gym_cutting_stock` với thông số:

```python
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Để hiển thị trực quan, có thể tắt bằng cách comment dòng này.
)
```

#### Chính sách (Policy)
Chương trình sử dụng lớp chính sách (`Policy2433042_2312763_2211199_2211299_2213444`) để đưa ra hành động tối ưu dựa trên trạng thái hiện tại của môi trường. 

Ví dụ:

```python
policy2433042_2312763_2211199_2211299_2213444 = Policy2433042_2312763_2211199_2211299_2213444(policy_id=1)
```

#### Vòng lặp chính
Chương trình thực hiện vòng lặp qua 200 bước, mỗi bước:
1. Gọi hành động từ chính sách.
2. Thực hiện hành động trên môi trường bằng lệnh `env.step(action)`.
3. In thông tin về trạng thái hiện tại của môi trường.
4. Nếu môi trường kết thúc (terminated hoặc truncated), reset trạng thái ban đầu.

#### Kết thúc môi trường
Khi kết thúc, môi trường được đóng lại bằng lệnh:
```python
env.close()
```

---

### 4. Đầu ra của chương trình

Chương trình sẽ in các thông tin sau trong quá trình thực thi:
- **Danh sách sản phẩm ban đầu**: Kích thước và số lượng từng sản phẩm cần cắt.
- **Danh sách phôi ban đầu**: Kích thước các tấm phôi.
- **Thông tin trạng thái môi trường sau mỗi bước**: Bao gồm phần thưởng và trạng thái hiện tại của phôi và sản phẩm.

---

## Cấu trúc file chính sách (Policy)
File `Policy2433042_2312763_2211199_2211299_2213444.py` cần định nghĩa một lớp `Policy2433042_2312763_2211199_2211299_2213444` với các phương thức chính sau:

- `__init__(self, policy_id: int)`: Khởi tạo chính sách.
- `get_action(self, observation, info)`: Trả về hành động dựa trên trạng thái hiện tại (`observation`) và thông tin môi trường (`info`).

---

## Lưu ý
1. Đảm bảo chỉ sửa đổi trong thư mục `student_submissions` để tránh bị từ chối PR.
2. Chỉ sử dụng chính sách trong phạm vi mã của bạn, không thay đổi cấu trúc thư viện gốc.
3. Đảm bảo reset môi trường sau khi kết thúc một tập (episode).

---

## Liên hệ
Nếu có bất kỳ vấn đề gì khi sử dụng hoặc chạy chương trình, vui lòng liên hệ thông qua nhóm hoặc người hướng dẫn.
