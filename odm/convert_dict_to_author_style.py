import pickle

# 读取你之前生成的 chinese_dict.txt
with open('E:/ODM/mmocr/data/chinese_dict.txt', 'r', encoding='utf-8') as f:
    chars = f.read().splitlines()

# 将字符转换为 Unicode 编码列表 (作者代码 data.py 第 64 行逻辑)
char_indices = [ord(c) for c in chars]

# 保存为 pickle 文件
with open('E:/ODM/chinese_handwriting_char_dict.pkl', 'wb') as f:
    pickle.dump(char_indices, f)