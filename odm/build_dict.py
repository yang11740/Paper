import os

def generate_dict(input_txt, output_dict):
    # 1. 读取文本内容
    if not os.path.exists(input_txt):
        print(f"错误：找不到文件 {input_txt}")
        return

    with open(input_txt, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. 提取唯一字符并去重
    # 使用 set 自动去重
    unique_chars = set(content)

    # 3. 过滤掉不需要的字符
    # 过滤换行符、制表符、全角/半角空格
    exclude = {'\n', '\r', '\t', ' ', '\u3000'}
    filtered_chars = [c for c in unique_chars if c not in exclude]

    # 4. 排序 (按照 Unicode 编码排序，保证索引稳定)
    filtered_chars.sort()

    # 5. 写入目标文件
    os.makedirs(os.path.dirname(output_dict), exist_ok=True)
    with open(output_dict, 'w', encoding='utf-8') as f:
        for char in filtered_chars:
            f.write(f"{char}\n")

    print(f"字典生成成功！")
    print(f"源文件：{input_txt}")
    print(f"目标字典：{output_dict}")
    print(f"字符总数（不含空格）：{len(filtered_chars)}")

if __name__ == "__main__":
    # 根据你的实际路径修改
    generate_dict('脂砚斋评石头记.txt', 'E:/ODM/mmocr/data/chinese_dict.txt')