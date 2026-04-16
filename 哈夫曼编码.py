import heapq
import os
import json

class Node:
    def __init__(self, char=None, freq=0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    #堆排序依据
    def __lt__(self, other):
        return self.freq < other.freq



def count_frequency(text):
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return freq


#构建哈夫曼树
def build_huffman_tree(freq_dict):
    heap = [Node(ch, freq) for ch, freq in freq_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        parent = Node(None, left.freq + right.freq)
        parent.left = left
        parent.right = right

        heapq.heappush(heap, parent)

    return heap[0]


#生成编码表
def generate_codes(root):
    codes = {}

    def dfs(node, code):
        if node is None:
            return
        if node.char is not None:
            codes[node.char] = code
        dfs(node.left, code + "0")
        dfs(node.right, code + "1")

    dfs(root, "")
    return codes


#编码文本
def encode_text(text, codes):
    return "".join(codes[ch] for ch in text)


# 二进制写入（补齐8位）
def pad_encoded_text(encoded_text):
    extra_bits = 8 - len(encoded_text) % 8
    for _ in range(extra_bits):
        encoded_text += "0"

    padded_info = "{0:08b}".format(extra_bits)
    return padded_info + encoded_text


def get_byte_array(padded_text):
    b = bytearray()
    for i in range(0, len(padded_text), 8):
        byte = padded_text[i:i+8]
        b.append(int(byte, 2))
    return b


# 压缩文件
def compress(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    freq = count_frequency(text)
    root = build_huffman_tree(freq)
    codes = generate_codes(root)

    encoded_text = encode_text(text, codes)
    padded_text = pad_encoded_text(encoded_text)
    byte_array = get_byte_array(padded_text)

    # 保存压缩数据
    with open(output_file, "wb") as f:
        f.write(bytes(byte_array))

    # 保存编码表
    with open(output_file + ".json", "w", encoding="utf-8") as f:
        json.dump(codes, f, ensure_ascii=False)

    print("压缩完成！")


#解码辅助
def remove_padding(padded_text):
    padded_info = padded_text[:8]
    extra_bits = int(padded_info, 2)

    encoded_text = padded_text[8:]
    return encoded_text[:-extra_bits]


def decode_text(encoded_text, codes):
    reverse_codes = {v: k for k, v in codes.items()}

    current_code = ""
    decoded_text = ""

    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_codes:
            decoded_text += reverse_codes[current_code]
            current_code = ""

    return decoded_text


#解压文件
def decompress(input_file, output_file):
    with open(input_file, "rb") as f:
        bit_string = ""
        byte = f.read(1)
        while byte:
            byte = ord(byte)
            bit_string += f"{byte:08b}"
            byte = f.read(1)

    encoded_text = remove_padding(bit_string)

    # 读取编码表
    with open(input_file + ".json", "r", encoding="utf-8") as f:
        codes = json.load(f)

    decoded_text = decode_text(encoded_text, codes)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(decoded_text)

    print("解压完成！")


#文件大小对比
def compare_size(original, compressed):
    original_size = os.path.getsize(original)
    compressed_size = os.path.getsize(compressed)

    print(f"原文件: {original_size} bytes")
    print(f"压缩后: {compressed_size} bytes")
    print(f"压缩率: {compressed_size / original_size:.2%}")


#主程序测试
if __name__ == "__main__":
    input_file = "input.txt"
    compressed_file = "compressed.bin"
    output_file = "output.txt"

    compress(input_file, compressed_file)
    decompress(compressed_file, output_file)
    compare_size(input_file, compressed_file)