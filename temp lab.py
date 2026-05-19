Создано
с
помощью
Perplexity
import numpy as np
from PIL import Image
import math
import os
import concurrent.futures
from collections import Counter
import heapq

QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

QC = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)


def create_dct_matrix(size):
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == 0:
                matrix[i, j] = 1 / np.sqrt(size)
            else:
                matrix[i, j] = np.sqrt(2 / size) * np.cos((2 * j + 1) * i * np.pi / (2 * size))
    return matrix


DCT_MATRIX_8 = create_dct_matrix(8)
IDCT_MATRIX_8 = DCT_MATRIX_8.T


def generate_zigzag_order(size=8):
    order = []
    for d in range(2 * size - 1):
        if d % 2 == 0:
            for i in range(d + 1):
                j = d - i
                if i < size and j < size:
                    order.append((i, j))
        else:
            for i in range(d, -1, -1):
                j = d - i
                if i < size and j < size:
                    order.append((i, j))
    return order


ZIGZAG_ORDER_8 = generate_zigzag_order(8)


def zigzag_traverse(matrix, order=ZIGZAG_ORDER_8):
    return np.array([matrix[i, j] for i, j in order], dtype=np.float32)


def zigzag_reverse(sequence, order=ZIGZAG_ORDER_8, size=8):
    matrix = np.zeros((size, size), dtype=np.float32)
    for idx, (i, j) in enumerate(order):
        if idx < len(sequence):
            matrix[i, j] = sequence[idx]
    return matrix


class RLESymbol:
    def __init__(self, symbol_type, value=None, count=1):
        self.type = symbol_type
        self.value = value
        self.count = count

    def __repr__(self):
        if self.type == 'EOB':
            return 'EOB'
        elif self.type == 'ZERO_RUN':
            return f'0x{self.count}'
        else:
            return f'{self.value}x{self.count}' if self.count > 1 else f'{self.value}'

    def __hash__(self):
        return hash((self.type, self.value, self.count))

    def __eq__(self, other):
        return (isinstance(other, RLESymbol) and
                self.type == other.type and
                self.value == other.value and
                self.count == other.count)


def rle_encode(sequence):
    encoded = []
    i = 0
    sequence = np.asarray(sequence, dtype=int)

    while i < len(sequence):
        if sequence[i] == 0:
            zero_count = 0
            while i < len(sequence) and sequence[i] == 0 and zero_count < 255:
                zero_count += 1
                i += 1
            encoded.append(RLESymbol('ZERO_RUN', count=zero_count))
        else:
            encoded.append(RLESymbol('VALUE', value=int(sequence[i])))
            i += 1

    encoded.append(RLESymbol('EOB'))
    return encoded


def rle_decode(encoded_sequence):
    decoded = []
    for symbol in encoded_sequence:
        if symbol.type == 'ZERO_RUN':
            decoded.extend([0] * symbol.count)
        elif symbol.type == 'VALUE':
            decoded.append(symbol.value)
    return np.array(decoded, dtype=np.float32)


class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(symbols):
    freq_map = Counter(symbols)

    heap = [HuffmanNode(symbol=sym, freq=freq)
            for sym, freq in freq_map.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(freq=left.freq + right.freq,
                             left=left, right=right)
        heapq.heappush(heap, parent)

    return heap[0] if heap else None


def generate_huffman_codes(tree, prefix='', codes=None):
    if codes is None:
        codes = {}

    if tree is None:
        return codes

    if tree.symbol is not None:
        codes[tree.symbol] = prefix if prefix else '0'
    else:
        generate_huffman_codes(tree.left, prefix + '0', codes)
        generate_huffman_codes(tree.right, prefix + '1', codes)

    return codes


def huffman_encode(rle_sequence):
    tree = build_huffman_tree(rle_sequence)
    codes = generate_huffman_codes(tree)

    encoded_bits = ''.join(codes[symbol] for symbol in rle_sequence)

    return {
        'codes': codes,
        'encoded_bits': encoded_bits,
        'total_bits': len(encoded_bits),
        'tree': tree
    }


def calculate_entropy(rle_sequence):
    freq_map = Counter(rle_sequence)
    n = len(rle_sequence)

    entropy = 0.0
    for count in freq_map.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def rgb_to_ycbcr(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=2).astype(np.float32)


def ycbcr_to_rgb(ycbcr):
    y, cb, cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)


def dct2(block):
    return DCT_MATRIX_8 @ block @ DCT_MATRIX_8.T


def idct2(block):
    return IDCT_MATRIX_8 @ block @ IDCT_MATRIX_8.T


def quantize(block, quantization_table, quality=50):
    scale_factor = 5000 / quality if quality < 50 else 200 - 2 * quality
    scaled_quant_table = np.floor((quantization_table * scale_factor + 50) / 100)
    scaled_quant_table[scaled_quant_table < 1] = 1
    scaled_quant_table[scaled_quant_table > 255] = 255

    quantized = np.round(block / scaled_quant_table)
    return quantized, scaled_quant_table


def dequantize(block, quantization_table):
    return block * quantization_table


def process_block_full(args):
    block, quantization_table, quality, channel_type = args

    if channel_type == 'Y':
        block = block - 128

    dct_block = dct2(block)
    quantized_block, quant_table = quantize(dct_block, quantization_table, quality)

    zigzag_seq = zigzag_traverse(quantized_block)

    rle_seq = rle_encode(zigzag_seq)

    huffman_result = huffman_encode(rle_seq)

    rle_decoded = rle_decode(rle_seq)
    quantized_restored = zigzag_reverse(rle_decoded)
    dequantized_block = dequantize(quantized_restored, quant_table)
    idct_block = idct2(dequantized_block)

    if channel_type == 'Y':
        idct_block = idct_block + 128

    return {
        'restored_block': idct_block,
        'rle_sequence': rle_seq,
        'huffman_result': huffman_result,
        'original_size': len(zigzag_seq) * 8,
        'compressed_size': huffman_result['total_bits']
    }


def process_channel_full(channel, quantization_table, quality, channel_type):
    height, width = channel.shape
    padded_height = ((height + 7) // 8) * 8
    padded_width = ((width + 7) // 8) * 8
    padded_channel = np.zeros((padded_height, padded_width), dtype=np.float32)
    padded_channel[:height, :width] = channel

    blocks = []
    positions = []
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            block = padded_channel[i:i + 8, j:j + 8].copy()
            blocks.append((block, quantization_table, quality, channel_type))
            positions.append((i, j))

    processed_blocks = []
    compression_stats = {
        'total_rle_symbols': 0,
        'total_huffman_bits': 0,
        'total_original_bits': 0,
        'all_rle_sequences': []
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_block_full, blocks))

    for result in results:
        processed_blocks.append(result['restored_block'])
        compression_stats['total_rle_symbols'] += len(result['rle_sequence'])
        compression_stats['total_huffman_bits'] += result['huffman_result']['total_bits']
        compression_stats['total_original_bits'] += result['original_size']
        compression_stats['all_rle_sequences'].extend(result['rle_sequence'])

    processed_channel = np.zeros_like(padded_channel)
    for (i, j), block in zip(positions, processed_blocks):
        processed_channel[i:i + 8, j:j + 8] = block

    return processed_channel[:height, :width], compression_stats


def jpeg_encode_full(image, quality=50):
    ycbcr = rgb_to_ycbcr(image.astype(np.float32))

    y_channel = ycbcr[:, :, 0]
    cb_channel = ycbcr[:, :, 1]
    cr_channel = ycbcr[:, :, 2]

    processed_y, stats_y = process_channel_full(y_channel, QY, quality, 'Y')
    processed_cb, stats_cb = process_channel_full(cb_channel, QC, quality, 'C')
    processed_cr, stats_cr = process_channel_full(cr_channel, QC, quality, 'C')

    total_stats = {
        'original_bits': stats_y['total_original_bits'] + stats_cb['total_original_bits'] + stats_cr[
            'total_original_bits'],
        'huffman_bits': stats_y['total_huffman_bits'] + stats_cb['total_huffman_bits'] + stats_cr['total_huffman_bits'],
        'rle_symbols': stats_y['total_rle_symbols'] + stats_cb['total_rle_symbols'] + stats_cr['total_rle_symbols'],
        'entropy': calculate_entropy(
            stats_y['all_rle_sequences'] + stats_cb['all_rle_sequences'] + stats_cr['all_rle_sequences'])
    }

    processed_ycbcr = np.stack([processed_y, processed_cb, processed_cr], axis=2)
    return ycbcr_to_rgb(processed_ycbcr), total_stats


def print_compression_report(original_size_bytes, compressed_stats, quality):
    print("\n" + "=" * 70)
    print(f"JPEG COMPRESSION REPORT (Quality: {quality})")
    print("=" * 70)

    original_bits = compressed_stats['original_bits']
    huffman_bits = compressed_stats['huffman_bits']

    print(f"\nOriginal Size:                {original_size_bytes} bytes ({original_size_bytes * 8} bits)")
    print(f"After DCT + Quantization:     {original_bits} bits")
    print(f"RLE Symbols:                  {compressed_stats['rle_symbols']}")
    print(f"After Huffman Encoding:       {huffman_bits} bits")
    print(f"Shannon Entropy:              {compressed_stats['entropy']:.3f} bits/symbol")

    compression_ratio = huffman_bits / (original_size_bytes * 8) * 100
    savings = 100 - compression_ratio

    print(f"\nOverall Compression Ratio:    {compression_ratio:.1f}% of original")
    print(f"Data Savings:                 {savings:.1f}%")
    print("=" * 70 + "\n")


def main():
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    input_path = os.path.join(desktop_path, "Без имени-1.jpg")
    output_low_path = os.path.join(desktop_path, "output_low_q10.jpg")
    output_mid_path = os.path.join(desktop_path, "output_mid_q50.jpg")
    output_high_path = os.path.join(desktop_path, "output_high_q90.jpg")

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        print("Creating test image...")

        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        rgb_image = test_image
        original_size = test_image.shape[0] * test_image.shape[1] * test_image.shape[2]
    else:
        try:
            input_image = Image.open(input_path)
            rgb_image = np.array(input_image.convert('RGB'))
            original_size = rgb_image.shape[0] * rgb_image.shape[1] * rgb_image.shape[2]
            print(f"Loaded image: {rgb_image.shape}")
        except Exception as e:
            print(f"Error: {e}")
            return

    qualities = [10, 50, 90]

    for quality in qualities:
        print(f"\nProcessing quality {quality}...")

        compressed_image, stats = jpeg_encode_full(rgb_image, quality=quality)

        print_compression_report(original_size, stats, quality)

        if quality == 10:
            output_path = output_low_path
        elif quality == 50:
            output_path = output_mid_path
        else:
            output_path = output_high_path

        Image.fromarray(compressed_image).save(output_path)
        print(f"Saved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()