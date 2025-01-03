# Huffman Encoding
from collections import Counter


def get_freq_dict(src):
    if type(src) is not list:
        flattened = src.flatten()
    else:
        flattened = src
    freq = Counter(flattened)
    freq = {str(k): v for k, v in freq.items()}
    return freq


# Creating tree nodes
class NodeTree(object):

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

    def __str__(self):
        return '%s_%s' % (self.left, self.right)


# Main function implementing huffman coding
def huffman_code_tree(node, left=True, binString=''):
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, True, binString + '0'))
    d.update(huffman_code_tree(r, False, binString + '1'))
    return d


def build_huffman_tree(freq_dict):
    freq_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    nodes = freq_dict
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
        # print(nodes)

    return nodes[0][0]


def generate_huffman_codes(tree_root):
    return huffman_code_tree(tree_root)

def get_huffman_codes(freq, extar_bits=0):
    root = build_huffman_tree(freq)
    huffmanCode = generate_huffman_codes(root)

    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    encoded_bits = 0
    for (char, frequency) in sorted_freq:
        print(f"Character: {char:>3}, Code: {huffmanCode[char]:>17}, Length of Code: {len(huffmanCode[char]):>2}, Frequency: {frequency:>5}")
        encoded_bits += ((len(huffmanCode[char])+extar_bits) * frequency)
    return root, huffmanCode, encoded_bits
