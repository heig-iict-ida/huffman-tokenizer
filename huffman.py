import json
import heapq
import nltk
import io
import sys

from sacremoses import MosesTruecaser, MosesTokenizer, MosesPunctNormalizer, MosesDetokenizer, MosesDetruecaser
from collections import defaultdict

# Special symbols used as markers in embedded tokenization:
# 0x2400 (Null): marks an unknown token (word not in tree)
# 0x2420 (Space): marks a space in the original text
CTRL_UNK = chr(0x2400)
CTRL_SPACE = chr(0x2420)

# Ensure text stream
def txtStream(fop, mode="rt"):
	if isinstance(fop, io.TextIOBase):
			return fop
	elif isinstance(fop, str):
		return open(fop, mode, encoding="utf8")


# https://stackoverflow.com/a/5921708
def intersperse(lst, item):
	result = [item] * (len(lst) * 2 - 1)
	result[0::2] = lst
	return result

# Original code taken from: https://bhrigu.me/post/huffman-coding-python-implementation/
# Create a custom class HuffmanCoding that will contain all the necessary elements
class HuffmanCoding:
	def __init__(self, excluded, symbols):
		# The Huffman tree
		self.tokens = defaultdict(int)
		self.heap = []
		self.codes = {}
		# Used to decode, swap the keys and values so that a encoded input gives its decoded version
		self.reverse_mapping = {}
		self.excluded = set(excluded + [CTRL_UNK, CTRL_SPACE])
		self.symbols = symbols

	# Custom class used to represent a Huffman tree node
	class HeapNode:
		def __init__(self, word, freq):
			self.word = word
			self.freq = freq
			self.children = []

		# Defining comparators less_than and equals
		def __lt__(self, other):
			return self.freq < other.freq

		def __eq__(self, other):
			if (other == None):
				return False
			if (not isinstance(other, HeapNode)):
				return False
			return self.freq == other.freq

	# Create the heap from the frequency list
	def make_heap(self, frequency):
		for key in frequency:
			node = self.HeapNode(key, frequency[key])
			heapq.heappush(self.heap, node)

	# Merge together the d smallest nodes in the tree
	def merge_nodes(self):
		# Stop the merges when only one node remains -> it is the tree root
		while (len(self.heap) > 1):
			nodes = []
			tot_freq = 0
			for i in range(len(self.symbols)):
				if(len(self.heap) == 0):
					break

				# Pop and return the smallest item from the heap, we return the element with the smallest frequency, so it can be something else than the previously created node
				nodes.append(heapq.heappop(self.heap))
				tot_freq += nodes[i].freq

			# Create a new node with the combined frequency of it children as frequency and the list of popped nodes as children
			merged = self.HeapNode(None, tot_freq)
			for i in range(len(nodes)):
				merged.children.append(nodes[i])

			# Push the node in the heap so it can be used again
			heapq.heappush(self.heap, merged)

	def make_codes_helper(self, root, current_code):
		if (root == None):
			return

		# A word value of None means that the current node is not a final leaf
		if (root.word != None):
			self.codes[root.word] = current_code
			self.reverse_mapping[current_code] = root.word
			return

		# Call the function recursively on all childrens to walk the tree
		for i in range(len(root.children)):
			self.make_codes_helper(root.children[i], current_code + self.symbols[i])

	def make_codes(self):
		root = heapq.heappop(self.heap)
		current_code = ""
		self.make_codes_helper(root, current_code)

# This method compresses and tokenizes the input lines
	def compress(self, fop_in, fop_out, tokenized=True):
		stream_in = txtStream(fop_in)
		stream_out = txtStream(fop_out, "wt")

		while True:
			line = stream_in.readline()
			if line == "": break
			line = line.strip()
			if line == "": continue

			if tokenized:
				line = line.replace(" ", "")
				tokens = line.split(CTRL_SPACE)
			else:
				tokens = nltk.word_tokenize(line)

			compressed = []
			for tok in tokens:
				if tok in self.excluded:
					compressed.append(tok)
				elif tok in self.codes:
					compressed.append(self.codes[tok])
				else:
					compressed.append(CTRL_UNK)

			stream_out.write(str.join(CTRL_SPACE, compressed) + "\n")

	def decompress(self, fop_in, fop_out):
		stream_in = txtStream(fop_in)
		stream_out = txtStream(fop_out, "wt")

		while True:
			line = stream_in.readline()
			if line == "": break
			line = line.strip()

			decompressed = []
			for tok in line.split(CTRL_SPACE):
				if tok in self.excluded:
					decompressed.append(tok)
				elif tok in self.reverse_mapping:
					decompressed.append(self.reverse_mapping[tok])
				else:
					decompressed.append(CTRL_UNK)

			stream_out.write(str.join(" " + CTRL_SPACE + " ", decompressed) + "\n")

	def loadVocab(self, fop):
		stream = txtStream(fop)
		self.reverse_mapping = json.load(stream)

		for i, (code, word) in enumerate(self.reverse_mapping.items()):
			self.codes[word] = code

	def ingest(self, fop, tokenized=True):
		stream = txtStream(fop)

		while True:
			line = stream.readline()
			if line == "": break
			line = line.strip()

			if tokenized:
				line = line.replace(" ", "")
				tokens = line.split(CTRL_SPACE)
			else:
				tokens = nltk.word_tokenize(line)

			for t in tokens:
				self.tokens[t] += 1

	def digest(self):
		byFreq = [(k, self.tokens[k]) for k in sorted(self.tokens, key=lambda x: self.tokens[x], reverse=True)]
		frequency = dict(byFreq)

		# Remove the excluded symbols from the frequency dict
		for i in self.excluded:
			if i in frequency:
				frequency.pop(i)

		self.make_heap(frequency)
		self.merge_nodes()
		self.make_codes()

	def saveMapping(self, fop):
		stream = txtStream(fop, "wt")
		json.dump(self.reverse_mapping, stream, ensure_ascii=False)

class Tokenizer():
	def __init__(self, mosesLang="en", mosesCaseModel=None):
		self.mosesNorm = MosesPunctNormalizer()
		self.mosesTok = MosesTokenizer(lang=mosesLang)
		self.mosesDetok = MosesDetokenizer(lang=mosesLang)
		self.mosesCase = MosesTruecaser(mosesCaseModel) if mosesCaseModel is not None else None
		self.mosesDecase = MosesDetruecaser()

	def trainTrueCaser(self, fop):
		stream_in = txtStream(fop)
		self.mosesCase = MosesTruecaser()
		self.mosesCase.train(fop)

	def saveTrueCaserModel(self, path):
		self.mosesCase.save_model(path)

	def tokenize(self, fop_in, fop_out):
		stream_in = txtStream(fop_in)
		stream_out = txtStream(fop_out, "wt")

		while True:
			line = stream_in.readline()
			if line == "": break
			line = line.strip()
			if line == "": continue

			line = self.mosesNorm.normalize(line)
			if self.mosesCase: line = self.mosesCase.truecase(line, return_str=True)
			tokens = self.mosesTok.tokenize(line)
			tokens = intersperse(tokens, CTRL_SPACE)

			stream_out.write(str.join(" ", list(tokens)) + "\n")

	def detokenize(self, fop_in, fop_out, unknown="<unk>"):
		stream_in = txtStream(fop_in)
		stream_out = txtStream(fop_out, "wt")

		while True:
			line = stream_in.readline()
			if line == "": break
			line = line.strip()

			line = line.replace(" ", "")
			line = line.replace(CTRL_UNK, unknown)

			tokens = line.split(CTRL_SPACE)
			line = self.mosesDetok.detokenize(tokens)
			if self.mosesCase: line = self.mosesDecase.detruecase(line, return_str=True)

			stream_out.write(line + "\n")

	def splitChar(self, fop_in, fop_out):
		stream_in = txtStream(fop_in)
		stream_out = txtStream(fop_out, "wt")
		
		while True:
			line = stream_in.readline()
			if line == "": break
			line = line.strip()
			if line == "": continue

			line = line.replace(" ", "")
			line = str.join(" ", list(line))
			stream_out.write(line + "\n")

	def mergeChar(self, fop_in, fop_out):
		stream_in = txtStream(fop_in)
		stream_out = txtStream(fop_out, "wt")

		while True:
			line = stream_in.readline()
			if line == "": break
			line = line.strip()

			line = line.replace(" ", "")
			stream_out.write(line + "\n")
