#!/usr/bin/env python

import sys
import argparse
import json

from huffman import HuffmanCoding, Tokenizer, txtStream

START_SYMBOL = 0x4e00 # start of the CJK range, up to ~20k consecutive symbols
EXCLUDED = ["\n"]

def main():
	parser = argparse.ArgumentParser("Huffman")
	parser.add_argument("operation", help="Number of symbols", choices=["vocab", "truecase", "compress", "decompress", "tokenize", "detokenize", "char", "dechar"])
	parser.add_argument("-n", help="Number of symbols", type=int, required=False, default=1000, dest="nsymbols")
	parser.add_argument("-i", help="Input file to compute", required=False, default=None, dest="input")
	parser.add_argument("-o", help="Output file", required=False, default=None, dest="output")
	parser.add_argument("-v", help="Vocab file (for compression and decompression)", required=False, default=None, dest="vocab")
	parser.add_argument("-c", help="Path to the Moses truecase model", required=False, default=None, dest="truecase")
	parser.add_argument("-l", help="Moses language", required=False, default="en", dest="mosesLang")
	parser.add_argument("-u", help="Remove unknown token in detokenized text", required=False, action="store_true", dest="noUnk")
	args = parser.parse_args()

	symbols = [chr(i) for i in range(START_SYMBOL, START_SYMBOL + args.nsymbols)]
	h = HuffmanCoding(EXCLUDED, symbols)
	t = Tokenizer(args.mosesLang, args.truecase)

	stream = sys.stdin if args.input is None else open(args.input, "rt", encoding="utf8")
	output = args.output if args.output is not None else sys.stdout

	if args.operation == "vocab":
		h.ingest(stream)
		h.digest()
		h.saveMapping(output)
	elif args.operation == "truecase":
		t.trainTrueCaser(stream)
		t.saveTrueCaserModel(args.output)
	elif args.operation == "compress":
		h.loadVocab(args.vocab)
		h.compress(stream, output)
	elif args.operation == "decompress":
		h.loadVocab(args.vocab)
		h.decompress(stream, output)
	elif args.operation == "tokenize":
		t.tokenize(stream, output)
	elif args.operation == "detokenize":
		t.detokenize(stream, output, "" if args.noUnk else "<unk>")
	elif args.operation == "char":
		t.splitChar(stream, output)
	elif args.operation == "dechar":
		t.mergeChar(stream, output)

if __name__ == "__main__":
	main()
