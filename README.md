# huffman-tokenizer
Use of Huffman coding to create a new subword tokenizer for NLP and machine translation.

## Usage
The normal pipeline to use this script is:

- `huffman-tokenizer.py truecase [-l LANG] -i /path/to/input.txt -o /path/to/truecase.model`: trains Moses TrueCaser on LANG (default is en), this step is optional.
- `huffman-tokenizer.py tokenize [-l LANG -c /path/to/truecase.model] -i /path/to/input.txt -o /path/to/output.tok`: tokenizes input text with Moses, optionally using the TrueCaser, each token is separated with a special character CTRL_SPACE (U+2420: Unicode Character 'SYMBOL FOR SPACE'), for example "On ␠ the ␠ first ␠ floor".
- `huffman-tokenizer.py vocab [-n NB_SYMBOLS] -i /path/to/input.tok -o /path/to/mapping.json`: builds the Huffman tree using NB_SYMBOLS (default is 1000) and saves the resulting mapping in a json file (needed later for compression/decompression).
- `huffman-tokenizer.py compress -v /path/to/mapping.json -i /path/to/input.tok -o /path/to/output.enc`: compresses the input text using the provided mapping, in the result each token is represented by its Huffman code on the CJK range, all tokens are separated by the CTRL_SPACE character, for example "内函␠六倽␠其".
- `huffman-tokenizer.py char -i /path/to/input.enc -o /path/to/output.char`: splits the input stream by adding space between each character, for example "内 函 ␠ 六 倽 ␠ 其".

Now train the machine learning model on the SOURCE-LANG.char and TARGET-LANG.char files, for example with OpenNMT: the model will consider each character as a token itself. The model learns how to translate char->char files, so the operations have to be reverted in order to get a plain text file that can be compared to original input with sacrebleu or any other scoring method:

- `huffman-tokenizer.py dechar -i /path/to/input.char -o /path/to/output.enc`: converts a char file back, removing spaces between tokens
- `huffman-tokenizer.py decompress -v /path/to/mapping.json -i /path/to/input.enc -o /path/to/output.tok`: decompresses an encoded file back to the tokenized version
- `huffman-tokenizer.py detokenize [-l LANG] -i /path/to/input.tok -o /path/to/output.txt`: detokenizes the input with Moses

The tokenize/detokenize, compress/decompress and char/dechar are mirror operations, they should produce (almost) exact reversible streams. Also note that the `-i` and `-o` arguments are optional and the script will use respectively stdin and stdout if not provided, which makes the script pipe-able:

```echo "hello world" | ./huffman-tokenizer.py tokenize | ./huffman-tokenizer.py compress -v mappings/en.json | ./huffman-tokenizer.py char | ./huffman-tokenizer.py dechar | ./huffman-tokenizer.py decompress -v test/en.vocab | ./huffman-tokenizer.py detokenize```

## Data
Our training and test data come mostly from WMT 2014 (Bojar et al., 2014) and WMT 2019 (Barrault et al., 2019) and include also the JW300 data (Agić and Vulić, 2019). They are hosted as xz compressed files in [Switchdrive](https://drive.switch.ch/index.php/s/vm85Upk2NhB59O9).

## Acknowledgements
Parts of the code of the HuffmanCoding class defined in [huffman.py](./huffman.py) are based on the [huffman-coding](https://github.com/bhrigu123/huffman-coding) code by Mr. Bhrigu Srivastava with his explicit authorization.
