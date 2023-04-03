import json
import tensorflow as tf
import numpy as np
import math
import re
import chardet
import os


def initialize_TFRecords(tfrecords_filepath, num_tfrecords=10, filename="training"):
    training_writers = []
    for i in range(num_tfrecords):
        training_writers.append(tf.io.TFRecordWriter(tfrecords_filepath + "{}{}.tfrecords".format(filename,i)))
    return training_writers

def create_lookup_table(vocabulary_mapping, num_oov_buckets):
    keys = [k for k in vocabulary_mapping.keys()]
    values = [tf.constant(vocabulary_mapping[k], dtype=tf.int64) for k in keys]

    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=keys,
            values=values
        ),
        num_oov_buckets
    )
    return table

# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_feature(values):
    # Convert list of integers to bytes
    byte_string = bytes(values)
    # Create BytesList from bytes
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_string]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_vocabulary(vocabulary_filepath):
    """
    It reads and stores in a dictionary-like structure the data from the file passed as argument

    Parameters
    ----------
    vocabulary_filepath: str
        JSON-like file

    Return
    ------
    vocabulary_dict: dict
    """
    with open(vocabulary_filepath, "r") as vocab_file:
        vocabulary_dict = json.load(vocab_file)
    return vocabulary_dict

def serialize_mnemonics_example_IDs(mnemonic_IDs, label):
    """
    Creates a tf.Example message ready to be written to a file
    :param mnemonics: str -> "[4,67,109,...,402, 402]"
    :param label: int [0,8]
    :return:
    """
    feature = {
        'opcodes': _bytes_feature(np.array(mnemonic_IDs).tostring()),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_mnemonics_example(mnemonics, label):
    """
    Creates a tf.Example message ready to be written to a file
    :param mnemonics: str -> "push,pop,...,NONE"
    :param label: int [0,8]
    :return:
    """
    feature = {
        'opcodes': _bytes_feature(mnemonics.encode('UTF-8')),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_bytes_example(bytes, label):
    """
    Creates a tf.Example message ready to be written to a file
    :param bytes: str -> "00,FF,...,??,NONE"
    :param label: int [0,8]
    :return:
    """
    feature = {
        'bytes': _bytes_feature(bytes.encode('UTF-8')),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_apis_example(feature_vector, label):
    feature = {
        'APIs': _bytes_feature(feature_vector),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_hydra_example(opcodes, bytes, apis_values, label):
    feature = {
        'opcodes': _bytes_feature(opcodes.encode('UTF-8')),
        'bytes': _bytes_feature(bytes.encode('UTF-8')),
        'APIs': _bytes_feature(apis_values),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def load_parameters(parameters_path):
    """
    It loads the network parameters

    Parameters
    ----------
    parameters_path: str
        File containing the parameters of the network
    """
    with open(parameters_path, "r") as param_file:
        params = json.load(param_file)
    return params

class MetaPHOR:
    def __init__(self, asm_filepath):
        self.asm_filepath = asm_filepath
        self.vocab = {}

    def extract_windows_api_calls(self):
        # Define regular expressions for Windows API calls
        api_regex = re.compile(r'(call|jmp)\s+(\w+)(@.*)?$')
        winapi_regex = re.compile(r'^(A|W|Nt|Zw)[a-zA-Z]+')

        api_calls = set()

        with open(self.asm_filepath, 'r', encoding='KOI8-R') as f:
            for line in f:
                match = api_regex.search(line)
                if match:
                    api_name = match.group(2)
                    if winapi_regex.match(api_name):
                        api_calls.add(api_name)

        return api_calls

    def count_windows_api_calls(self):
        # Define regular expression for Windows API calls
        api_regex = re.compile(r'call\s+(\w+)')

        api_counts = {
            'VirtualAlloc': 0,
            'CreateFile': 0,
            'ReadFile': 0,
            'WriteFile': 0,
            'CloseHandle': 0,
            'GetModuleHandle': 0,
            'GetProcAddress': 0,
            'LoadLibrary': 0,
            'ExitProcess': 0,
            'OpenProcess': 0,
            'CreateProcess': 0,
            'CreateThread': 0,
            'RegOpenKeyEx': 0,
            'RegSetValueEx': 0,
            'RegQueryValueEx': 0,
            'InternetOpen': 0,
            'InternetConnect': 0,
            'HttpOpenRequest': 0,
            'WinExec': 0,
            'ShellExecute': 0
        }

        # with open(self.asm_filepath, 'rb') as f:
        #     encoding = chardet.detect(f.read())['encoding']
        #     print(encoding)

        with open(self.asm_filepath, 'r', encoding='KOI8-R') as f:
            # print("File opened with encoding: ", encoding)
            try: 
                for line in f:
                    try: 
                        match = api_regex.search(line)
                        if match:
                            api_name = match.group(1)
                            if api_name in api_counts:
                                api_counts[api_name] += 1
                    except:
                        print("Error in line: ", line)
                        continue
            except:
                print("Error in file: ", self.asm_filepath)
                return None

        return list(api_counts.values())
    
    def get_hexadecimal_data_as_list(self):
        # with open(self.asm_filepath, 'r', encoding='KOI8-R') as f:
        #     lines = f.readlines()
        hex_data = []

        with open(self.asm_filepath, 'r', encoding='KOI8-R') as asm_file:
            for line in asm_file:
                # Find the hex data in each line using regex
                hex_values = re.findall(r'\b[0-9A-Fa-f]{2}\b', line)
                hex_data.extend(hex_values)

        return hex_data

    def get_opcodes_data_as_list(self, vocab_mapping):
        opcodes = []

        with open(self.asm_filepath, 'r', encoding='KOI8-R') as asm_file:
            for line in asm_file:
                # Find the mnemonic in each line using regex
                opcode_match = re.search(r'\b[A-Za-z]+\b', line[30:]) # 30 is to skip the address and hex data
                if opcode_match:
                    opcodes.append(opcode_match.group())

        return opcodes