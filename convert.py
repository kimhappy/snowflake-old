import json
import struct
import sys

def read_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        return {
            "rnn_weight_i": data[ "state_dict" ][ "rec.weight_ih_l0" ],
            "rnn_weight_h": data[ "state_dict" ][ "rec.weight_hh_l0" ],
            "rnn_bias_i"  : data[ "state_dict" ][ "rec.bias_ih_l0"   ],
            "rnn_bias_h"  : data[ "state_dict" ][ "rec.bias_hh_l0"   ],
            "dense_weight": data[ "state_dict" ][ "lin.weight"       ],
            "dense_bias"  : data[ "state_dict" ][ "lin.bias"         ]
        }

def to_vec(data):
    vec = []
    for key in ["rnn_bias_i", "rnn_bias_h", "rnn_weight_i", "rnn_weight_h", "dense_bias", "dense_weight"]:
        for item in data[key]:
            if isinstance(item, list):
                for subitem in item:
                    vec.extend(float_to_bytes(subitem))
            else:
                vec.extend(float_to_bytes(item))
    return vec

def float_to_bytes(f):
    return ['0x' + format(i, '02X') for i in struct.pack('f', f)]

def print_hpp(vec, filename, ampname):
    with open(filename, 'w') as file:
        file.write(f"#ifndef SNOWFLAKE_MODEL_{ampname}_HPP\n")
        file.write(f"#define SNOWFLAKE_MODEL_{ampname}_HPP\n\n")
        file.write("#include <cstdint>\n\n")
        file.write("namespace snowflake::model {\n")
        file.write(f"  inline constexpr uint8_t {ampname}[] = {{\n")

        for i in range(0, len(vec), 16):
            line = ', '.join(vec[i: i + 16])
            file.write(f"    {line},\n")

        file.write(f"  }};\n}}\n\n#endif // SNOWFLAKE_MODEL_{ampname}_HPP\n")

if __name__ == "__main__":
    json_file = sys.argv[ 1 ]
    hpp_file  = sys.argv[ 2 ]
    ampname   = sys.argv[ 3 ]

    data = read_json(json_file)
    vec  = to_vec(data)
    print_hpp(vec, hpp_file, ampname)
