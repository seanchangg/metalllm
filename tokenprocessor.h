#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>

void build_mapping(std::string& input, std::unordered_map<char, uint32_t>& encoder_map, std::unordered_map<uint32_t, char>& decoder_map, uint32_t& vocab_size) {
    for (char& c : input) {
        if (encoder_map.find(c) == encoder_map.end()) {
            encoder_map[c] = vocab_size;
            decoder_map[vocab_size] = c;
            vocab_size++;
        }
    }
}

void encode(std::string& input, std::vector<uint32_t>& output, std::unordered_map<char , uint32_t>& encoder_map) {
    for (char c: input) {
        output.push_back(encoder_map[c]);
    }
};

void decode(std::vector<uint32_t>& input, std::string& output, std::unordered_map<uint32_t ,char >& decoder_map) {
    output = "";
    for (uint32_t i: input) {
        output += decoder_map[i];
    }
};