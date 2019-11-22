#pragma once
#include <string>
#include <iostream>
#include <stdio.h>

#define OBF_BCF __attribute__((__annotate__("bcf")))
#define OBF_SUB __attribute__((__annotate__("sub")))
#define OBF_FLA __attribute__((__annotate__("fla")))
#define OBF_SPLIT __attribute__((__annotate__("split")))


class Seco{
    public:
        Seco();
        ~Seco();
    public:
        int generate_keypair();
        int burn_pubkey(const std::string& pub_file);
        int encrypt_model(const std::string &model_file, const std::string& key_file,
            int type = 0);
        int read_pubkey_from_chip(unsigned char* c_pub_key);
        int parse_model(const unsigned char* pub_key, const std::string &model_file,
            unsigned char** out, long &out_length);

};
