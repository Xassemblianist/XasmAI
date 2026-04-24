#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Kullanim: encoder.exe <ham_metin.txt> <cikti.bin>\n";
        return 1;
    }

    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { 
        std::cout << "Girdi txt dosyasi bulunamadi kanka, yolu kontrol et!\n"; 
        return 1; 
    }

    std::ofstream out(argv[2], std::ios::binary);
    
    char c;
    int token_count = 0;
    
    // Metindeki her harfi XasmAI'nin anlayacağı 32-bit Token ID'sine çevir
    while (in.get(c)) {
        int32_t token = static_cast<int32_t>(static_cast<unsigned char>(c));
        out.write(reinterpret_cast<const char*>(&token), sizeof(int32_t));
        token_count++;
    }

    std::cout << "Gorev Basarili! Toplam " << token_count 
              << " token preslendi ve " << argv[2] << " dosyasina yazildi.\n";
    return 0;
}