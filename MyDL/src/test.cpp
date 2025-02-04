#include <iostream>
#include <cstdint>

struct Packet {
    uint8_t type;  // 패킷의 타입 (1: 32비트, 2: 16비트 두 개, 3: 8비트 네 개)
    
    union {
        uint32_t full;    // 32비트 전체 값
        uint16_t half[2]; // 16비트 2개 값
        uint8_t byte[4];  // 8비트 4개 값
    } data;
};

int main() {
    Packet p;

    // 32비트 데이터로 설정
    p.type = 1;
    p.data.full = 0x11223344;

    std::cout << "Packet Type: " << (int)p.type << std::endl;
    std::cout << "Full: " << std::hex << p.data.full << std::endl;

    // 16비트 데이터로 해석
    p.type = 2;
    std::cout << "Half: " << std::hex << p.data.half[0] << " " << p.data.half[1] << std::endl;

    // 8비트 데이터로 해석
    p.type = 3;
    std::cout << "Bytes: ";
    for (int i = 0; i < 4; i++) {
        std::cout << std::hex << (int)p.data.byte[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
