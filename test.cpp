#include <iostream>
#include <cstdint>
#include <cstring>

struct PacketHeader {
    uint8_t type;  // 패킷 종류 (1: 로그인, 2: 이동, 3: 공격)
    uint16_t size; // 패킷 크기
};

union PacketData {
    struct {
        char username[16];
        char password[16];
    } login;

    struct {
        float x;
        float y;
    } move;

    struct {
        uint32_t attackPower;
    } attack;
};

struct Packet {
    PacketHeader header;
    PacketData data;
};

int main() {
    Packet packet;

    // 로그인 패킷 설정
    packet.header.type = 1;
    packet.header.size = sizeof(packet.data.login);
    std::strcpy(packet.data.login.username, "Player1");
    std::strcpy(packet.data.login.password, "1234");

    std::cout << "Packet Type: " << (int)packet.header.type << std::endl;
    std::cout << "Username: " << packet.data.login.username << std::endl;
    std::cout << "Password: " << packet.data.login.password << std::endl;

    return 0;
}
