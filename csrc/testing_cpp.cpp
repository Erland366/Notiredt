#include <iostream>
#include <string>
#include <typeinfo>
#include <cxxabi.h>
#include <memory>

int main(int argc, char *argv[])
{
    auto s = 3;
    std::string t = "Hello world";
    std::cout << s << std::endl;
    std::cout << t << std::endl;

    int status;
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(typeid(t).name(), nullptr, nullptr, &status),
        std::free};
    std::cout << (status == 0 ? res.get() : typeid(t).name()) << std::endl;
    return 0;
}
