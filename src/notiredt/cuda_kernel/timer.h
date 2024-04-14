#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>
#include <string>

#define CYAN "\033[36m"
#define RESET "\033[0m"
#define GREEN "\033[1;32m"

struct Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
};

void startTime(Timer* timer){
    timer->start = std::chrono::high_resolution_clock::now();
}

void stopTime(Timer* timer){
    timer->end = std::chrono::high_resolution_clock::now();
}

void printElapsedTime(const Timer& timer, const std::string& message, const std::string& color=RESET){
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer.end - timer.start).count();
    std::cout << color << message << ": " << duration << " ms" << RESET << std::endl;
}

#endif