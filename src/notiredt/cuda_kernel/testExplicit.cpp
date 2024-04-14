#include <stdio.h>
#include <iostream>

using namespace std;

struct Example {
    int value;

    explicit Example(int x) : value(x) {}

    friend ostream& operator<<(ostream& os, const Example& obj){
        os << obj.value; // Print the value of the object
        return os;
    }
};

int main(int argc, char** argv){
    // Example obj = 20;
    Example obj2(10);
    cout << "The value of obj is: " << obj2 << endl;
    return 0;
}
