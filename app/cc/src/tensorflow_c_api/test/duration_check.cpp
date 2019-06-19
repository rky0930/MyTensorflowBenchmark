// C++ program to illustrate time point 
// and system clock functions 
#include <iostream> 
#include <chrono> 
#include <ctime> 
#include <unistd.h>
// Function to calculate 
// Fibonacci series 
long fibonacci(unsigned n) 
{ 
    if (n < 2) return n; 
    return fibonacci(n-1) + fibonacci(n-2); 
} 
  
int main() 
{ 
    // Using time point and system_clock 
    std::chrono::time_point<std::chrono::system_clock> start, end; 
    std::chrono::duration<double> total_elapsed_seconds; 
    for (int i=0; i<10; i++) {
        start = std::chrono::system_clock::now(); 
        // std::cout << "f(42) = " << fibonacci(42) << '\n'; 
        sleep(1);
        end = std::chrono::system_clock::now(); 
        std::chrono::duration<double> elapsed_seconds = end - start; 
        total_elapsed_seconds = total_elapsed_seconds + elapsed_seconds;
    }
  
    std::time_t end_time = std::chrono::system_clock::to_time_t(end); 
  
    std::cout << "finished computation at " << std::ctime(&end_time) 
              << "elapsed time: " << total_elapsed_seconds.count() /10 << "s\n"; 
} 