#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <numeric>
#include <cmath>
#include <map>
#include <algorithm> // Для std::reverse

// Для арифметики с большими числами идеально подошла бы библиотека типа GMP.
// В этом примере мы будем использовать стандартные типы и признаем это ограничение.

// Вспомогательная функция для модульного возведения в степень: (base^exp) % mod
long long power(long long base, long long exp, long long mod) {
    long long res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return res;
}

// Наибольший общий делитель (НОД) с использованием алгоритма Евклида
long long gcd(long long a, long long b) {
    while (b) {
        a %= b;
        std::swap(a, b);
    }
    return a;
}

// Тест на простоту Миллера-Рабина
bool isPrime(long long n, std::mt19937& gen, int k = 5) {
   if (n <= 1 || n == 4) return false;
   if (n <= 3) return true;
   if (n % 2 == 0) return false;

   long long d = n - 1;
   while (d % 2 == 0) {
       d /= 2;
   }

   for (int i = 0; i < k; i++) {
       std::uniform_int_distribution<long long> distrib(2, n - 2);
       long long a = distrib(gen);
       long long x = power(a, d, n);

       if (x == 1 || x == n - 1) {
           continue;
       }

       long long temp_d = d;
       while (temp_d != n - 1) {
           x = (x * x) % n;
           if (x == 1) return false;
           if (x == n - 1) break;
           temp_d *= 2;
       }
       if (x != n - 1) return false;
   }
   return true;
}

// Непрерывные дроби для нахождения периода 'r' из измерения
long long continued_fractions(double x, long long N) {
    if (std::abs(x) < 1.0e-9) return 0;
    long long a = floor(x);
    double x_rem = x - a;
    if (std::abs(x_rem) < 1.0 / (2.0 * N)) {
        return a;
    }
    return a + continued_fractions(1.0 / x_rem, N);
}


// --- Симуляция квантовых вычислений ---

const double PI = 3.14159265358979323846;

// Представляет квантовое состояние (суперпозицию базисных состояний)
using QuantumState = std::vector<std::complex<double>>;

// Применение вентиля Адамара к одному кубиту
void hadamard(QuantumState& state, int qubit_idx) {
    int n_qubits = log2(state.size());
    int k = n_qubits - qubit_idx;
    size_t block_size = 1ULL << (k - 1);
    size_t num_blocks = 1ULL << qubit_idx;

    for (size_t i = 0; i < num_blocks; ++i) {
        for (size_t j = 0; j < block_size; ++j) {
            size_t idx1 = i * 2 * block_size + j;
            size_t idx2 = idx1 + block_size;
            std::complex<double> temp = state[idx1];
            state[idx1] = (state[idx1] + state[idx2]) / sqrt(2.0);
            state[idx2] = (temp - state[idx2]) / sqrt(2.0);
        }
    }
}

// НОВОЕ: Обратное Квантовое Преобразование Фурье (IQFT)
void iqft(QuantumState& state) {
    int n = log2(state.size());
    
    // Перестановка кубитов в начале
    for (size_t i = 0; i < state.size(); ++i) {
        size_t reversed_i = 0;
        size_t temp_i = i;
        for (int j = 0; j < n; ++j) {
            reversed_i = (reversed_i << 1) | (temp_i & 1);
            temp_i >>= 1;
        }
        if (i < reversed_i) {
            std::swap(state[i], state[reversed_i]);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double angle = -2.0 * PI / (1 << (i - j + 1)); // Отрицательный угол для обратного преобразования
            for (size_t k = 0; k < state.size(); ++k) {
                if ((k >> i) & 1 && (k >> j) & 1) { // Условный фазовый сдвиг
                     state[k] *= std::exp(std::complex<double>(0, angle));
                }
            }
        }
        hadamard(state, i);
    }
}


// Измерение квантового состояния и его коллапс
int measure(const QuantumState& state, std::mt19937& gen) {
    std::uniform_real_distribution<double> distrib(0.0, 1.0);
    double r = distrib(gen);
    double cumulative_prob = 0.0;
    for (size_t i = 0; i < state.size(); ++i) {
        cumulative_prob += std::norm(state[i]);
        if (r < cumulative_prob) {
            return i;
        }
    }
    return state.size() - 1;
}

// --- Реализация алгоритма Шора ---

void shors_algorithm(long long N) {
    std::random_device rd;
    std::mt19937 gen(rd());

    if (isPrime(N, gen)) {
        std::cout << N << " is prime." << std::endl;
        return;
    }

    if (N % 2 == 0) {
        std::cout << "Factor: 2" << std::endl;
        return;
    }

    while (true) {
        std::uniform_int_distribution<long long> distrib(2, N - 1);
        long long a = distrib(gen);

        long long common_divisor = gcd(a, N);
        if (common_divisor > 1) {
            std::cout << "Factor found classically: " << common_divisor << std::endl;
            return;
        }

        int n_qubits = ceil(log2(N));
        int q_qubits = 2 * n_qubits;

        const int MAX_SIMULATED_QUBITS = 128; // Уменьшим лимит для стабильности
        if (q_qubits > MAX_SIMULATED_QUBITS) {
            std::cout << "Error: The number " << N << " is too large to simulate on a classical computer." << std::endl;
            std::cout << "It requires simulating " << q_qubits << " qubits, which is beyond the practical limit of " << MAX_SIMULATED_QUBITS << "." << std::endl;
            return;
        }
        
        size_t q_size = 1ULL << q_qubits;
        
        // --- ИЗМЕНЕННЫЙ БЛОК: Симуляция модульного возведения в степень ---
        // 1. Вычисляем f(x) = a^x mod N для всех x и группируем x по результатам.
        std::map<long long, std::vector<long long>> results_map;
        for (size_t i = 0; i < q_size; ++i) {
            long long res = power(a, i, N);
            results_map[res].push_back(i);
        }

        // 2. Симулируем измерение второго регистра, случайно выбирая один из результатов f(x).
        std::uniform_int_distribution<size_t> map_distrib(0, results_map.size() - 1);
        auto it = results_map.begin();
        std::advance(it, map_distrib(gen));
        const std::vector<long long>& measured_indices = it->second;

        // 3. Создаем новое состояние, которое коллапсировало в периодическую суперпозицию.
        QuantumState state(q_size, {0.0, 0.0});
        double amplitude = 1.0 / sqrt(measured_indices.size());
        for (long long idx : measured_indices) {
            state[idx] = {amplitude, 0.0};
        }
        // --- КОНЕЦ ИЗМЕНЕННОГО БЛОКА ---


        // 4. Применяем Обратное Квантовое Преобразование Фурье (IQFT)
        iqft(state);

        // 5. Измеряем состояние
        int measurement = measure(state, gen);

        if (measurement == 0) continue;

        double phase = (double)measurement / q_size;
        long long r = continued_fractions(phase, N);

        if (r == 0) continue;

        if (r % 2 != 0) {
            continue;
        }

        long long factor1 = gcd(power(a, r / 2, N) - 1, N);
        long long factor2 = gcd(power(a, r / 2, N) + 1, N);

        if (factor1 != 1 && factor1 != N) {
            std::cout << "Factor found: " << factor1 << std::endl;
            return;
        }
        if (factor2 != 1 && factor2 != N) {
            std::cout << "Factor found: " << factor2 << std::endl;
            return;
        }
    }
}

int main() {
    long long N;
    std::cout << "Enter a number to factor (e.g., 15, 21): ";
    std::cin >> N;

    if (N <= 1) {
        std::cout << "Please enter a number greater than 1." << std::endl;
        return 1;
    }
    
    shors_algorithm(N);

    return 0;
}
