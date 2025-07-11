// Для работы этого кода нужны зависимости:
// num-complex = "0.4"
// rand = "0.8"

use num_complex::Complex;
use rand::Rng;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::io;




type QuantumState = Vec<Complex<f64>>;




fn power(base: i64, exp: i64, modulus: i64) -> i64 {
    let mut res = 1;
    let mut base = base % modulus;
    let mut exp = exp;
    while exp > 0 {
        if exp % 2 == 1 {
            res = (res * base) % modulus;
        }
        base = (base * base) % modulus;
        exp /= 2;
    }
    res
}



// наибольний общий делитель
fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let temp = a % b;
        a = b;
        b = temp;
    }
    a.abs()
}




fn is_prime(n: i64, k: i32) -> bool {
    if n <= 1 || n == 4 { return false; }
    if n <= 3 { return true; }
    if n % 2 == 0 { return false; }

    let mut d = n - 1;
    while d % 2 == 0 {
        d /= 2;
    }

    let mut rng = rand::thread_rng();
    for _ in 0..k {
        let a = rng.gen_range(2..n - 2);
        let mut x = power(a, d, n);

        if x == 1 || x == n - 1 {
            continue;
        }

        let mut temp_d = d;
        while temp_d != n - 1 {
            x = (x * x) % n;
            if x == 1 { return false; }
            if x == n - 1 { break; }
            temp_d *= 2;
        }
        if x != n - 1 { return false; }
    }
    true
}




fn continued_fractions(x: f64, n: i64) -> i64 {
    if (x - x.floor()).abs() < 1.0 / (2.0 * n as f64) {
        return x.floor() as i64;
    }
    if x.abs() < 1.0e-9 {
        return 0;
    }
    x.floor() as i64 + continued_fractions(1.0 / (x - x.floor()), n)
}



fn hadamard(state: &mut QuantumState, qubit: usize) {
    let n_qubits = (state.len() as f64).log2() as usize;
    // Эта реализация применяет вентиль Адамара к указанному `qubit` (0-индексация от младшего бита, LSB).
    // Исходная версия использовала неконсистентную индексацию, что приводило к ошибке в IQFT.
    let stride = 1 << (qubit + 1);
    let block_size = 1 << qubit;
    let sqrt2_inv = 1.0 / 2.0f64.sqrt();

    // Итерация по блокам состояний, которые затрагивает вентиль.
    for i in 0..(1 << (n_qubits - 1 - qubit)) {
        let block_start = i * stride;
        for j in 0..block_size {
            let idx1 = block_start + j;
            let idx2 = idx1 + block_size;
            let temp = state[idx1];
            state[idx1] = (state[idx1] + state[idx2]) * sqrt2_inv;
            state[idx2] = (temp - state[idx2]) * sqrt2_inv;
        }
    }
}

fn iqft(state: &mut QuantumState) {
    let n = (state.len() as f64).log2() as usize;

    // Bit reversal
    for i in 0..state.len() {
        let reversed_i = i.reverse_bits() >> (usize::BITS as usize - n);
        if i < reversed_i {
            state.swap(i, reversed_i);
        }
    }

    for i in 0..n {
        for j in 0..i {
            let angle = -2.0 * PI / (1 << (i - j + 1)) as f64;
            let phase_shift = Complex::new(angle.cos(), angle.sin());

            // Оптимизация управляемого фазового сдвига.
            // Исходный цикл for k in 0..state.len() итерировал по всему вектору состояний (2^n элементов)
            // для каждой пары кубитов (i, j), что крайне неэффективно (сложность O(n^2 * 2^n)).
            //
            // Новый код напрямую вычисляет индексы состояний `k`, где управляющий кубит `j` и целевой кубит `i`
            // равны 1, и применяет сдвиг только к ним. Это сокращает сложность этой части до O(n^2 * 2^(n-2)),
            // что в 4 раза быстрее для каждого вентиля и значительно ускоряет всю симуляцию.
            let target = i;
            let control = j;

            let target_bit = 1 << target;
            let control_bit = 1 << control;
            let high_stride = 1 << (target + 1);
            let mid_stride = 1 << (control + 1);

            for high_part in 0..(1 << (n - target - 1)) {
                let high_offset = high_part * high_stride;
                for mid_part in 0..(1 << (target - control - 1)) {
                    let mid_offset = mid_part * mid_stride;
                    for low_part in 0..(1 << control) {
                        let index = high_offset + mid_offset + low_part + target_bit + control_bit;
                        state[index] *= phase_shift;
                    }
                }
            }
        }
        hadamard(state, i); // Адамар ПОСЛЕ фазовых сдвигов КРИТИЧЕСКАЯ ОШИБКА была
    }
}

fn measure(state: &QuantumState) -> usize {
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    let mut cumulative_prob = 0.0;
    for (i, amplitude) in state.iter().enumerate() {
        cumulative_prob += amplitude.norm_sqr();
        if r < cumulative_prob {
            return i;
        }
    }
    state.len() - 1
}


fn shors_algorithm(n: i64) {
    if is_prime(n, 5) {
        println!("{} is prime.", n);
        return;
    }

    if n % 2 == 0 {
        println!("Factor found: 2");
        return;
    }

    let mut rng = rand::thread_rng();

    loop {
        let a = rng.gen_range(2..n);
        let common_divisor = gcd(a, n);
        if common_divisor > 1 {
            println!("Factor found classically: {}", common_divisor);
            return;
        }

        let n_bits = (n as f64).log2().ceil() as u32;
        let q_qubits = 2 * n_bits;

        const MAX_SIMULATED_QUBITS: u32 = 128;
        if q_qubits > MAX_SIMULATED_QUBITS {
            println!("Error: The number {} is too large to simulate.", n);
            return;
        }

        let q_size = 1 << q_qubits;

        let mut results_map: HashMap<i64, Vec<i64>> = HashMap::new();
        for i in 0..q_size {
            let res = power(a, i as i64, n);
            results_map.entry(res).or_default().push(i as i64);
        }

        let measured_key_index = rng.gen_range(0..results_map.keys().len());
        let measured_key = *results_map.keys().nth(measured_key_index).unwrap();
        let measured_indices = results_map.get(&measured_key).unwrap();

        let mut state = vec![Complex::new(0.0, 0.0); q_size];
        let amplitude = 1.0 / (measured_indices.len() as f64).sqrt();
        for &idx in measured_indices {
            state[idx as usize] = Complex::new(amplitude, 0.0);
        }

        iqft(&mut state);
        let measurement = measure(&state);

        if measurement == 0 { continue; }

        let phase = measurement as f64 / q_size as f64;
        let r = continued_fractions(phase, n);

        if r == 0 || r % 2 != 0 { continue; }
        
        // Добавим проверку, чтобы избежать херни при вычитании
        let base = power(a, r / 2, n);
        if base == n - 1 { continue; }


        let factor1 = gcd(base + 1, n);
        let factor2 = gcd(base - 1, n);

        if factor1 != 1 && factor1 != n {
            println!("Factor found: {}", factor1);
            return;
        }
        if factor2 != 1 && factor2 != n {
            println!("Factor found: {}", factor2);
            return;
        }
    }
}

fn main() {
    println!("Enter a number to factor (e.g., 15, 21):");
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");

    let n: i64 = match input.trim().parse() {
        Ok(num) => num,
        Err(_) => {
            println!("Please enter a valid number.");
            return;
        }
    };

    if n <= 1 {
        println!("Please enter a number greater than 1.");
        return;
    }

    shors_algorithm(n);
}
