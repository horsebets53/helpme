// Оптимизированная версия алгоритма Шора
// num-complex = "0.4"
// rand = "0.8"
// rayon = "1.7" // для параллельных вычислений

use num_complex::Complex;
use rand::seq::IteratorRandom;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::io;
use std::env;

type QuantumState = Vec<Complex<f64>>;

/// Оптимизированное быстрое возведение в степень по модулю
fn power(base: i64, exp: i64, modulus: i64) -> i64 {
    if modulus == 0 {
        panic!("Модуль не может быть нулевым");
    }
    if modulus == 1 {
        return 0;
    }
    
    let mut res = 1i64;
    let mut base = base % modulus;
    let mut exp = exp;
    while exp > 0 {
        if exp & 1 == 1 {  // Используем битовую операцию вместо % 2
            res = (res * base) % modulus;
        }
        base = (base * base) % modulus;
        exp >>= 1;  // Битовый сдвиг вместо деления на 2
    }
    res
}


// наибольний общий делитель
/// Улучшенный алгоритм Евклида с swap
fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    
    while b != 0 {
        std::mem::swap(&mut a, &mut b);
        b %= a;
    }
    a
}

/// Вычисляет целочисленный квадратный корень с использованием метода Ньютона.
/// # Аргументы
/// * `n` - Неотрицательное целое число.
/// # Возвращаемое значение
/// `Result<i64, String>`, содержащий целочисленный квадратный корень или ошибку.
fn integer_sqrt(n: i64) -> Result<i64, String> {
    if n < 0 {
        return Err("Отрицательное число не имеет реального квадратного корня.".to_string());
    }
    if n == 0 {
        return Ok(0);
    }

    // Используем u128 для вычислений, чтобы избежать переполнения.
    let n_u128 = n as u128;
    let mut x = n_u128; // Начальное приближение.

    loop {
        let y = (x + n_u128 / x) / 2;
        if y >= x {
            // Сходимость достигнута. `x` гарантированно помещается в i64,
            // так как sqrt(i64::MAX) помещается в i64.
            return Ok(x as i64);
        }
        x = y;
    }
}

/// Улучшенный тест простоты с детерминистическими проверками
fn is_prime(n: i64, k: u32) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    
    // Проверка малых делителей до sqrt(n)
    if n < 1000 {
        match integer_sqrt(n) {
            Ok(sqrt_n) => {
                for i in (5..=sqrt_n).step_by(6) {
                    if n % i == 0 || n % (i + 2) == 0 {
                        return false;
                    }
                }
                // Если делители не найдены, число простое.
                return true;
            }
            Err(_) => {
                // Ошибка вычисления корня означает, что что-то не так;
                // безопаснее считать число не простым.
                return false;
            }
        }
    }
    
    // Тест Миллера-Рабина для больших чисел
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    let mut rng = rand::thread_rng();
    'witness_loop: for _ in 0..k {
        let a = rng.gen_range(2..n - 1);
        let mut x = power(a, d, n);

        if x == 1 || x == n - 1 {
            continue;
        }

        for _ in 0..(r - 1) {
            x = power(x, 2, n);
            if x == 1 {
                return false;
            }
            if x == n - 1 {
                continue 'witness_loop;
            }
        }
        return false;
    }
    true
}

/// Оптимизированная и распараллеленная реализация вентиля Адамара.
/// Эта версия исправляет логическую ошибку в предложенной "упрощённой" реализации,
/// сохраняя при этом корректный блочный подход, который правильно работает для любого кубита.
///
/// # Примечание об эффективности
///
/// Использование `rayon` для распараллеливания может быть не всегда эффективно
/// из-за накладных расходов на создание потоков, особенно на малых квантовых состояниях.
/// Рекомендуется использовать профилирование для оценки реального ускорения.
fn hadamard(state: &mut QuantumState, qubit: usize) {
    let n_qubits = (state.len() as f64).log2() as usize;
    if qubit >= n_qubits {
        panic!("Индекс кубита за границами предела(hadamard)");
    }

    let sqrt2_inv = std::f64::consts::FRAC_1_SQRT_2;
    let stride = 1 << (qubit + 1);
    let block_size = 1 << qubit;

    // Разделяем вектор на чанки, которые можно обрабатывать параллельно.
    // Каждый чанк соответствует одному "блоку" в классической итерации.
    state
        .par_chunks_mut(stride)
        .for_each(|chunk| {
            let (first_half, second_half) = chunk.split_at_mut(block_size);
            for i in 0..block_size {
                let a = first_half[i];
                let b = second_half[i];
                first_half[i] = (a + b) * sqrt2_inv;
                second_half[i] = (a - b) * sqrt2_inv;
            }
        }
    );
}

/// Оптимизированная фазовая операция
fn phase_shift(state: &mut QuantumState, control: usize, target: usize, angle: f64) {
    let n_qubits = (state.len() as f64).log2() as usize;
    if control >= n_qubits || target >= n_qubits {
        panic!("Индекс кубита за границами предела(phase_shift)");
    }
    
    let phase = Complex::new(angle.cos(), angle.sin());
    let control_mask = 1 << control;
    let target_mask = 1 << target;
    let combined_mask = control_mask | target_mask;
    
    // Распараллеливаем итерацию по изменяемым элементам вектора.
    // `par_iter_mut()` безопасно предоставляет изменяемый доступ к каждому элементу
    // из разных потоков, так как каждый поток работает с уникальным элементом.
    state.par_iter_mut()
        .enumerate()
        .filter(|(i, _)| (i & combined_mask) == combined_mask)
        .for_each(|(_, amp)| {
            *amp *= phase;
        });
}

/// Улучшенная реализация IQFT с лучшей структурой
fn iqft(state: &mut QuantumState) {
    let n = (state.len() as f64).log2() as usize;

    // Bit reversal - этот шаг должен быть в НАЧАЛЕ для данной реализации IQFT.
    // Перестановка битов подготавливает состояние для каскада вентилей.
    // Операция `swap` здесь безопасна, так как `reverse_bits` гарантирует,
    // что `j` находится в пределах `0..state.len()`. Для дополнительной
    // безопасности используется `split_at_mut`, чтобы избежать любых
    // потенциальных проблем с заимствованием при обмене элементов.
    for i in 0..state.len() {
        let j = reverse_bits(i, n);
        if i < j {
            // Безопасный обмен элементов с помощью `split_at_mut`
            let (left, right) = state.split_at_mut(j);
            std::mem::swap(&mut left[i], &mut right[0]);
        }
    }
    
    // Основная IQFT схема
    for i in 0..n {
        for j in 0..i {
            let angle = -2.0 * PI / (1 << (i - j + 1)) as f64;
            phase_shift(state, j, i, angle);
        }
        hadamard(state, i);
    }
}

/// Вспомогательная функция для обращения битов
fn reverse_bits(mut n: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }
    result
}

/// Оптимизированная функция измерения с проверкой нормализации
fn measure(state: &QuantumState) -> Result<usize, String> {
    // Проверка нормализации
    let total_prob: f64 = state.iter().map(|amp| amp.norm_sqr()).sum();
    if (total_prob - 1.0).abs() > 1e-10 {
        return Err(format!("State is not normalized: total probability = {}", total_prob));
    }
    
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    let mut cumulative_prob = 0.0;
    
    for (i, amplitude) in state.iter().enumerate() {
        cumulative_prob += amplitude.norm_sqr();
        if r < cumulative_prob {
            return Ok(i);
        }
    }
    
    Ok(state.len() - 1)
}

/// Улучшенный алгоритм непрерывных дробей
fn get_continued_fraction_denominators(num: i64, den: i64, max_val: i64) -> Vec<i64> {
    let mut results = Vec::new();
    let mut a = num;
    let mut b = den;
    let mut q_prev = 0i64;
    let mut q_curr = 1i64;

    while b != 0 {
        let quotient = a / b;
        let remainder = a % b;
        
        // Проверка на переполнение
        if let Some(next_q) = quotient.checked_mul(q_curr) {
            if let Some(q_next) = next_q.checked_add(q_prev) {
                if q_next <= max_val {
                    results.push(q_next);
                    a = b;
                    b = remainder;
                    q_prev = q_curr;
                    q_curr = q_next;
                } else {
                    break;
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }
    
    results
}

/// Основной алгоритм Шора с улучшениями
///
/// # Примечание о симуляции
///
/// Эта реализация использует "классический оракул" для ускорения. Вместо того,
/// чтобы симулировать сложный квантовый оператор модульного возведения в степень
/// (U_f|x⟩|0⟩ = |x⟩|a^x mod n⟩), она сначала классически вычисляет все значения
/// a^x mod n и группирует их в HashMap. Затем она "симулирует" измерение,
/// выбирая случайный результат, и искусственно создает конечное квантовое состояние.
///
/// Это распространенное и эффективное упрощение, но оно принципиально отличается
/// от того, как работает настоящий квантовый компьютер, который создает единую
/// запутанную суперпозицию всех состояний за одну операцию.
///
fn shors_algorithm(n: i64, force_quantum: bool) -> Result<i64, String> {
    if n <= 1 {
        return Err("Число должно быть больше 1".to_string());
    }
    
    if is_prime(n, 10) {
        return Err(format!("Число {} - простое", n));
    }

    if n % 2 == 0 {
        return Ok(2);
    }
    
    if !force_quantum {
        println!("Выполняются быстрые классические проверки...");
        // Проверка на точные степени
        for k in 2..((n as f64).log2() as i64 + 1) {
            let root = (n as f64).powf(1.0 / k as f64).round() as i64;
            if power(root, k, i64::MAX) == n {
                return Ok(root);
            }
        }

        // Улучшенная классическая проверка: пробное деление на малые простые числа.
        let limit = (n as f64).sqrt() as i64;
        for i in (3..=limit.min(1000)).step_by(2) {
            if n % i == 0 {
                return Ok(i);
            }
        }
    } else {
        println!("Классические проверки пропущены. Принудительный запуск квантовой симуляции.");
    }

    // --- Квантовая часть ---
    // Теперь, когда все быстрые классические проверки провалились,
    // проверяем, можем ли мы вообще запустить квантовую симуляцию.
    let n_bits = (n as f64).log2().ceil() as u32;
    let q_qubits = 2 * n_bits;
    let max_simulated_qubits: u32 = env::var("MAX_QUBITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(26);

    if q_qubits > max_simulated_qubits {
        return Err(format!(
            "Число {} требует {} кубитов для симуляции, что превышает лимит в {}. Классические проверки не нашли множителей.",
            n, q_qubits, max_simulated_qubits
        ));
    }

    let mut rng = rand::thread_rng();
    let max_attempts = 100;

    'attempt_loop: for attempt in 0..max_attempts {
        let a = rng.gen_range(2..n);
        let common_divisor = gcd(a, n);
        if common_divisor > 1 {
            return Ok(common_divisor);
        }

        let q_size = 1 << q_qubits;
        println!("Попытка {}: используем {} qubits, a = {}", attempt + 1, q_qubits, a);

        // --- Начало ОПТИМИЗИРОВАННОЙ и корректной симуляции оракула ---

        // 1. Эффективно группируем все входы 'x' по их результатам 'y = a^x mod n'.
        //    Это позволяет избежать создания огромного промежуточного вектора.
        let mut results_map: HashMap<i64, Vec<usize>> = HashMap::new();
        for x in 0..q_size {
            let y = power(a, x as i64, n);
            results_map.entry(y).or_default().push(x);
        }

        // 2. Симулируем измерение, выбирая случайный ключ (результат 'y') из карты.
        //    Это гарантирует, что мы выбираем только из реально возможных исходов.
        let measured_y = *results_map.keys().choose(&mut rng)
            .ok_or_else(|| "Не удалось выбрать результат из возможных".to_string())?;

        // 3. Получаем список всех входов 'x', которые привели к измеренному 'y'.
        let periodic_inputs = results_map.get(&measured_y).unwrap();

        // 4. Создаем итоговое состояние суперпозиции.
        let amplitude = Complex::new(1.0 / (periodic_inputs.len() as f64).sqrt(), 0.0);
        let mut state = vec![Complex::new(0.0, 0.0); q_size];
        for x in periodic_inputs {
            state[*x] = amplitude;
        }

        // --- Конец ОПТИМИЗИРОВАННОЙ симуляции ---

        // Применение IQFT
        iqft(&mut state);
        
        // Измерение
        let measurement = match measure(&state) {
            Ok(m) => m,
            Err(e) => {
                println!("Ошибка в измерении: {}", e);
                continue;
            }
        };

        if measurement == 0 {
            continue;
        }

        // Поиск периода через непрерывные дроби
        let candidates = get_continued_fraction_denominators(measurement as i64, q_size as i64, n);

        // Перебираем кандидатов в периоды.
        // Логика здесь в том, чтобы использовать первого же "хорошего" кандидата.
        // Если он не срабатывает, вся попытка с текущим 'a' считается неудачной.
        for r in candidates {
            if r == 0 || r % 2 != 0 {
                continue;
            }

            let base = power(a, r / 2, n);
            if base == n - 1 {
                // Это известный случай неудачи для данного 'a'.
                // Нет смысла проверять других кандидатов, переходим к следующей попытке.
                continue 'attempt_loop;
            }
            if base == 1 {
                // Это может означать, что r - четное кратное истинного периода.
                // Попробуем следующего кандидата из списка.
                continue;
            }

            // Найден нетривиальный множитель
            let factor = gcd(base + 1, n);
            if factor != 1 && factor != n {
                return Ok(factor);
            }
            // Если мы дошли сюда, значит, этот кандидат не сработал.
            // Прерываем цикл по кандидатам и переходим к следующей попытке с новым 'a'.
            break;
        }
    }

    Err(format!(
        "Не удалось найти множители для числа {} после {} попыток.",
        n, max_attempts
    ))
}

fn main() {
    println!("Алгоритм Шора");
    println!("Введите число для факторизации (добавьте флаг --force-quantum для пропуска классических проверок):");
    
    let args: Vec<String> = env::args().collect();
    let force_quantum = args.contains(&"--force-quantum".to_string());
    
    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(_) => {},
        Err(e) => {
            println!("Ошибка при чтении строки: {}", e);
            return;
        }
    }

    let n: i64 = match input.trim().parse() {
        Ok(num) => num,
        Err(_) => {
            println!("Пожалуйста, введите корректное число.");
            return;
        }
    };

    match shors_algorithm(n, force_quantum) {
        Ok(factor) => {
            println!("Найден множитель: {}", factor);
            println!("Проверка: {} = {} x {}", n, factor, n / factor);
        }
        Err(e) => {
            println!("Ошибка: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power() {
        assert_eq!(power(2, 10, 1000), 24);
        assert_eq!(power(3, 4, 5), 1);
        assert_eq!(power(7, 3, 13), 5);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(17, 13), 1);
        assert_eq!(gcd(100, 75), 25);
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime(17, 5));
        assert!(is_prime(97, 5));
        assert!(!is_prime(15, 5));
        assert!(!is_prime(91, 5));
    }

    #[test]
    fn test_hadamard() {
        let mut state = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        hadamard(&mut state, 0);
        
        let expected_amp = std::f64::consts::FRAC_1_SQRT_2;
        assert!((state[0].re - expected_amp).abs() < 1e-10);
        assert!((state[1].re - expected_amp).abs() < 1e-10);
    }
}
