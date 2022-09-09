// File: primes.rs
// Name: D.Saravanan
// Date: 26/08/2022

/* Script to print prime numbers */

fn prime(number: i32) -> String {
    /* function to check prime */
    let sqrt_number = f64::sqrt(number.into());
    for i in 2..(sqrt_number as i32) + 1 {
        if number % i == 0 {
            return format!("{} is not a prime number", number);
        }
    }
    return format!("{} is a prime number", number);
}

fn main() {
    for n in 4..30 {
        println!("{}", prime(n));
    }
}
