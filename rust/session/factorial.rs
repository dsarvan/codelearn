fn factorial1(n: usize) -> usize {
    let mut f = vec![1; n + 1];
    for i in 1..n + 1 {
        f[i] = i * f[i - 1];
    }
    return f[n];
}

fn factorial2(n: usize) -> i32 {
    let m: i32 = n as i32 + 1;
    let value: Vec<i32> = (1..m).collect();
    value.iter().product::<i32>()
}

fn factorial3(n: usize) -> usize {
    let mut product = 1;
    for i in 1..n + 1 {
        product = product * i;
    }
    product
}

/* ref: Programming Rust, page no: 109 */
fn factorial4(n: usize) -> usize {
    (1..n+1).product()
}

fn main() {
    const N: usize = 12;

    println!("{}", factorial1(N));
    println!("{}", factorial2(N));
    println!("{}", factorial3(N));
    println!("{}", factorial4(N));
}
